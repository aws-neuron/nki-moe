"""Driver script for NKI contests - Extended with inference_demo.py arguments"""

import argparse
import ast
import base64
import copy
import csv

# use this to load the local qwen
import importlib
import json
import os
import time
from enum import Enum

import torch
from neuronx_distributed_inference.models.config import (
    OnDeviceSamplingConfig,
    to_torch_dtype,
)

# load the baseline model
from neuronx_distributed_inference.models.qwen3_moe import (
    modeling_qwen3_moe as baseline_qwen,
)
from neuronx_distributed_inference.modules.generation.sampling import (
    prepare_sampling_params,
)
from neuronx_distributed_inference.utils import argparse_utils
from neuronx_distributed_inference.utils.accuracy import get_generate_outputs
from neuronx_distributed_inference.utils.benchmark import (
    Benchmark,
    create_submodule_latency_collectors,
    generate_report,
    register_latency_collectors,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)
from neuronx_distributed_inference.utils.random import set_random_seed
from torch_neuronx.pyhlo.hlo_pb2 import HloModuleProto
from torch_neuronx.testing.validation import logit_validation
from transformers import AutoTokenizer, GenerationConfig

from test import parse_prompt_data, parse_prompts


class QuantizationType(Enum):
    PER_TENSOR_SYMMETRIC = "per_tensor_symmetric"
    PER_CHANNEL_SYMMETRIC = "per_channel_symmetric"
    BLOCKWISE_SYMMETRIC = "blockwise_symmetric"
    EXPERT_WISE_PER_CHANNEL_SYMMETRIC = "expert_wise_per_channel_symmetric"


class ActivationQuantizationType(Enum):
    DYNAMIC = "dynamic"
    NONE = None


BENCHMARK_REPORT_FILENAME = "benchmark_report.json"

set_random_seed(0)


def parse_args():
    parser = argparse.ArgumentParser()

    # contest specific
    parser.add_argument(
        "--mode",
        choices=[
            "evaluate_single",
            "evaluate_all",
            "validate",
            "generate",
        ],
    )
    parser.add_argument("--qwen", type=str, default="qwen")
    parser.add_argument("--enable-nki", action="store_true")
    parser.add_argument("--base-latency", type=float, default=526.15)
    parser.add_argument("--base-throughput", type=float, default=134.61)
    # new arguments for the leaderboard
    parser.add_argument(
        "--team-id", type=str, help="Team identifier for score tracking"
    )
    parser.add_argument(
        "--member-id", type=str, help="Team member identifier for score tracking"
    )

    # Model path
    parser.add_argument(
        "--model-path", type=str, default="/home/ubuntu/Qwen3-30B-A3B/hf_model"
    )
    parser.add_argument(
        "--compiled-model-path",
        type=str,
        default="/home/ubuntu/Qwen3-30B-A3B/traced_model",
    )

    # Evaluation
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--divergence-difference-tol", type=float, default=0.001)
    parser.add_argument("--tol-map", type=str)
    parser.add_argument("--num-tokens-to-check", type=int)

    # Generation
    parser.add_argument("--prompt", dest="prompts", action="append")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--global-topk", type=int)
    parser.add_argument("--do-sample", type=bool, default=True)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--pad-token-id", type=int, default=2)
    parser.add_argument("--top-k-kernel-enabled", action="store_true", default=False)

    # Basic config
    parser.add_argument("--torch-dtype", type=to_torch_dtype, default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--padding-side", type=str)
    parser.add_argument("--allow-input-truncation", action="store_true")
    parser.add_argument("--seq-len", type=int, default=640)
    parser.add_argument("--n-active-tokens", type=int)
    parser.add_argument("--n-positions", type=int)
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--rpl-reduce-dtype", type=to_torch_dtype)
    parser.add_argument("--attention-dtype", type=to_torch_dtype)
    parser.add_argument("--output-logits", action="store_true")
    parser.add_argument("--vocab-parallel", action="store_true")
    parser.add_argument("--layer-boundary-markers", action="store_true", default=False)
    parser.add_argument("--platform-target", type=str, default="trn2")

    # Attention
    parser.add_argument("--fused-qkv", action="store_true")
    parser.add_argument("--sequence-parallel-enabled", action="store_true")
    parser.add_argument("--weight-gather-seq-len-threshold", type=int)
    parser.add_argument("--flash-decoding-enabled", action="store_true")

    # Continuous batching
    parser.add_argument("--ctx-batch-size", type=int)
    parser.add_argument("--tkg-batch-size", type=int)
    parser.add_argument("--max-batch-size", type=int)
    parser.add_argument("--is-continuous-batching", action="store_true")

    # KV cache
    parser.add_argument("--kv-cache-batch-size", type=int)
    parser.add_argument("--kv-cache-padding-size", type=int)
    parser.add_argument("--disable-kv-cache-tiling", action="store_true")

    # On device sampling
    parser.add_argument("--on-device-sampling", action="store_true")

    # Bucketing
    parser.add_argument("--enable-bucketing", type=bool, default=True)
    parser.add_argument("--bucket-n-active-tokens", action="store_true")
    parser.add_argument("--context-encoding-buckets", nargs="+", type=int)
    parser.add_argument("--prefix-buckets", nargs="+", type=int)
    parser.add_argument("--token-generation-buckets", nargs="+", type=int)

    # Quantization
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--quantized-checkpoints-path", type=str)
    parser.add_argument(
        "--quantization-type", type=str, choices=[t.value for t in QuantizationType]
    )
    parser.add_argument("--kv-cache-quant", action="store_true")
    parser.add_argument("--quantization-dtype", type=str)
    parser.add_argument(
        "--modules-to-not-convert-file",
        type=get_modules_to_not_convert_json,
        dest="modules_to_not_convert_lists",
    )

    # MoE
    parser.add_argument("--capacity-factor", type=float)
    parser.add_argument("--early-expert-affinity-modulation", action="store_true")
    parser.add_argument("--disable-normalize-top-k-affinities", action="store_true")
    parser.add_argument("--fused-shared-experts", action="store_true")

    # Router Config
    parser.add_argument("--router-act-fn", type=str)
    parser.add_argument("--router-dtype", type=str)

    # Speculative decoding
    parser.add_argument("--draft-model-path", type=str)
    parser.add_argument("--draft-model-tp-degree", type=int, default=None)
    parser.add_argument("--compiled-draft-model-path", type=str)
    parser.add_argument(
        "--enable-fused-speculation", action="store_true", default=False
    )
    parser.add_argument(
        "--enable-eagle-speculation", action="store_true", default=False
    )
    parser.add_argument(
        "--enable-eagle-draft-input-norm", action="store_true", default=False
    )
    parser.add_argument("--speculation-length", type=int)
    parser.add_argument("--spec-batch-size", type=int)

    # Medusa decoding
    parser.add_argument("--is-medusa", action="store_true")
    parser.add_argument("--medusa-speculation-length", type=int)
    parser.add_argument("--num-medusa-heads", type=int)
    parser.add_argument("--medusa-tree-json", type=load_json_file, dest="medusa_tree")

    # Token Tree
    parser.add_argument(
        "--token-tree-json", type=load_json_file, dest="token_tree_config"
    )

    # Parallelism
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--cp-degree", type=int)
    parser.add_argument("--mlp-cp-degree", type=int)
    parser.add_argument("--attention-dp-degree", type=int)
    parser.add_argument("--pp-degree", type=int)
    parser.add_argument("--ep-degree", type=int)
    parser.add_argument("--moe-tp-degree", type=int, default=1)
    parser.add_argument("--moe-ep-degree", type=int, default=1)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--start_rank_id", type=int)
    parser.add_argument("--local_ranks_size", type=int)
    parser.add_argument("--enable-torch-dist", action="store_true")
    parser.add_argument("--save-sharded-checkpoint", action="store_true")
    parser.add_argument("--skip-sharding", action="store_true")

    # PA and CF
    parser.add_argument(
        "--enable-block-kv-layout", dest="is_block_kv_layout", action="store_true"
    )
    parser.add_argument("--pa-num-blocks", type=int)
    parser.add_argument("--pa-block-size", type=int)
    parser.add_argument(
        "--enable-chunked-prefill", dest="is_chunked_prefill", action="store_true"
    )
    parser.add_argument(
        "--enable-prefix-caching", dest="is_prefix_caching", action="store_true"
    )
    parser.add_argument("--max-num-seqs", type=int)

    # Async
    parser.add_argument("--async-mode", action="store_true")

    # Windowed Context Encoding
    parser.add_argument("--windowed-context-encoding-size", type=int)

    # Lora
    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--enable-dynamic-multi-lora", action="store_true")
    parser.add_argument("--max-loras", type=int, default=1)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--target-modules", nargs="+")
    parser.add_argument("--max-cpu-loras", type=int, default=1)
    parser.add_argument(
        "--lora-ckpt-path", dest="lora_ckpt_paths", type=str, action="append"
    )
    parser.add_argument(
        "--lora-ckpt-path-cpu", dest="lora_ckpt_paths_cpu", type=str, action="append"
    )
    parser.add_argument(
        "--lora-ckpt-json", dest="lora_ckpt_json", type=str, default=None
    )
    parser.add_argument("--adapter-id", dest="adapter_ids", type=str, action="append")

    # Kernels
    parser.add_argument("--qkv-kernel-enabled", action="store_true")
    parser.add_argument("--qkv-nki-kernel-enabled", action="store_true")
    parser.add_argument("--qkv-cte-nki-kernel-fuse-rope", action="store_true")
    parser.add_argument("--qkv-kernel-nbsd-layout", action="store_true")
    parser.add_argument("--attn-kernel-enabled", action="store_true")
    parser.add_argument(
        "--strided-context-parallel-kernel-enabled", action="store_true"
    )
    parser.add_argument("--mlp-kernel-enabled", action="store_true")
    parser.add_argument("--mlp-tkg-nki-kernel-enabled", action="store_true")
    parser.add_argument("--quantized-mlp-kernel-enabled", action="store_true")
    parser.add_argument("--fused-rmsnorm-skip-gamma", action="store_true")
    parser.add_argument(
        "--activation-quantization-type",
        type=str,
        choices=[e.value for e in ActivationQuantizationType],
    )
    parser.add_argument("--rmsnorm-quantize-kernel-enabled", action="store_true")
    parser.add_argument("--quantize-clamp-bound", type=float, default=float("inf"))
    parser.add_argument("--mlp-kernel-fuse-residual-add", action="store_true")
    parser.add_argument("--qkv-kernel-fuse-residual-add", action="store_true")
    parser.add_argument("--attn-tkg-nki-kernel-enabled", action="store_true")
    parser.add_argument("--attn-tkg-builtin-kernel-enabled", action="store_true")
    parser.add_argument("--attn-block-tkg-nki-kernel-enabled", action="store_true")
    parser.add_argument(
        "--attn-block-tkg-nki-kernel-cascaded-attention", action="store_true"
    )
    parser.add_argument("--attn-block-tkg-nki-kernel-cache-update", action="store_true")
    parser.add_argument("--attn-block-cte-nki-kernel-enabled", action="store_true")
    parser.add_argument("--k-cache-transposed", action="store_true")
    parser.add_argument("--is-eagle3", action="store_true")

    # Logical NeuronCore Configuration (LNC)
    parser.add_argument("--logical-neuron-cores", type=int)
    parser.add_argument("--logical-nc-config", type=int)

    # Compiler Args
    parser.add_argument("--cc-pipeline-tiling-factor", type=int, default=2)
    parser.add_argument("--enable-spill-reload-dge", action="store_true")
    parser.add_argument("--scratchpad-page-size", type=int)

    # CPU
    parser.add_argument("--on-cpu", action="store_true")

    # Report generation
    parser.add_argument(
        "--benchmark-report-path", type=str, default=BENCHMARK_REPORT_FILENAME
    )

    # Debugging
    parser.add_argument(
        "--capture-indices",
        nargs="+",
        type=int,
        action=argparse_utils.StringOrIntegers,
        default=None,
    )
    parser.add_argument("--input-capture-save-dir", type=str, default=None)
    parser.add_argument(
        "--cast-type", choices=["config", "as-declared"], default="config"
    )

    # Optional demo arguments
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--compile-dry-run", action="store_true")
    parser.add_argument("--hlo-debug", action="store_true")
    parser.add_argument("--apply-seq-ids-mask", action="store_true")
    parser.add_argument("--input-start-offsets", nargs="+", default=None, type=int)
    parser.add_argument("--enable-output-completion-notifications", action="store_true")

    return parser.parse_args()


def load_json_file(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_modules_to_not_convert_json(json_path):
    modules_to_not_convert, draft_model_modules_to_not_convert = None, None
    assert os.path.exists(json_path), f"File not found: {json_path}"
    data = load_json_file(json_path)
    if "model" in data:
        modules_to_not_convert = data["model"]["modules_to_not_convert"]
    elif "modules_to_not_convert" in data:
        modules_to_not_convert = data["modules_to_not_convert"]
    # Handle draft model modules if they exist
    if "draft_model" in data:
        draft_model_modules_to_not_convert = data["draft_model"][
            "modules_to_not_convert"
        ]
    return modules_to_not_convert, draft_model_modules_to_not_convert


def load_tokenizer(model_path, compiled_model_path, neuron_config):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(compiled_model_path)
    return tokenizer


def prepare_inference(model_cls, args):
    # Initialize configs.
    print("Loading inference configs...")

    # Skip values not specified in the args to avoid setting values to None in the config.
    config_kwargs = copy.deepcopy(vars(args))
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

    if args.on_device_sampling:
        config_kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(
            **config_kwargs
        )

    neuron_config = model_cls.get_neuron_config_cls()(**config_kwargs)

    config = model_cls.get_config_cls()(
        neuron_config, load_config=load_pretrained_config(args.model_path)
    )

    model = model_cls(args.model_path, config)

    if not args.skip_compile:
        # Compile and save model.
        # to do, add save sharded checkpoint here
        compiling_start_time = time.monotonic()
        print("\nCompiling and saving model...")
        model.compile(
            args.compiled_model_path,
            debug=args.hlo_debug if hasattr(args, "hlo_debug") else False,
        )

        compiling_end_time = time.monotonic()
        total_compiling_time = compiling_end_time - compiling_start_time
        print(f"Compiling and tracing time: {total_compiling_time} seconds")

    # Load compiled model to Neuron.
    print("\nLoading model to Neuron...")
    model.load(args.compiled_model_path)

    # Load tokenizer.
    tokenizer = load_tokenizer(args.model_path, args.compiled_model_path, neuron_config)
    neuron_config.pad_token_id = tokenizer.pad_token_id

    # Configure generation config.
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    generation_config_args = [
        "do_sample",
        "top_k",
        "pad_token_id",
        "dynamic",
        "top_p",
        "temperature",
    ]
    generation_config_kwargs = {
        k: getattr(args, k)
        for k in generation_config_args
        if getattr(args, k) is not None
    }
    generation_config.update(**generation_config_kwargs)

    return model, tokenizer, generation_config


def generate_submodule_reports(latency_collectors, neuron_config, num_runs):
    reports = {}
    for key, collector in latency_collectors.items():
        tokens_len = neuron_config.max_length
        if key == "context_encoding_model":
            tokens_len = neuron_config.seq_len - neuron_config.max_new_tokens
        elif key == "token_generation_model":
            tokens_len = neuron_config.max_new_tokens
        reports[key] = generate_report(
            collector.latency_list, tokens_len, neuron_config.max_batch_size, num_runs
        )
    return reports


def benchmark_sampling(model, tokenizer, generation_config, prompts):

    print("Beginning benchmark sampling")

    neuron_config = model.neuron_config

    sampling_params = prepare_sampling_params(
        batch_size=neuron_config.batch_size,
        top_k=generation_config.top_k
        if isinstance(generation_config.top_k, list)
        else [generation_config.top_k],
        top_p=generation_config.top_p
        if isinstance(generation_config.top_p, list)
        else [generation_config.top_p],
        temperature=generation_config.temperature
        if isinstance(generation_config.temperature, list)
        else [generation_config.temperature],
    )

    report = {}

    # on_device_sampling flow does not support min_new_tokens
    # to override eos_tokens so we remove EOS tokens to ensure
    # token generation happens.
    modified_generation_config = copy.deepcopy(generation_config)
    if model.on_device_sampling:
        modified_generation_config.eos_token_id = []

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    neuron_config.max_new_tokens = neuron_config.seq_len - input_ids.shape[1]

    input_param = {
        "input_ids": input_ids,
        "generation_config": modified_generation_config,
        "attention_mask": attention_mask,
        "min_new_tokens": neuron_config.max_new_tokens,
        "max_new_tokens": neuron_config.max_new_tokens,
        "top_k": 1,
        "do_sample": not neuron_config.enable_fused_speculation,
        "sampling_params": sampling_params,
        "max_length": neuron_config.max_length
        if neuron_config.max_new_tokens is None
        else None,
    }

    latency_collectors = create_submodule_latency_collectors(model)

    def post_warmup_func():
        register_latency_collectors(latency_collectors, model)

    # Register latency collectors after warm-up to avoid recording warm-up metrics.
    generation_model = HuggingFaceGenerationAdapter(model)
    e2e_benchmark = Benchmark(
        generation_model.generate,
        input_param,
        preprocess_func=model.reset,
        post_warmup_func=post_warmup_func,
    )
    e2e_benchmark.run()
    report["e2e_model"] = generate_report(
        e2e_benchmark.latency_list,
        neuron_config.max_length,
        neuron_config.max_batch_size,
        n_runs=e2e_benchmark.num_runs,
    )

    report.update(
        generate_submodule_reports(
            latency_collectors, neuron_config, e2e_benchmark.num_runs
        )
    )

    model.reset()

    print("Benchmark completed and its result is as following")
    print(json.dumps(report, indent=4))
    with open(BENCHMARK_REPORT_FILENAME, "w") as f:
        json.dump(report, f)
    print("Completed saving result to " + BENCHMARK_REPORT_FILENAME)

    return report


def check_accuracy_logits(
    base_model,
    base_generation_config,
    neuron_model,
    tokenizer,
    generation_config,
    prompts,
    divergence_difference_tol,
    tol_map,
    num_tokens_to_check,
):
    assert prompts is not None

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    initial_input_ids = inputs.input_ids
    initial_attention_mask = inputs.attention_mask
    seq_len = neuron_model.config.neuron_config.seq_len

    neuron_model.config.neuron_config.max_new_tokens = (
        seq_len - initial_input_ids.shape[1]
    )

    model = HuggingFaceGenerationAdapter(base_model)
    new_tokens = neuron_model.config.neuron_config.max_new_tokens
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=initial_input_ids,
            attention_mask=initial_attention_mask,
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            generation_config=base_generation_config,
        )
    expected_logits = torch.stack(outputs.scores)

    if num_tokens_to_check is not None:
        print(f"Validating logits for first {num_tokens_to_check} tokens")
        expected_logits = expected_logits[:num_tokens_to_check, :, :]

    expected_token_ids = expected_logits.argmax(dim=2).T
    expected_tokens = tokenizer.batch_decode(
        expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Expected Output: ", expected_tokens, expected_token_ids)
    print("Expected Logits Shape: ", expected_logits.shape)

    model = HuggingFaceGenerationAdapter(neuron_model)
    expected_attention_mask = torch.ones(
        (
            initial_attention_mask.shape[0],
            expected_token_ids.shape[1],
        ),
        dtype=torch.int32,
    )
    extrapolated_attention_mask = torch.cat(
        (initial_attention_mask, expected_attention_mask), dim=1
    )

    def generate_fn(input_ids):
        input_length = input_ids.shape[1]
        attention_mask = extrapolated_attention_mask[:, :input_length]
        with torch.inference_mode():
            model_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=seq_len - input_length,
                min_new_tokens=seq_len - input_length,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                generation_config=generation_config,
            )

        actual_logits = torch.stack(model_outputs.scores)
        actual_token_ids = actual_logits.argmax(dim=2).T
        actual_tokens = tokenizer.batch_decode(
            actual_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print("Actual Output: ", actual_tokens, actual_token_ids)
        print("Actual Logits Shape: ", actual_logits.shape)
        return torch.stack(model_outputs.scores)

    passed, _, status_msg = logit_validation(
        input_ids=initial_input_ids,
        generate_fn=generate_fn,
        expected_logits=expected_logits,
        tol_map=tol_map,
        divergence_difference_tol=divergence_difference_tol,
    )
    print("STATUS MSG", status_msg)
    assert passed, status_msg

    print("Passed logits validation")


def run_generation(model, tokenizer, prompts, generation_config):
    print("\nGenerating outputs...")
    print(f"Prompts: {prompts}")

    _, output_tokens = get_generate_outputs(
        model,
        prompts,
        tokenizer,
        is_hf=False,
        generation_config=generation_config,
        max_length=model.neuron_config.max_length,
    )

    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


def run_accuracy_check(
    base_model,
    base_generation_config,
    model,
    tokenizer,
    generation_config,
    prompt,
    divergence_difference_tol,
    tol_map,
    num_tokens_to_check=None,
):
    if tol_map:
        tol_map = ast.literal_eval(tol_map)

    try:
        check_accuracy_logits(
            base_model=base_model,
            base_generation_config=base_generation_config,
            neuron_model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            prompts=prompt,
            divergence_difference_tol=divergence_difference_tol,
            tol_map=tol_map,
            num_tokens_to_check=num_tokens_to_check,
        )
    except AssertionError:
        return False

    return True


def count_nki_flop_ratio(hlo_path_context_enc, hlo_path_token_gen):
    hlo_macs = 0
    nki_macs = 0

    def parse_hlo_file(hlo_file_path):
        with open(hlo_file_path, "rb") as f:
            hlo_data = f.read()

        hlo_proto = HloModuleProto()
        hlo_proto.ParseFromString(hlo_data)
        return hlo_proto

    def count_mac(hlo_proto):
        nki_mac = 0
        hlo_mac = 0

        for computation in hlo_proto.computations:
            instruction_map = {instr.id: instr for instr in computation.instructions}

            for instruction in computation.instructions:
                # Finding NKI ops
                if instruction.opcode == "custom-call":
                    if instruction.custom_call_target == "AwsNeuronCustomNativeKernel":
                        try:
                            backend_config = instruction.backend_config
                            config = json.loads(base64.b64decode(backend_config))
                            mac_count = int(config["mac_count"])
                        except Exception:
                            mac_count = 0

                        nki_mac += mac_count
                        hlo_mac += mac_count
                elif instruction.opcode == "dot":
                    # Get dot dimension numbers
                    dnums = instruction.dot_dimension_numbers

                    # Get shapes of operands using operand_ids
                    lhs_shape = instruction_map[instruction.operand_ids[0]].shape
                    rhs_shape = instruction_map[instruction.operand_ids[1]].shape

                    # Initialize counters
                    lhs_batch = 1
                    lhs_contracting_size = 1
                    lhs_non_contracting_size = 1
                    rhs_non_contracting_size = 1

                    # Process LHS shape
                    for i in range(len(lhs_shape.dimensions)):
                        if i in dnums.lhs_contracting_dimensions:
                            lhs_contracting_size *= lhs_shape.dimensions[i]
                        elif i in dnums.lhs_batch_dimensions:
                            lhs_batch *= lhs_shape.dimensions[i]
                        else:
                            lhs_non_contracting_size *= lhs_shape.dimensions[i]

                    # Process RHS shape
                    for i in range(len(rhs_shape.dimensions)):
                        if (
                            i not in dnums.rhs_contracting_dimensions
                            and i not in dnums.rhs_batch_dimensions
                        ):
                            rhs_non_contracting_size *= rhs_shape.dimensions[i]

                    mac_count = (
                        lhs_batch
                        * lhs_non_contracting_size
                        * lhs_contracting_size
                        * rhs_non_contracting_size
                    )
                    hlo_mac += mac_count

        return hlo_mac, nki_mac

    hlo_proto_context_enc = parse_hlo_file(hlo_path_context_enc)
    hlo_proto_token_gen = parse_hlo_file(hlo_path_token_gen)
    hlo_mac_context_enc, nki_mac_context_enc = count_mac(hlo_proto_context_enc)
    hlo_mac_token_gen, nki_mac_token_gen = count_mac(hlo_proto_token_gen)

    # FIXME: Need to consider token gen get executed more
    hlo_macs = hlo_mac_context_enc + hlo_mac_token_gen
    nki_macs = nki_mac_context_enc + nki_mac_token_gen

    if hlo_macs == 0:
        assert nki_macs == 0
        nki_flop_ratio = 0
    else:
        nki_flop_ratio = nki_macs / hlo_macs

    return nki_flop_ratio


# Added team_id, member_id and other OPTIONAL input parameters for connecting metric scores to team that submitted their nki-moe script
def calculate_score(
    base_latency,
    base_throughput,
    accuracy,
    latency,
    throughput,
    nki_flop_ratio,
    team_id=None,
    member_id=None,
    qwen_module=None,
    platform_target=None,
):

    increased_throughput = throughput / base_throughput
    reduced_latency = base_latency / latency

    # resetting nki_flop_ratio as the baseline solution uses NKI completely
    final_score = accuracy * reduced_latency * increased_throughput * nki_flop_ratio

    print(
        "In this final score of ",
        final_score,
        " the contestant got a breakdown as follows.",
    )
    print("accuracy: ", accuracy)
    print("reduced_latency: ", reduced_latency)
    print("increased throughput: ", increased_throughput)
    print("nki flop ratio: ", nki_flop_ratio)

    # Write parameters to CSV file based on team_id and qwen_module identifier
    # Build filename with optional qwen_module suffix
    platform_suffix = f"-{platform_target}" if platform_target else ""
    if team_id:
        base_filename = f"{team_id}_qwen3-30b-a3b{platform_suffix}"
        if qwen_module and qwen_module != "qwen":
            csv_filename = f"{base_filename}_{qwen_module}_score_records.csv"
        else:
            csv_filename = f"{base_filename}_score_records.csv"
    else:
        if qwen_module and qwen_module != "qwen":
            csv_filename = (
                f"qwen3-30b-a3b{platform_suffix}_{qwen_module}_score_records.csv"
            )
        else:
            csv_filename = f"qwen3-30b-a3b{platform_suffix}_score_records.csv"

    # check whether metrics CSV file exists locally
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, "a", newline="") as csvfile:
        fieldnames = [
            "team_id",
            "member_id",
            "base_latency",
            "base_throughput",
            "accuracy",
            "latency",
            "throughput",
            "nki_flop_ratio",
            "increased_throughput",
            "reduced_latency",
            "final_score",
            "timestamp",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write column header if file is new
        if not file_exists:
            writer.writeheader()

        # Write the record columns
        writer.writerow(
            {
                "team_id": team_id,
                "member_id": member_id,
                "base_latency": base_latency,
                "base_throughput": base_throughput,
                "accuracy": accuracy,
                "latency": latency,
                "throughput": throughput,
                "nki_flop_ratio": nki_flop_ratio,
                "increased_throughput": increased_throughput,
                "reduced_latency": reduced_latency,
                "final_score": final_score,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    print(
        f"Inference Metrics from the current Test are written to file: {csv_filename}"
    )

    return final_score


def find_hlos():

    # this path is defined by default NxD, the string matching works with Neuron SDK 2.27
    enc_dir = "/tmp/nxd_model/context_encoding_model/_tp0_bk0"
    ctx_enc = [f for f in os.listdir(enc_dir) if "hlo_module" in f.lower()]
    assert len(ctx_enc) == 1
    ctx_rt = os.path.join(enc_dir, ctx_enc[0])

    tkg_dir = "/tmp/nxd_model/token_generation_model/_tp0_bk0"
    tkg_gen = [f for f in os.listdir(tkg_dir) if "hlo_module" in f.lower()]
    assert len(tkg_gen) == 1
    tkg_rt = os.path.join(tkg_dir, tkg_gen[0])

    print("Found your HLOs")

    return ctx_rt, tkg_rt


def main():
    args = parse_args()
    if not args.prompts:
        args.prompts = ["I believe the meaning of life is"]

    args.batch_size = len(args.prompts)
    args.max_length = args.seq_len
    args.tol_map = (
        "{None: (1e-5, 0.05), 1000: (1e-5, 0.03), 50: (1e-5, 0.03), 5: (1e-5, 0.03)}"
    )

    # points to your local model definition from qwen.py or qwen_with_nki.py
    if args.enable_nki:
        print("Loading qwen_with_nki module (NKI-accelerated RMSNorm enabled)")
        qwen = importlib.import_module("qwen_with_nki")
    else:
        qwen = importlib.import_module(args.qwen)

    if args.mode == "generate":
        model, tokenizer, generation_config = prepare_inference(
            qwen.NeuronQwen3MoeForCausalLM, args
        )

        run_generation(model, tokenizer, args.prompts, generation_config)

    elif args.mode == "validate":
        if args.platform_target == "trn2":
            print("Validation not supported for trn2, exiting.")
            quit()

        model, tokenizer, generation_config = prepare_inference(
            qwen.NeuronQwen3MoeForCausalLM, args
        )

        base_model, _, base_generation_config = prepare_inference(
            baseline_qwen.NeuronQwen3MoeForCausalLM, args
        )

        passed = run_accuracy_check(
            base_model,
            base_generation_config,
            model,
            tokenizer,
            generation_config,
            args.prompts,
            args.divergence_difference_tol,
            args.tol_map,
            num_tokens_to_check=args.num_tokens_to_check,
        )

        status = "passed" if passed else "failed"
        print(f"Validation {status}.")

    elif args.mode == "evaluate_single":
        if args.platform_target == "trn2":
            model, tokenizer, generation_config = prepare_inference(
                qwen.NeuronQwen3MoeForCausalLM, args
            )

            accuracy = 1

        elif args.platform_target == "trn3":
            model, tokenizer, generation_config = prepare_inference(
                qwen.NeuronQwen3MoeForCausalLM, args
            )

            base_model, _, base_generation_config = prepare_inference(
                baseline_qwen.NeuronQwen3MoeForCausalLM, args
            )

            accuracy = run_accuracy_check(
                base_model,
                base_generation_config,
                model,
                tokenizer,
                generation_config,
                args.prompts,
                args.divergence_difference_tol,
                args.tol_map,
                num_tokens_to_check=args.num_tokens_to_check,
            )

        report = benchmark_sampling(model, tokenizer, generation_config, args.prompts)

        latency = report["e2e_model"]["latency_ms_p99"]
        throughput = report["e2e_model"]["throughput"]

        ctx_enc_hlo_path, tkg_gen_hlo_path = find_hlos()

        nki_flop_ratio = count_nki_flop_ratio(ctx_enc_hlo_path, tkg_gen_hlo_path)

        score = calculate_score(
            args.base_latency,
            args.base_throughput,
            accuracy,
            latency,
            throughput,
            nki_flop_ratio,
            args.team_id,
            args.member_id,
            args.qwen,
            args.platform_target,
        )
        print(
            f"Prompt: {args.prompts[0]}\n"
            f"Final Score: {score}\n"
            f"\tAccuracy: {accuracy}\n"
            f"\tLatency: {latency}\n"
            f"\tThroughput: {throughput}\n"
            f"\tNKI FLOPs Ratio: {nki_flop_ratio}"
        )

    elif args.mode == "evaluate_all" and args.platform_target == "trn2":
        model, tokenizer, generation_config = prepare_inference(
            qwen.NeuronQwen3MoeForCausalLM, args
        )

        accuracy = 1

        prompts = parse_prompts("prompts.txt")
        prompt_data = parse_prompt_data("prompt_data_trn2.csv")
        assert len(prompts) == len(prompt_data)

        total_score = 0

        # to do - move both of these calls into batch mode
        # Iterate through the prompts
        for i, prompt in enumerate(prompts):
            data = prompt_data[i]
            base_latency = float(data[3])
            base_throughput = float(data[4])

            report = benchmark_sampling(model, tokenizer, generation_config, [prompt])

            latency = report["e2e_model"]["latency_ms_p99"]
            throughput = report["e2e_model"]["throughput"]

            ctx_enc_hlo_path, tkg_gen_hlo_path = find_hlos()

            nki_flop_ratio = count_nki_flop_ratio(ctx_enc_hlo_path, tkg_gen_hlo_path)

            score = calculate_score(
                base_latency,
                base_throughput,
                accuracy,
                latency,
                throughput,
                nki_flop_ratio,
                args.team_id,
                args.member_id,
                args.qwen,
                args.platform_target,
            )
            print(
                f"Prompt: {prompt}\n"
                f"Final Score: {score}\n"
                f"\tAccuracy: {accuracy}\n"
                f"\tLatency: {latency}\n"
                f"\tThroughput: {throughput}\n"
                f"\tNKI FLOPs Ratio: {nki_flop_ratio}"
            )
            total_score += score

        print(f"\nTotal Score: {total_score}\n")

    elif args.mode == "evaluate_all" and args.platform_target == "trn3":
        model, tokenizer, generation_config = prepare_inference(
            qwen.NeuronQwen3MoeForCausalLM, args
        )

        base_model, _, base_generation_config = prepare_inference(
            baseline_qwen.NeuronQwen3MoeForCausalLM, args
        )

        prompts = parse_prompts("prompts.txt")
        prompt_data = parse_prompt_data("prompt_data_trn3.csv")
        assert len(prompts) == len(prompt_data)

        total_score = 0

        # Iterate through the prompts
        for i, prompt in enumerate(prompts):
            data = prompt_data[i]
            base_latency = float(data[3])
            base_throughput = float(data[4])

            accuracy = run_accuracy_check(
                base_model,
                base_generation_config,
                model,
                tokenizer,
                generation_config,
                [prompt],
                args.divergence_difference_tol,
                args.tol_map,
                num_tokens_to_check=args.num_tokens_to_check,
            )

            report = benchmark_sampling(model, tokenizer, generation_config, [prompt])

            latency = report["e2e_model"]["latency_ms_p99"]
            throughput = report["e2e_model"]["throughput"]

            ctx_enc_hlo_path, tkg_gen_hlo_path = find_hlos()

            nki_flop_ratio = count_nki_flop_ratio(ctx_enc_hlo_path, tkg_gen_hlo_path)

            score = calculate_score(
                base_latency,
                base_throughput,
                accuracy,
                latency,
                throughput,
                nki_flop_ratio,
                args.team_id,
                args.member_id,
                args.qwen,
                args.platform_target,
            )
            print(
                f"Prompt: {prompt}\n"
                f"Final Score: {score}\n"
                f"\tAccuracy: {accuracy}\n"
                f"\tLatency: {latency}\n"
                f"\tThroughput: {throughput}\n"
                f"\tNKI FLOPs Ratio: {nki_flop_ratio}"
            )
            total_score += score

        print(f"\nTotal Score: {total_score}\n")
    else:
        assert False, "Undefined mode"


if __name__ == "__main__":
    main()

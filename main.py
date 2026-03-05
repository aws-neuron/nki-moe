"""Driver script for NKI contests"""
import argparse
import ast
import base64
import copy
import json
import os
import time
import torch

from torch_neuronx.pyhlo.hlo_pb2 import HloModuleProto
from torch_neuronx.testing.validation import logit_validation
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import OnDeviceSamplingConfig, to_torch_dtype
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from neuronx_distributed_inference.utils.accuracy import get_generate_outputs
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter

from neuronx_distributed_inference.utils.random import set_random_seed

from neuronx_distributed_inference.utils.benchmark import create_submodule_latency_collectors, register_latency_collectors, generate_report, Benchmark

# load the baseline model
from neuronx_distributed_inference.models.qwen3_moe import modeling_qwen3_moe as baseline_qwen

# use this to load the local qwen
import importlib

from test import parse_prompts, parse_prompt_data

BENCHMARK_REPORT_FILENAME = "benchmark_report.json"

set_random_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()

    # contest specific
    parser.add_argument("--mode", choices=["evaluate_single", "evaluate_all", "validate", "generate", "generate_accuracy_baselines"])
    parser.add_argument("--qwen", type=str, default="qwen")
    parser.add_argument("--enable-nki", action="store_true")
    parser.add_argument("--base-latency", type=float, default=526.15)
    parser.add_argument("--base-throughput", type=float, default=134.61)

    # Model path
    parser.add_argument("--model-path", type=str, default="/home/ubuntu/Qwen3-30B-A3B/hf_model")
    parser.add_argument("--compiled-model-path", type=str,
                        default="/home/ubuntu/Qwen3-30B-A3B/traced_model")

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

    # Basic config
    parser.add_argument("--torch-dtype", type=to_torch_dtype, default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--padding-side", type=str)
    parser.add_argument("--seq-len", type=int, default=640)
    parser.add_argument("--n-active-tokens", type=int)
    parser.add_argument("--n-positions", type=int)
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--rpl-reduce-dtype", type=to_torch_dtype)
    parser.add_argument("--output-logits", action="store_true")
    parser.add_argument("--vocab-parallel", action="store_true")
    parser.add_argument("--skip-compile", type=bool, default=False)
    parser.add_argument("--save_sharded_checkpoint", type=bool, default=True)
    parser.add_argument("--platform-target", type=str, default='trn2') 

    # Attention
    parser.add_argument("--fused-qkv", action="store_true")
    parser.add_argument("--sequence-parallel-enabled", action="store_true")
    parser.add_argument("--flash-decoding-enabled", action="store_true")

    # On device sampling
    parser.add_argument("--on-device-sampling", action="store_true")

    # Bucketing
    parser.add_argument("--enable-bucketing", type=bool, default=True)
    parser.add_argument("--bucket-n-active-tokens", action="store_true")
    parser.add_argument("--context-encoding-buckets", nargs="+", type=int)
    parser.add_argument("--token-generation-buckets", nargs="+", type=int)

    # Parallelism
    parser.add_argument("--tp-degree", type=int, default=4)

    # Kernels
    parser.add_argument("--qkv-kernel-enabled", action="store_true")
    parser.add_argument("--attn-kernel-enabled", action="store_true")
    parser.add_argument("--mlp-kernel-enabled", action="store_true")
    parser.add_argument("--quantized-mlp-kernel-enabled", action="store_true")
    parser.add_argument("--rmsnorm-quantize-kernel-enabled", action="store_true")
    parser.add_argument("--quantized-kernel-lower-bound", type=float, default=1200.0)
    parser.add_argument("--mlp-kernel-fuse-residual-add", action="store_true")

    return parser.parse_args()


def validate_file_exists(path):
    if not os.path.exists(path) or not os.path.isfile(path):
        raise argparse.ArgumentError("Path must exist and be a file")
    return path


def load_tokenizer(model_path, compiled_model_path, neuron_config):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(compiled_model_path)
    return tokenizer


def prepare_inference(model_cls, args):
    # Initialize configs.
    print("Loading configs...")

    # Skip values not specified in the args to avoid setting values to None in the config.
    config_kwargs = copy.deepcopy(vars(args))
    config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

    if args.on_device_sampling:
        config_kwargs["on_device_sampling_config"] = OnDeviceSamplingConfig(**config_kwargs)

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
        model.compile(args.compiled_model_path, debug=False)
    
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
        k: getattr(args, k) for k in generation_config_args if getattr(args, k) is not None
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

    print ('Beginning benchmark sampling')
    
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


def check_accuracy_logits(base_model, base_generation_config, neuron_model, tokenizer, generation_config, prompts, divergence_difference_tol, tol_map, num_tokens_to_check):
    assert (prompts is not None)

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    initial_input_ids = inputs.input_ids
    initial_attention_mask = inputs.attention_mask
    seq_len = neuron_model.config.neuron_config.seq_len

    neuron_model.config.neuron_config.max_new_tokens = seq_len - initial_input_ids.shape[1]

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
            actual_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
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
        with open(hlo_file_path, 'rb') as f:
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
                    if instruction.custom_call_target == 'AwsNeuronCustomNativeKernel':
                        try:
                            backend_config = instruction.backend_config
                            config = json.loads(base64.b64decode(backend_config))
                            mac_count = int(config['mac_count'])
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
                        if i not in dnums.rhs_contracting_dimensions and \
                           i not in dnums.rhs_batch_dimensions:
                            rhs_non_contracting_size *= rhs_shape.dimensions[i]

                    mac_count = (lhs_batch * lhs_non_contracting_size *
                                 lhs_contracting_size * rhs_non_contracting_size)
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


def calculate_score(base_latency, base_throughput, accuracy, latency, throughput, nki_flop_ratio):
    

    increased_throughput = throughput / base_throughput
    reduced_latency = base_latency / latency

    # resetting nki_flop_ratio as the baseline solution uses NKI completely
    final_score = accuracy * reduced_latency * increased_throughput * nki_flop_ratio

    print ('In this final score of ', final_score, ' the contestant got a breakdown as follows.')
    print ('accuracy: ', accuracy)
    print ('reduced_latency: ', reduced_latency)
    print ('increased throughput: ',  increased_throughput)
    print ('nki flop ratio: ', nki_flop_ratio)
    
    return final_score

def find_hlos():
    
    # this path is defined by default NxD, the string matching works with Neuron SDK 2.27
    enc_dir = '/tmp/nxd_model/context_encoding_model/_tp0_bk0'
    ctx_enc = [f for f in os.listdir(enc_dir) if 'hlo_module' in f.lower()]
    assert len(ctx_enc) == 1
    ctx_rt = os.path.join(enc_dir, ctx_enc[0])

    tkg_dir = '/tmp/nxd_model/token_generation_model/_tp0_bk0'
    tkg_gen = [f for f in os.listdir(tkg_dir) if 'hlo_module' in f.lower()]
    assert len(tkg_gen) == 1
    tkg_rt = os.path.join(tkg_dir, tkg_gen[0])

    print ('Found your HLOs')

    return ctx_rt, tkg_rt

def main():
    args = parse_args()
    if not args.prompts:
        args.prompts = ["I believe the meaning of life is"]
        
    args.batch_size = len(args.prompts)
    args.max_length = args.seq_len
    args.tol_map = "{None: (1e-5, 0.05), 1000: (1e-5, 0.03), 50: (1e-5, 0.03), 5: (1e-5, 0.03)}"

    # points to your local model definition from qwen.py or qwen_with_nki.py
    if args.enable_nki:
        print("Loading qwen_with_nki module (NKI-accelerated RMSNorm enabled)")
        qwen = importlib.import_module("qwen_with_nki")
    else:
        qwen = importlib.import_module(args.qwen)   

    if args.mode == "generate":
        model, tokenizer, generation_config = prepare_inference(qwen.NeuronQwen3MoeForCausalLM, args)
        
        run_generation(
            model,
            tokenizer,
            args.prompts,
            generation_config
        )

    elif args.mode == "validate":
        if args.platform_target == 'trn2':
            print ('Validation not supported for trn2, exiting.')
            quit()
            
        model, tokenizer, generation_config = prepare_inference(qwen.NeuronQwen3MoeForCausalLM, args)
        
        base_model, _, base_generation_config = prepare_inference(baseline_qwen.NeuronQwen3MoeForCausalLM, args)

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

        if args.platform_target == 'trn2':
            
            model, tokenizer, generation_config = prepare_inference(qwen.NeuronQwen3MoeForCausalLM, args)

            accuracy = 1
            
        elif args.platform_target == 'trn3': 
        
            model, tokenizer, generation_config = prepare_inference(qwen.NeuronQwen3MoeForCausalLM, args)

            base_model, _, base_generation_config = prepare_inference(baseline_qwen.NeuronQwen3MoeForCausalLM, args)

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

        score = calculate_score(args.base_latency, args.base_throughput, accuracy, latency, throughput, nki_flop_ratio)
        print(
            f"Prompt: {args.prompts[0]}\n"
            f"Final Score: {score}\n"
            f"\tAccuracy: {accuracy}\n"
            f"\tLatency: {latency}\n"
            f"\tThroughput: {throughput}\n"
            f"\tNKI FLOPs Ratio: {nki_flop_ratio}"
        )

    elif args.mode == 'evaluate_all' and args.platform_target == 'trn2':
        
        model, tokenizer, generation_config = prepare_inference(qwen.NeuronQwen3MoeForCausalLM, args)

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

            score = calculate_score(base_latency, base_throughput, accuracy, latency, throughput, nki_flop_ratio)
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
        
    elif args.mode == "evaluate_all" and args.platform_target == 'trn3':
        
        model, tokenizer, generation_config = prepare_inference(qwen.NeuronQwen3MoeForCausalLM, args)

        base_model, _, base_generation_config = prepare_inference(baseline_qwen.NeuronQwen3MoeForCausalLM, args)
        
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

            score = calculate_score(base_latency, base_throughput, accuracy, latency, throughput, nki_flop_ratio)
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

    elif args.mode == "generate_accuracy_baselines":

        base_model, tokenizer, base_generation_config = prepare_inference(baseline_qwen.NeuronQwen3MoeForCausalLM, args)

        prompts = parse_prompts("prompts.txt")

        # Iterate through the prompts
        for i, prompt in enumerate(prompts):
        
            inputs = tokenizer(args.prompts, padding=True, return_tensors="pt")
            initial_input_ids = inputs.input_ids
            initial_attention_mask = inputs.attention_mask
            seq_len = base_model.config.neuron_config.seq_len
            
            base_model.config.neuron_config.max_new_tokens = seq_len - initial_input_ids.shape[1]
            
            base_model_generative = HuggingFaceGenerationAdapter(base_model)
            
            new_tokens = base_model.config.neuron_config.max_new_tokens 
            
            with torch.inference_mode():
                outputs = base_model_generative.generate(
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
    
            # write logits to a file 
            torch.save(expected_logits, f'expected_logits_{i}.pt')
        
    else:
        assert False, "Undefined mode"


if __name__ == "__main__":
    main()

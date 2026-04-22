# AWS Trainium2/3 MoE Kernel Challenge

**MLSys 2026 Competition Track**

Participants will write custom kernels with the Neuron Kernel Interface (NKI) for the Qwen3-30B-A3B Mixture of Experts model and optimize inference performance on AWS Trainium2/3 hardware.

For full details on the competition, read [the competition guidelines](https://github.com/aws-neuron/nki-moe/blob/main/CONTEST.md). Team registration is closed at this time.

For teams who have already registered, we are excited to share that we are exposing a leaderboard! This will give you a way to upload your script, get results on performance, and see where you stand in the rankings. You'll need to register for the leaderboard submission site [here](https://dutu9c4a6raza.cloudfront.net/). Please note down the team id that is given to you, and email this to the organizers. Once we confirmed that you're already registered, we will enable you within the leaderboard. The leaderboard will take submissions starting 3/23.

For teams who have registered on the leaderboard, the submission page is open [here](https://dutu9c4a6raza.cloudfront.net/submit.html). You can view the results of the leaderboard evaluations [here](https://aws-neuron.github.io/nki-moe/#leaderboard).

### Round one results

Congratulations to the following teams for making it to Round Two: 5493, 3459, 7013, 8662, 2598, 9376, 9263, 1784, 7284, 3857, 7356, 6752, 9290, 2374, 8031. We are working on your Trn3 access now, and will provide this within the next few days. If you have not sent us your AWS account ID, please do this asap so we can ensure you have the right access you need.

We are also updating the leaderboard system now to run on Trn3. We'll be updating the baseline solution to use the latest Neuron SDK release in the next few days, please watch the repo to pull this down in your tests. The leaderboard will reopen early next week to take your Trn3 submissions, and will remain open through April 24. This date will not be extended, please ensure you make at least one submission early so you can be sure you have a spot.

For those of you who have developed NKI kernels as part of your solution, thank you! We're very excited to see your progress. For those of you who have not, please consider exploring this. Performance alone will count for 85% of your total score, but the remaining 15% will come from NKI kernel innovation.

We will also be making additional credit codes available. Please reach out to us at nki-mlsys-2026@amazon.com with any questions.

### Round two: Trn3 in April
Round two of the competition focuses on Trn3. We will take submissions from April 14-29. Each of the top 15 teams from round one will receive access to a dedicated single-chip Trn3 instance. The evaluation environment will use Neuron SDK 2.29 with a single Trn3 chip.

## Submission guidelines
1. Participants should plan to replace the `qwen_with_nki.py` file with your own kernels and model code.
2. Participants **must contain their code within a single file**. The submission site will only accept one Python file per upload.
3. This file will be invoked by `main.py`, exactly in the same way contained within the repository.
4. The evaluation environment will already have this repository cloned within a Neuron SDK 2.29 environment. Participants do not need to install any packages already contained within the repository.
5. The contest organizers will expose a submission url on March 23.
6. It is not necessary to submit technical documentation at this time; we will only require this of the winning teams.

## Getting Started

To learn NKI, follow [the official NKI guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html) and various example NKI kernels from the [nki-samples repository](https://github.com/aws-neuron/nki-samples). Another tool to help with optimizing NKI kernels is [NKI autotune](https://github.com/awslabs/nki-autotune).

## Setup Steps

1. Create a Trainium2 instance with AWS Neuron SDK v2.27 using EC2 based on the [setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/neuron-setup/multiframework/multi-framework-ubuntu24-neuron-dlami.html#setup-ubuntu24-multi-framework-dlami).
2. Activate the Neuron virtual environment to run inference by running the appropriate activation command for your SDK version:
   ```bash
   source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
   ```
3. Clone this repository and run `cd [PATH]/nki-moe` where `[PATH]` is the directory where you have performed the clone.
4. Download the Qwen3-30B-A3B model to a `~/qwen-30b-a3b/hf_model` folder in your root directory. We recommend doing so using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli). You can install this by running `pip3 install huggingface_hub[cli]`.
5. To run inference, navigate to `[PATH]/nki-moe` and run:
   ```bash
   python3 main.py --mode generate --model-path ~/qwen-30b-a3b/hf_model --compiled-model-path ~/qwen-30b-a3b/traced_model --prompt "What is the capital of France?"
   ```

## NKI Kernel Development

This repository contains the standard model implementation in `qwen.py`.

Your task is to identify parts of the model (operators, fused operators, layers, or even the whole model) that can be implemented as NKI kernels and add them to create optimized versions of the model.

### Sample NKI Kernels

This repository includes two NKI kernel examples to help you get started:

#### 1. Tensor Add Example (`nki_tensor_add_example.py`)

A simple NKI kernel demonstrating basic tensor operations. This serves as a minimal reference implementation showing:
- Basic NKI kernel structure with `@nki.jit` decorator
- Tensor indexing and loading from HBM to SBUF
- Element-wise operations
- Storing results back to HBM

This example is not integrated into the model but provides a foundation for understanding NKI kernel development.

#### 2. RMSNorm Kernel (`nki_custom_rmsnorm.py`)

A production-ready NKI RMSNorm implementation integrated into the Qwen model. This kernel follows the pattern from the [official AWS NKI RMSNorm tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.0/general/nki/tutorials/rmsnorm.html).


We also have `qwen_with_nki.py` which has model implementation with custom NKI kernels integrated. To test the different implementations:

```bash
# Standard inference (uses qwen.py)
python3 main.py --mode generate --model-path ~/qwen-30b-a3b/hf_model --compiled-model-path ~/qwen-30b-a3b/traced_model --prompt "What is the capital of France?"

# With NKI RMSNorm kernel (uses qwen_with_nki.py)
python3 main.py --mode generate --enable-nki --model-path ~/qwen-30b-a3b/hf_model --compiled-model-path ~/qwen-30b-a3b/traced_model --prompt "What is the capital of France?"
```

**Important:** When switching between NKI and standard modes, remove the traced model directory and compile cache to ensure proper recompilation:
```bash
rm -rf ~/qwen-30b-a3b/traced_model
rm -rf /var/tmp/neuron-compile-cache/*
```

The `--enable-nki` flag in `main.py` controls which model file is loaded:
- Without flag: loads `qwen.py` (standard implementation)
- With flag: loads `qwen_with_nki.py` (NKI-accelerated implementation)

Key areas to focus on:
* MoE routing and expert selection logic
* Expert computation (gate_proj, up_proj, down_proj)
* Attention mechanisms with MoE-specific optimizations
* Memory-efficient tensor operations for sparse expert execution

## Evaluation and Scoring

The contest organizers will execute each team's submission across the twenty withheld benchmarks on a dedicated Trainium instance. The submissions will be evaluated on:

1) Accuracy of generated output vs. our reference implementation. Accuracy evaluation will be a binary assessor: Any benchmark that fails an accuracy threshold will result in a score of 0\.   
2) Latency (End-to-end model latency per prompt)  
3) Throughput measured as output tokens / second  
4) Amount of model written in NKI (measured as NKI FLOPS / total model FLOPS) (will be applied as a scaling factor for (b) and (c)). Note: NKI FLOPs measures the number of multiply-accumulate (MAC) operations.

Rankings will be established by calculating the total normalized number of points per team, where points are normalized against the baseline.

We define **points** as **Accuracy** (binary) **\* Reduced Latency \* Increased Throughput \* (1 + Normalized NKI FLOPS)**, where:

* **Accuracy** = 1 if accuracy matches or exceeds a predetermined threshold, 0 otherwise  
* **Reduced Latency** = Reference implementation E2E latency divided by submission E2E latency  
* **Increased Throughput** = Submission tokens/sec divided by reference implementation tokens/sec  
* **Normalized NKI FLOPS** = Submission NKI FLOPS divided by total model FLOPS

For example, a submission that is sufficiently accurate, with 10x reduced latency, 2x increased throughput, and 0.85 normalized NKI FLOPS would obtain 1 \* 10 \* 2 \* 1.85 \= 37 points.

## Additional Tools

1. **Profiling:** If you would like to profile your implementation in order to get a better understanding of performance bottlenecks and opportunities for optimization, you can use the [Neuron Explorer](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/index.html).
2. **Benchmarking:** You can also leverage the [NKI benchmarking API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.benchmark.html) to retrieve execution latency statistics.

## FAQ's
1. What batch size will be used in evalution? We will use a batch size of 1 in the final evaluation.
2. Should I use NKI 1 or NKI 2? We will make both NKI 1 and NKI 2 available in the evaluation suite, but we will prefer solutions that use NKI 2.
3. How can I access Trn2? Please follow the notes [here](https://github.com/aws-neuron/nki-moe/issues/9).

## Contact

**Email**: [nki-mlsys-2026@amazon.com](mailto:nki-mlsys-2026@amazon.com)

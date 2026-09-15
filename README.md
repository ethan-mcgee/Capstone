# Llama 3.1 8B and Qwen3-14B inference benchmark

`airllm_test.py` runs the same prompt and deterministic generation settings with either `meta-llama/Llama-3.1-8B-Instruct` or `Qwen/Qwen3-14B`. The checked-in defaults are the `llama` model profile and the `airllm` runtime mode.

## Model and runtime selection

Edit these constants near the top of `airllm_test.py`:

```python
MODEL_PROFILE = "llama"  # "llama" or "qwen"
RUNTIME_MODE = "airllm"  # "resident", "airllm", or "hybrid"
```

Model selection intentionally remains source-level so benchmark configuration is visible in the checked-in script. The runtime modes are:

- `resident`: loads the selected model entirely on `cuda:0` with bitsandbytes 8-bit quantization and rejects CPU, disk, or meta placement.
- `airllm`: streams one layer at a time through `cuda:0`. It uses less VRAM and is not fully GPU-resident.
- `hybrid`: keeps a fixed native-BF16 prefix and the boundary modules on `cuda:0`, then streams the decoder suffix with AirLLM 4.0.0's pinned-memory and one-layer-ahead prefetch behavior. It is not fully GPU-resident.

The script never changes models, reduces a hybrid partition, or switches runtime modes after a loading failure.

## Validated model profiles

| Profile | Hugging Face model | Decoder layers | Hybrid resident | Hybrid streamed |
| --- | --- | ---: | ---: | ---: |
| `llama` | `meta-llama/Llama-3.1-8B-Instruct` | 32 | 16 | 16 |
| `qwen` | `Qwen/Qwen3-14B` | 40 | 24 | 16 |

Llama is a gated checkpoint. Before running it, accept Meta's license terms on the [Llama 3.1 8B Instruct model page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and authenticate the local Hugging Face client with an account that has access. AirLLM lists Llama 3.1 among the supported model families in its [project documentation](https://github.com/lyogavin/airllm).

AirLLM shards are stored beneath `AIRLLM_SHARD_ROOT`, currently `C:\AI\airllm-layers`. Each profile has an isolated child directory:

```text
C:\AI\airllm-layers\llama-3.1-8b-instruct
C:\AI\airllm-layers\qwen3-14b
```

This prevents a profile switch from reusing another model's split checkpoint. Change `AIRLLM_SHARD_ROOT` if the shards belong elsewhere.
AirLLM places the shard files in a `splitted_model` directory inside each listed profile directory.

## Environment

The versions in `requirements.txt` reproduce the dependency stack verified with Python 3.12.6 for this project:

| Dependency | Version |
| --- | --- |
| AirLLM | 4.0.0 |
| Transformers | 5.17.0 |
| Accelerate | 1.15.0 |
| PyTorch | 2.14.0+cu132 |
| bitsandbytes | 0.50.2 |

Create and populate a virtual environment on a machine with a CUDA 13.2-compatible NVIDIA driver:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the selected inference:

```powershell
.\.venv\Scripts\python.exe .\airllm_test.py
```

The fixed prompt asks for a three-sentence explanation of machine learning. All profiles and modes retain the 512-token sequence limit, 64-token output limit, greedy deterministic decoding, CUDA-only execution, and identical metric definitions. Resident downloads and initial loading count toward end-to-end time. Generation throughput is measured only around `generate()`, with CUDA synchronization before and after it.

## Fixed hybrid budget and reporting

Hybrid mode checks the actual sizes of the selected profile's AirLLM shard files before materializing weights. The required budget is the complete resident set plus the largest active streamed layer. If it does not fit in currently free `cuda:0` memory, the script reports the selected model, fixed decoder split, measured resident shard volume, and current free and total VRAM, then stops without fallback.

Successful hybrid summaries report the selected model, the fixed decoder split, actual resident shard size, and total streamed weight volume per forward. All modes report input and generated token counts, synchronized generation and end-to-end time, generated tokens per second, and peak allocated and reserved CUDA memory. Generated-token counts and throughput exclude prompt tokens.

## Comparable hardware runs

For a direct RTX 3090 Ti comparison:

1. Set `MODEL_PROFILE = "llama"` and run `resident`, `airllm`, and `hybrid` in turn.
2. Confirm the response contains three sentences, the summary identifies the Llama checkpoint, hybrid reports a 16/16 split, and resident validation finds no CPU or disk placement.
3. Set `MODEL_PROFILE = "qwen"`, select `hybrid`, and run the Qwen baseline once. Confirm the 24/16 split.
4. Record each complete performance summary without changing the prompt, limits, decoding settings, dependency versions, or other GPU workload.

Do not claim performance parity until both model runs have completed on the same hardware. No hardware benchmark results are checked in by this change.

# Qwen3-14B inference test

`airllm_test.py` runs the same fixed Qwen3-14B prompt with deterministic generation in one of two modes:

- `resident` is the default. It loads the entire model on `cuda:0` with bitsandbytes 8-bit quantization and rejects CPU, disk, or meta placement.
- `airllm` uses AirLLM to stream one layer at a time through `cuda:0`. It uses less VRAM, but the model is not fully GPU-resident.

Set `RUNTIME_MODE` near the top of `airllm_test.py` to select a mode. The script never switches modes automatically after a loading failure.

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

Run the fixed inference:

```powershell
.\.venv\Scripts\python.exe .\airllm_test.py
```

Resident mode downloads the original `Qwen/Qwen3-14B` checkpoint through Hugging Face if the weights are not already cached. The download and initial 8-bit loading are included in end-to-end time, but generation throughput is measured only around `generate()` with CUDA synchronization before and after it.

AirLLM mode reads or creates its split checkpoint under `AIRLLM_SHARD_PATH`, currently `C:\AI\airllm-layers`. Change that constant if the shards are stored elsewhere.

After each successful response, the script prints JSON metrics for input and generated token counts, synchronized generation and end-to-end time, generated tokens per second, and peak allocated and reserved CUDA memory. Generated-token counts and throughput exclude prompt tokens.

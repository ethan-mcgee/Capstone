# Qwen3-14B inference test

`airllm_test.py` runs the same fixed Qwen3-14B prompt with deterministic generation in one of three modes. The checked-in selection is `airllm`.

- `resident` loads the entire model on `cuda:0` with bitsandbytes 8-bit quantization and rejects CPU, disk, or meta placement.
- `airllm` uses AirLLM to stream one layer at a time through `cuda:0`. It uses less VRAM, but the model is not fully GPU-resident.
- `hybrid` is a fixed RTX 3090 Ti layout for `Qwen/Qwen3-14B`. It keeps native BF16 embeddings, final normalization, the language-model head, and decoder layers 0 through 23 on `cuda:0`. Decoder layers 24 through 39 retain AirLLM 4.0.0's pinned-memory and one-layer-ahead streaming behavior. It is not fully GPU-resident.

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

AirLLM and hybrid modes read or create their split checkpoint under `AIRLLM_SHARD_PATH`, currently `C:\AI\airllm-layers`. Change that constant if the shards are stored elsewhere. Shard compression is not enabled, so hybrid weights retain the checkpoint's native BF16 precision.

## Fixed hybrid budget

The hybrid layout is deliberately static. Its 24 resident decoder layers plus embeddings, normalization, and output head occupy 17.66 GiB of BF16 shards. One streamed decoder layer raises the active weight footprint to approximately 18.28 GiB. The 16-layer suffix transfers 9.84 GiB per model forward, compared with 27.51 GiB for pure AirLLM. Remaining VRAM is available for CUDA context, activations, KV cache, and allocator overhead.

Before materializing weights, hybrid mode checks current free and total memory on `cuda:0`. If the fixed resident set plus one active streamed layer does not fit, it reports the requested layer count, resident shard size, and current free and total VRAM, then stops. It never reduces the resident layer count or switches runtime modes. Dynamic hardware-based layer selection is possible using free-VRAM inspection and shard metadata, but it is outside this change and is neither implemented nor designed here.

After each successful response, the script prints a performance summary for input and generated token counts, synchronized generation and end-to-end time, generated tokens per second, and peak allocated and reserved CUDA memory. Hybrid summaries also show the 24/16 decoder split, 17.66 GiB resident shard size, and 9.84 GiB streamed weight volume per forward. Time, throughput, and CUDA memory values use two decimal places. Generated-token counts and throughput exclude prompt tokens.

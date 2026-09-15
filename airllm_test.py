import time

SCRIPT_START_TIME = time.perf_counter()

import logging
from pathlib import Path

import torch


# Change this value manually to either "resident" or "airllm".
RUNTIME_MODE = "resident"
MODEL_ID = "Qwen/Qwen3-14B"
AIRLLM_SHARD_PATH = Path(r"C:\AI\airllm-layers")
MAX_SEQUENCE_LENGTH = 512
MAX_NEW_TOKENS = 64
CUDA_DEVICE = "cuda:0"

MESSAGES = [
    {
        "role": "system",
        "content": "You are a concise and helpful assistant.",
    },
    {
        "role": "user",
        "content": "Explain what machine learning is in three sentences.",
    },
]


class SuppressBitsAndBytesCastNotice(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.name == "bitsandbytes.autograd._functions"
            and record.getMessage().startswith("MatMul8bitLt: inputs will be cast")
        )


def configure_terminal_logging() -> None:
    bitsandbytes_logger = logging.getLogger("bitsandbytes.autograd._functions")
    bitsandbytes_logger.addFilter(SuppressBitsAndBytesCastNotice())


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for both runtime modes, but PyTorch cannot access a CUDA GPU. "
            "Install the CUDA-enabled dependency stack documented in README.md and verify the "
            "NVIDIA driver."
        )


def is_cuda_zero(device: object) -> bool:
    if isinstance(device, int):
        return device == 0

    try:
        parsed = torch.device(device)
    except (RuntimeError, TypeError):
        return False

    return parsed.type == "cuda" and parsed.index in (None, 0)


def validate_resident_model(model: torch.nn.Module) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        invalid_mappings = {
            name or "<root>": str(device)
            for name, device in device_map.items()
            if not is_cuda_zero(device)
        }
        if invalid_mappings:
            raise RuntimeError(
                "Resident loading placed model components outside cuda:0: "
                f"{invalid_mappings}. Free more VRAM and retry. No AirLLM fallback was attempted."
            )

    invalid_tensors = []
    tensor_count = 0
    for tensor_kind, named_tensors in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in named_tensors:
            tensor_count += 1
            if not is_cuda_zero(tensor.device):
                invalid_tensors.append(f"{tensor_kind} {name}: {tensor.device}")

    if tensor_count == 0:
        raise RuntimeError(
            "The loaded model exposes no parameters or buffers, so GPU residency cannot be "
            "verified. No AirLLM fallback was attempted."
        )

    if invalid_tensors:
        preview = ", ".join(invalid_tensors[:10])
        if len(invalid_tensors) > 10:
            preview += f", and {len(invalid_tensors) - 10} more"
        raise RuntimeError(
            "Resident loading left model tensors outside cuda:0: "
            f"{preview}. Free more VRAM and retry. No AirLLM fallback was attempted."
        )

    if device_map:
        printable_map = {name or "<root>": str(device) for name, device in device_map.items()}
        print(f"Validated resident device map: {printable_map}")
    else:
        print("Transformers did not expose hf_device_map for this fixed single-device load.")
    print(f"Validated {tensor_count} model parameters and buffers on cuda:0.")


def load_resident_model():
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    print("Loading the 8-bit model fully into cuda:0 (CPU and disk offload disabled)...")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=False,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map={"": CUDA_DEVICE},
            low_cpu_mem_usage=True,
        )
        validate_resident_model(model)
    except Exception as exc:
        raise RuntimeError(
            "Resident mode could not load Qwen3-14B entirely on cuda:0 in 8-bit precision. "
            "Verify the pinned dependencies, CUDA support, checkpoint access, and available VRAM. "
            "The script will not silently switch to AirLLM; set RUNTIME_MODE = \"airllm\" "
            "explicitly if layer streaming is desired."
        ) from exc

    model.eval()
    return model


def load_airllm_model():
    from airllm import AutoModel

    print(
        "Loading with AirLLM layer streaming. This mode moves layers through cuda:0 as needed "
        "and is not fully GPU-resident."
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        layer_shards_saving_path=str(AIRLLM_SHARD_PATH),
        max_seq_len=MAX_SEQUENCE_LENGTH,
        device=CUDA_DEVICE,
    )
    return model


def load_runtime():
    from transformers import AutoTokenizer

    if RUNTIME_MODE not in {"resident", "airllm"}:
        raise ValueError(
            f"Unsupported RUNTIME_MODE {RUNTIME_MODE!r}. Choose either 'resident' or 'airllm'."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if RUNTIME_MODE == "resident":
        return load_resident_model(), tokenizer, True
    return load_airllm_model(), tokenizer, False


def prepare_input(tokenizer) -> torch.Tensor:
    input_ids = tokenizer.apply_chat_template(
        MESSAGES,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=False,
    )

    required_length = input_ids.shape[-1] + MAX_NEW_TOKENS
    if required_length > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"The prompt plus output limit requires {required_length} tokens, exceeding "
            f"MAX_SEQUENCE_LENGTH={MAX_SEQUENCE_LENGTH}."
        )
    return input_ids.to(CUDA_DEVICE)


def generate_response(model, tokenizer, input_ids: torch.Tensor):
    input_token_count = input_ids.shape[-1]

    torch.cuda.synchronize(CUDA_DEVICE)
    torch.cuda.reset_peak_memory_stats(CUDA_DEVICE)
    generation_start = time.perf_counter()
    result = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
    )
    torch.cuda.synchronize(CUDA_DEVICE)
    generation_seconds = time.perf_counter() - generation_start

    generated_token_ids = result.sequences[0, input_token_count:]
    answer = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    end_to_end_seconds = time.perf_counter() - SCRIPT_START_TIME

    generated_token_count = generated_token_ids.numel()
    metrics = {
        "runtime_mode": RUNTIME_MODE,
        "fully_gpu_resident": RUNTIME_MODE == "resident",
        "input_tokens": input_token_count,
        "generated_tokens": generated_token_count,
        "generation_seconds": generation_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "generated_tokens_per_second": (
            generated_token_count / generation_seconds if generation_seconds else 0.0
        ),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(CUDA_DEVICE) / (1024**3),
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(CUDA_DEVICE) / (1024**3),
    }
    return answer, metrics


def print_results(answer: str, metrics: dict) -> None:
    separator = "=" * 62
    rows = [
        ("Runtime mode", metrics["runtime_mode"]),
        ("Fully GPU-resident", "Yes" if metrics["fully_gpu_resident"] else "No"),
        ("Input tokens", f'{metrics["input_tokens"]:,}'),
        ("Generated tokens", f'{metrics["generated_tokens"]:,}'),
        ("Generation time", f'{metrics["generation_seconds"]:.2f} seconds'),
        ("End-to-end time", f'{metrics["end_to_end_seconds"]:.2f} seconds'),
        ("Generation speed", f'{metrics["generated_tokens_per_second"]:.2f} tokens/second'),
        ("Peak CUDA allocated", f'{metrics["peak_cuda_allocated_gib"]:.2f} GiB'),
        ("Peak CUDA reserved", f'{metrics["peak_cuda_reserved_gib"]:.2f} GiB'),
    ]

    print(f"\n{separator}")
    print("MODEL RESPONSE")
    print(separator)
    print(answer.strip())
    print(f"\n{separator}")
    print("PERFORMANCE SUMMARY")
    print(separator)
    for label, value in rows:
        print(f"{label:<27} {value}")
    print(separator)


def main() -> None:
    configure_terminal_logging()
    require_cuda()
    model, tokenizer, fully_gpu_resident = load_runtime()
    input_ids = prepare_input(tokenizer)
    answer, metrics = generate_response(model, tokenizer, input_ids)

    if metrics["fully_gpu_resident"] != fully_gpu_resident:
        raise AssertionError("Runtime residency state changed unexpectedly.")

    print_results(answer, metrics)


if __name__ == "__main__":
    main()

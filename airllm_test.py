import time

SCRIPT_START_TIME = time.perf_counter()

import logging
from pathlib import Path

import torch

from model_profiles import get_model_profile, model_shard_path


# Change these source constants to select a model and runtime.
MODEL_PROFILE = "llama"  # "llama" or "qwen"
RUNTIME_MODE = "airllm"
AIRLLM_SHARD_ROOT = Path(r"C:\AI\airllm-layers")
MAX_SEQUENCE_LENGTH = 512
MAX_NEW_TOKENS = 64
CUDA_DEVICE = "cuda:0"

MODEL = get_model_profile(MODEL_PROFILE)
MODEL_ID = MODEL.model_id
AIRLLM_SHARD_PATH = model_shard_path(AIRLLM_SHARD_ROOT, MODEL)

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
            "CUDA is required for every runtime mode, but PyTorch cannot access a CUDA GPU. "
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

    print(
        f"Loading {MODEL_ID} in 8-bit fully into cuda:0 "
        "(CPU and disk offload disabled)..."
    )
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
            f"Resident mode could not load {MODEL_ID} entirely on cuda:0 in 8-bit precision. "
            "Verify the pinned dependencies, CUDA support, checkpoint access, and available VRAM. "
            "The script will not silently switch to AirLLM; set RUNTIME_MODE = \"airllm\" "
            "explicitly if layer streaming is desired."
        ) from exc

    model.eval()
    return model


def load_airllm_model():
    from airllm import AutoModel

    print(
        f"Loading {MODEL_ID} with AirLLM layer streaming from {AIRLLM_SHARD_PATH}. "
        "This mode moves layers through cuda:0 as needed and is not fully GPU-resident."
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        layer_shards_saving_path=str(AIRLLM_SHARD_PATH),
        max_seq_len=MAX_SEQUENCE_LENGTH,
        device=CUDA_DEVICE,
    )
    return model


def load_hybrid_model():
    from hybrid_airllm import HybridAirLLM

    print(
        f"Loading {MODEL_ID} with fixed-budget hybrid AirLLM in native BF16 from "
        f"{AIRLLM_SHARD_PATH}. The first {MODEL.resident_decoder_layers} decoder layers "
        f"and boundary modules remain on cuda:0; the final {MODEL.streamed_decoder_layers} "
        "decoder layers are streamed. This mode is not fully GPU-resident."
    )
    return HybridAirLLM(
        MODEL_ID,
        resident_decoder_layers=MODEL.resident_decoder_layers,
        layer_shards_saving_path=str(AIRLLM_SHARD_PATH),
        max_seq_len=MAX_SEQUENCE_LENGTH,
        device=CUDA_DEVICE,
    )


def load_runtime():
    from transformers import AutoTokenizer

    if RUNTIME_MODE not in {"resident", "airllm", "hybrid"}:
        raise ValueError(
            f"Unsupported RUNTIME_MODE {RUNTIME_MODE!r}. Choose 'resident', 'airllm', or 'hybrid'."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if RUNTIME_MODE == "resident":
        return load_resident_model(), tokenizer, True
    if RUNTIME_MODE == "hybrid":
        return load_hybrid_model(), tokenizer, False
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
        "model_id": MODEL_ID,
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
    if RUNTIME_MODE == "hybrid":
        metrics.update(
            {
                "resident_decoder_layers": MODEL.resident_decoder_layers,
                "streamed_decoder_layers": MODEL.streamed_decoder_layers,
                "resident_shard_gib": model.resident_shard_bytes / (1024**3),
                "streamed_weight_gib_per_forward": model.streamed_weight_bytes / (1024**3),
            }
        )
    return answer, metrics


def print_results(answer: str, metrics: dict) -> None:
    separator = "=" * 62
    rows = [
        ("Model", metrics["model_id"]),
        ("Runtime mode", metrics["runtime_mode"]),
        ("Fully GPU-resident", "Yes" if metrics["fully_gpu_resident"] else "No"),
    ]
    if metrics["runtime_mode"] == "hybrid":
        rows.extend(
            [
                ("Resident decoder layers", str(metrics["resident_decoder_layers"])),
                ("Streamed decoder layers", str(metrics["streamed_decoder_layers"])),
                ("Resident shard size", f'{metrics["resident_shard_gib"]:.2f} GiB'),
                (
                    "Streamed weights/forward",
                    f'{metrics["streamed_weight_gib_per_forward"]:.2f} GiB',
                ),
            ]
        )
    rows.extend([
        ("Input tokens", f'{metrics["input_tokens"]:,}'),
        ("Generated tokens", f'{metrics["generated_tokens"]:,}'),
        ("Generation time", f'{metrics["generation_seconds"]:.2f} seconds'),
        ("End-to-end time", f'{metrics["end_to_end_seconds"]:.2f} seconds'),
        ("Generation speed", f'{metrics["generated_tokens_per_second"]:.2f} tokens/second'),
        ("Peak CUDA allocated", f'{metrics["peak_cuda_allocated_gib"]:.2f} GiB'),
        ("Peak CUDA reserved", f'{metrics["peak_cuda_reserved_gib"]:.2f} GiB'),
    ])

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

"""Fixed-budget, profile-driven hybrid AirLLM runtime."""

from __future__ import annotations

import inspect
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from airllm import AirLLMBaseModel
from model_profiles import MODEL_PROFILES_BY_ID

SUPPORTED_AIRLLM_VERSION = "4.0.0"


def build_hybrid_partition(
    total_decoder_layers: int, resident_decoder_layers: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the resident prefix and streamed suffix decoder indices."""
    if total_decoder_layers <= 0:
        raise ValueError("total_decoder_layers must be greater than zero.")
    if not 0 < resident_decoder_layers < total_decoder_layers:
        raise ValueError(
            "resident_decoder_layers must be greater than zero and smaller than "
            f"total_decoder_layers ({total_decoder_layers}); got {resident_decoder_layers}."
        )
    return (
        tuple(range(resident_decoder_layers)),
        tuple(range(resident_decoder_layers, total_decoder_layers)),
    )


def validate_airllm_compatibility(base_class=AirLLMBaseModel) -> None:
    """Fail early when the protected AirLLM API this subclass uses has drifted."""
    try:
        installed_version = version("airllm")
    except PackageNotFoundError as exc:
        raise RuntimeError("AirLLM is not installed; version 4.0.0 is required.") from exc
    if installed_version != SUPPORTED_AIRLLM_VERSION:
        raise RuntimeError(
            "Hybrid mode requires AirLLM 4.0.0 because it relies on protected streaming "
            f"interfaces; found {installed_version}. Use the pinned requirements.txt version."
        )

    required_methods = {
        "load_layer_to_cpu": ("self", "layer_name"),
        "move_layer_to_device": ("self", "state_dict"),
        "_setup_expert_streaming": ("self",),
        "_load_streamed_layer": ("self", "idx"),
        "_pre_hook": ("self", "module", "args"),
        "_post_hook": ("self", "module", "args", "output"),
    }
    drift = []
    for name, expected_parameters in required_methods.items():
        method = getattr(base_class, name, None)
        if method is None:
            drift.append(f"missing {name}")
            continue
        actual_parameters = tuple(inspect.signature(method).parameters)
        if actual_parameters != expected_parameters:
            drift.append(f"{name}{actual_parameters!r} does not match {expected_parameters!r}")
    init_parameters = inspect.signature(base_class.__init__).parameters
    for name in ("install_hooks", "load_resident", "prefetching"):
        if name not in init_parameters:
            drift.append(f"AirLLMBaseModel.__init__ is missing {name}")
    if drift:
        raise RuntimeError(
            "AirLLM protected interface drift prevents safe hybrid loading: "
            + "; ".join(drift)
            + ". Install airllm==4.0.0 or update HybridAirLLM for the new interface."
        )


class HybridAirLLM(AirLLMBaseModel):
    """Keep a profile's fixed native-BF16 prefix resident and stream its suffix."""

    def __init__(
        self,
        model_local_path_or_repo_id: str,
        *,
        resident_decoder_layers: int | None = None,
        **kwargs,
    ) -> None:
        validate_airllm_compatibility()
        try:
            profile = MODEL_PROFILES_BY_ID[model_local_path_or_repo_id]
        except KeyError as exc:
            supported = ", ".join(repr(model_id) for model_id in MODEL_PROFILES_BY_ID)
            raise ValueError(
                f"Hybrid mode supports only {supported}; got {model_local_path_or_repo_id!r}."
            ) from exc
        requested_layers = (
            profile.resident_decoder_layers
            if resident_decoder_layers is None
            else resident_decoder_layers
        )
        if kwargs.get("compression") is not None:
            raise ValueError("Hybrid mode preserves native BF16 weights; compression is unsupported.")
        if requested_layers != profile.resident_decoder_layers:
            raise ValueError(
                f"Hybrid mode uses a fixed {profile.resident_decoder_layers}-layer resident "
                f"prefix for {profile.model_id}; got {requested_layers}. Dynamic layer selection "
                "is not supported."
            )
        build_hybrid_partition(profile.decoder_layers, requested_layers)
        self.profile = profile
        self.resident_decoder_layer_count = requested_layers
        kwargs.update(dtype=torch.bfloat16, compression=None, prefetching=True)
        super().__init__(model_local_path_or_repo_id, **kwargs)

    def _shard_size(self, layer_name: str) -> int:
        checkpoint_path = Path(self.checkpoint_path)
        candidates = [
            path
            for path in checkpoint_path.glob(f"{layer_name}.*")
            if path.is_file() and not path.name.endswith(".done")
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one AirLLM shard for {layer_name!r} in {checkpoint_path}, "
                f"found {len(candidates)}. Hybrid mode cannot validate its fixed VRAM budget."
            )
        return candidates[0].stat().st_size

    def _check_vram_budget(
        self, resident_names: tuple[str, ...], streamed_names: tuple[str, ...]
    ) -> None:
        resident_bytes = sum(self._shard_size(name) for name in resident_names)
        streamed_sizes = tuple(self._shard_size(name) for name in streamed_names)
        required_bytes = resident_bytes + max(streamed_sizes)
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        self.resident_shard_bytes = resident_bytes
        self.streamed_weight_bytes = sum(streamed_sizes)
        if free_bytes < required_bytes:
            gib = 1024**3
            raise RuntimeError(
                f"Insufficient VRAM for fixed hybrid mode with {self.profile.model_id}: requested "
                f"{self.resident_decoder_layer_count} resident decoder layers, resident shards "
                f"{resident_bytes / gib:.2f} GiB, and one active streamed layer require at least "
                f"{required_bytes / gib:.2f} GiB; cuda:0 currently has "
                f"{free_bytes / gib:.2f} GiB free of {total_bytes / gib:.2f} GiB total. "
                "No layer-count reduction or runtime fallback was attempted."
            )

    @staticmethod
    def _require_module_on_cuda(module: torch.nn.Module, label: str) -> None:
        invalid = [
            f"{name}: {parameter.device}"
            for name, parameter in module.named_parameters()
            if parameter.device != torch.device("cuda:0")
        ]
        if invalid:
            raise RuntimeError(
                f"Hybrid initialization left {label} parameters outside cuda:0: "
                + ", ".join(invalid[:5])
            )

    def _load_fixed_resident_modules(self, resident_indices: tuple[int, ...]) -> None:
        try:
            for idx in resident_indices:
                state_dict = self.load_layer_to_cpu(self.layer_names[idx])
                self.move_layer_to_device(state_dict)
                del state_dict
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            gib = 1024**3
            raise RuntimeError(
                f"Failed to load the fixed hybrid resident set for {self.profile.model_id}: "
                f"requested {self.resident_decoder_layer_count} resident decoder layers, resident "
                f"shards {self.resident_shard_bytes / gib:.2f} GiB; cuda:0 now has "
                f"{free_bytes / gib:.2f} GiB free of {total_bytes / gib:.2f} GiB total. "
                "No layer-count reduction or runtime fallback was attempted."
            ) from exc
        for idx in resident_indices:
            self._require_module_on_cuda(self.layers[idx], self.layer_names[idx])

    def _prefetch_first_streamed(self, module, args) -> None:
        del module, args
        first_streamed = self._streamed_indices[0]
        if self._prefetch_future is None:
            self._prefetch_future = self._executor.submit(
                self._load_streamed_layer, first_streamed
            )
            self._prefetched_idx = first_streamed

    def _install_streaming_hooks(self) -> None:
        total_decoder_layers = len(self.layer_names) - 3
        if total_decoder_layers != self.profile.decoder_layers:
            raise RuntimeError(
                f"Hybrid mode requires {self.profile.model_id} with exactly "
                f"{self.profile.decoder_layers} decoder layers; found {total_decoder_layers}."
            )
        resident_decoders, streamed_decoders = build_hybrid_partition(
            total_decoder_layers, self.resident_decoder_layer_count
        )
        decoder_layer_indices = tuple(range(1, total_decoder_layers + 1))
        self.resident_decoder_indices = resident_decoders
        self.streamed_decoder_indices = streamed_decoders
        resident_indices = (
            0,
            *(decoder_layer_indices[index] for index in resident_decoders),
            len(self.layer_names) - 2,
            len(self.layer_names) - 1,
        )
        self._streamed_indices = [decoder_layer_indices[index] for index in streamed_decoders]
        self._streamed_set = set(self._streamed_indices)
        resident_names = tuple(self.layer_names[index] for index in resident_indices)
        streamed_names = tuple(self.layer_names[index] for index in self._streamed_indices)
        self._check_vram_budget(resident_names, streamed_names)
        self._load_fixed_resident_modules(resident_indices)
        self.tie_word_embeddings = False
        self._setup_expert_streaming()
        for idx in self._streamed_indices:
            layer = self.layers[idx]
            layer._airllm_idx = idx
            layer.register_forward_pre_hook(self._pre_hook)
            layer.register_forward_hook(self._post_hook)
        self.layers[1].register_forward_pre_hook(self._prefetch_first_streamed)

        gib = 1024**3
        print(
            f"Hybrid layout loaded for {self.profile.model_id} in native BF16: "
            f"{self.profile.resident_decoder_layers} resident decoder layers, "
            f"{self.profile.streamed_decoder_layers} streamed decoder layers, "
            f"{self.resident_shard_bytes / gib:.2f} GiB resident shards, and "
            f"{self.streamed_weight_bytes / gib:.2f} GiB streamed per forward."
        )


# Compatibility alias for callers of the former single-model implementation.
HybridQwen3AirLLM = HybridAirLLM

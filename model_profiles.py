"""Validated benchmark model profiles and isolated shard paths."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    key: str
    model_id: str
    decoder_layers: int
    resident_decoder_layers: int
    shard_directory: str

    @property
    def streamed_decoder_layers(self) -> int:
        return self.decoder_layers - self.resident_decoder_layers


MODEL_PROFILES = {
    "llama": ModelProfile(
        "llama", "meta-llama/Llama-3.1-8B-Instruct", 32, 16, "llama-3.1-8b-instruct"
    ),
    "qwen": ModelProfile("qwen", "Qwen/Qwen3-14B", 40, 24, "qwen3-14b"),
}
MODEL_PROFILES_BY_ID = {profile.model_id: profile for profile in MODEL_PROFILES.values()}


def get_model_profile(profile_key: str) -> ModelProfile:
    try:
        return MODEL_PROFILES[profile_key]
    except KeyError as exc:
        choices = ", ".join(repr(key) for key in MODEL_PROFILES)
        raise ValueError(f"Unsupported MODEL_PROFILE {profile_key!r}. Choose {choices}.") from exc


def model_shard_path(shard_root: Path, profile: ModelProfile) -> Path:
    return shard_root / profile.shard_directory

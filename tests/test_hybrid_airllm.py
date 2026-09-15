import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

from airllm import AirLLMBaseModel

from hybrid_airllm import (
    HybridAirLLM,
    build_hybrid_partition,
    validate_airllm_compatibility,
)
from model_profiles import MODEL_PROFILES, get_model_profile, model_shard_path


class ModelProfileTests(unittest.TestCase):
    def test_checked_in_profiles_have_fixed_partitions(self):
        expected = {
            "llama": ("meta-llama/Llama-3.1-8B-Instruct", 32, 16, 16),
            "qwen": ("Qwen/Qwen3-14B", 40, 24, 16),
        }

        for key, values in expected.items():
            with self.subTest(profile=key):
                profile = get_model_profile(key)
                self.assertEqual(
                    (
                        profile.model_id,
                        profile.decoder_layers,
                        profile.resident_decoder_layers,
                        profile.streamed_decoder_layers,
                    ),
                    values,
                )

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported MODEL_PROFILE"):
            get_model_profile("unknown")

    def test_each_profile_has_an_isolated_shard_path(self):
        root = Path("shards")
        paths = {model_shard_path(root, profile) for profile in MODEL_PROFILES.values()}

        self.assertEqual(len(paths), len(MODEL_PROFILES))
        self.assertTrue(all(path.parent == root for path in paths))


class HybridPartitionTests(unittest.TestCase):
    def test_profile_partitions_are_complete_and_disjoint(self):
        for profile in MODEL_PROFILES.values():
            with self.subTest(profile=profile.key):
                resident, streamed = build_hybrid_partition(
                    profile.decoder_layers, profile.resident_decoder_layers
                )

                self.assertEqual(resident, tuple(range(profile.resident_decoder_layers)))
                self.assertEqual(
                    streamed,
                    tuple(range(profile.resident_decoder_layers, profile.decoder_layers)),
                )
                self.assertEqual(
                    set(resident) | set(streamed), set(range(profile.decoder_layers))
                )
                self.assertFalse(set(resident) & set(streamed))

    def test_invalid_resident_counts_are_rejected(self):
        for count in (-1, 0, 40, 41):
            with self.subTest(count=count), self.assertRaisesRegex(
                ValueError, "resident_decoder_layers"
            ):
                build_hybrid_partition(40, count)

    def test_runtime_rejects_non_fixed_valid_counts_for_both_profiles(self):
        for profile in MODEL_PROFILES.values():
            with self.subTest(profile=profile.key), self.assertRaisesRegex(
                ValueError, f"fixed {profile.resident_decoder_layers}-layer resident prefix"
            ):
                HybridAirLLM(
                    profile.model_id,
                    resident_decoder_layers=profile.resident_decoder_layers - 1,
                )

    def test_runtime_rejects_unsupported_model(self):
        with self.assertRaisesRegex(ValueError, "supports only"):
            HybridAirLLM("unvalidated/model")


class AirLLMCompatibilityTests(unittest.TestCase):
    def test_installed_airllm_interface_is_compatible(self):
        validate_airllm_compatibility()

    def test_interface_drift_has_actionable_error(self):
        class DriftedBase(AirLLMBaseModel):
            def _pre_hook(self, module):
                return module

        with self.assertRaisesRegex(RuntimeError, "protected interface drift"):
            validate_airllm_compatibility(DriftedBase)


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, function, index):
        self.calls.append((function, index))
        future = Future()
        future.set_result({})
        return future


class HybridPrefetchTests(unittest.TestCase):
    def test_first_streamed_layer_prefetch_starts_from_resident_prefix(self):
        model = object.__new__(HybridAirLLM)
        model._streamed_indices = [25, *range(26, 41)]
        model._prefetch_future = None
        model._prefetched_idx = None
        model._executor = FakeExecutor()
        model._load_streamed_layer = lambda index: {"index": index}

        model._prefetch_first_streamed(None, ())

        self.assertEqual(model._prefetched_idx, 25)
        self.assertEqual(len(model._executor.calls), 1)
        self.assertEqual(model._executor.calls[0][1], 25)

    def test_streamed_layer_keeps_airllm_prefetch_chain(self):
        model = object.__new__(HybridAirLLM)
        model.prefetching = True
        model._streamed_indices = [25, 26]
        model._streamed_set = {25, 26}
        model._prefetched_idx = 25
        first = Future()
        first.set_result({"first": object()})
        model._prefetch_future = first
        model._executor = FakeExecutor()
        model._expert_streaming = False
        model.move_layer_to_device = lambda state: list(state)
        model._load_streamed_layer = lambda index: {f"layer-{index}": object()}

        class Module:
            _airllm_idx = 25

        module = Module()
        model._pre_hook(module, ())

        self.assertEqual(module._airllm_moved, ["first"])
        self.assertEqual(model._prefetched_idx, 26)
        self.assertEqual(model._executor.calls[0][1], 26)


class HybridVramTests(unittest.TestCase):
    def test_measured_shard_sizes_are_recorded(self):
        model = object.__new__(HybridAirLLM)
        model.device = "cuda:0"
        model.profile = MODEL_PROFILES["llama"]
        model.resident_decoder_layer_count = 16
        sizes = {"boundary": 2 * 1024**3, "resident": 8 * 1024**3, "streamed": 3 * 1024**3}
        model._shard_size = sizes.__getitem__

        with patch("torch.cuda.mem_get_info", return_value=(20 * 1024**3, 24 * 1024**3)):
            model._check_vram_budget(("boundary", "resident"), ("streamed",))

        self.assertEqual(model.resident_shard_bytes, 10 * 1024**3)
        self.assertEqual(model.streamed_weight_bytes, 3 * 1024**3)

    def test_vram_exhaustion_does_not_fallback(self):
        model = object.__new__(HybridAirLLM)
        model.device = "cuda:0"
        model.profile = MODEL_PROFILES["qwen"]
        model.resident_decoder_layer_count = 24
        sizes = {"resident": 18 * 1024**3, "streamed": 1024**3}
        model._shard_size = sizes.__getitem__

        with patch("torch.cuda.mem_get_info", return_value=(17 * 1024**3, 24 * 1024**3)):
            with self.assertRaisesRegex(
                RuntimeError,
                "requested 24 resident decoder layers.*No layer-count reduction or runtime fallback",
            ):
                model._check_vram_budget(("resident",), ("streamed",))


if __name__ == "__main__":
    unittest.main()

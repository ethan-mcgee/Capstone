import unittest
from concurrent.futures import Future
from unittest.mock import patch

from airllm import AirLLMBaseModel

from hybrid_airllm import (
    HybridQwen3AirLLM,
    build_hybrid_partition,
    validate_airllm_compatibility,
)


class HybridPartitionTests(unittest.TestCase):
    def test_fixed_partition_is_complete_and_disjoint(self):
        resident, streamed = build_hybrid_partition(40, 24)

        self.assertEqual(resident, tuple(range(24)))
        self.assertEqual(streamed, tuple(range(24, 40)))
        self.assertEqual(set(resident) | set(streamed), set(range(40)))
        self.assertFalse(set(resident) & set(streamed))

    def test_invalid_resident_counts_are_rejected(self):
        for count in (-1, 0, 40, 41):
            with self.subTest(count=count), self.assertRaisesRegex(
                ValueError, "resident_decoder_layers"
            ):
                build_hybrid_partition(40, count)

    def test_runtime_rejects_a_non_fixed_valid_count(self):
        with self.assertRaisesRegex(ValueError, "fixed 24-layer resident prefix"):
            HybridQwen3AirLLM(
                "Qwen/Qwen3-14B",
                resident_decoder_layers=23,
            )


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
        model = object.__new__(HybridQwen3AirLLM)
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
        model = object.__new__(HybridQwen3AirLLM)
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
    def test_vram_exhaustion_does_not_fallback(self):
        model = object.__new__(HybridQwen3AirLLM)
        model.device = "cuda:0"
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

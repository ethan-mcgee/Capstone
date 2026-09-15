import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import airllm_test
from model_profiles import MODEL_PROFILES


class BenchmarkConfigurationTests(unittest.TestCase):
    def test_checked_in_defaults_select_llama_airllm(self):
        self.assertEqual(airllm_test.MODEL_PROFILE, "llama")
        self.assertEqual(airllm_test.RUNTIME_MODE, "airllm")
        self.assertEqual(airllm_test.MODEL, MODEL_PROFILES["llama"])
        self.assertEqual(
            airllm_test.AIRLLM_SHARD_PATH.name,
            MODEL_PROFILES["llama"].shard_directory,
        )

    def test_hybrid_metrics_use_measured_model_values(self):
        gib = 1024**3
        model = SimpleNamespace(
            resident_shard_bytes=11 * gib,
            streamed_weight_bytes=4 * gib,
            generate=lambda *_args, **_kwargs: SimpleNamespace(
                sequences=torch.tensor([[1, 2, 3, 4]])
            ),
        )
        tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "response")
        input_ids = torch.tensor([[1, 2]])

        with (
            patch.object(airllm_test, "RUNTIME_MODE", "hybrid"),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.max_memory_allocated", return_value=0),
            patch("torch.cuda.max_memory_reserved", return_value=0),
        ):
            _, metrics = airllm_test.generate_response(model, tokenizer, input_ids)

        self.assertEqual(metrics["model_id"], MODEL_PROFILES["llama"].model_id)
        self.assertEqual(metrics["resident_decoder_layers"], 16)
        self.assertEqual(metrics["streamed_decoder_layers"], 16)
        self.assertEqual(metrics["resident_shard_gib"], 11)
        self.assertEqual(metrics["streamed_weight_gib_per_forward"], 4)

    def test_results_include_model_identity_and_hybrid_layout(self):
        metrics = {
            "model_id": MODEL_PROFILES["llama"].model_id,
            "runtime_mode": "hybrid",
            "fully_gpu_resident": False,
            "resident_decoder_layers": 16,
            "streamed_decoder_layers": 16,
            "resident_shard_gib": 11.0,
            "streamed_weight_gib_per_forward": 4.0,
            "input_tokens": 10,
            "generated_tokens": 3,
            "generation_seconds": 1.0,
            "end_to_end_seconds": 2.0,
            "generated_tokens_per_second": 3.0,
            "peak_cuda_allocated_gib": 12.0,
            "peak_cuda_reserved_gib": 13.0,
        }

        with patch("sys.stdout", new_callable=io.StringIO) as output:
            airllm_test.print_results("response", metrics)

        rendered = output.getvalue()
        self.assertIn(MODEL_PROFILES["llama"].model_id, rendered)
        self.assertIn("Resident decoder layers     16", rendered)
        self.assertIn("Streamed decoder layers     16", rendered)
        self.assertIn("Resident shard size         11.00 GiB", rendered)
        self.assertIn("Streamed weights/forward    4.00 GiB", rendered)


if __name__ == "__main__":
    unittest.main()

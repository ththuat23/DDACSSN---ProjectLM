
import unittest
import torch

from decoding.beam_search import beam_search_from_logits


class TestBeamSearchLogits(unittest.TestCase):

    def test_select_top_tokens(self):
        logits = torch.tensor([[1.0, 3.0, 2.0, 0.5]])

        top_log_probs, top_tokens = beam_search_from_logits(
            logits,
            beam_width=2
        )

        self.assertEqual(top_tokens.tolist(), [[1, 2]])
        self.assertEqual(top_log_probs.shape, (1, 2))

    def test_log_probs_normalized(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        top_log_probs, top_tokens = beam_search_from_logits(
            logits,
            beam_width=3
        )

        total_probability = torch.exp(top_log_probs).sum().item()

        self.assertAlmostEqual(total_probability, 1.0, places=5)

    def test_invalid_beam_width(self):
        logits = torch.tensor([[1.0, 2.0]])

        with self.assertRaises(ValueError):
            beam_search_from_logits(logits, 0)


if __name__ == "__main__":
    unittest.main()

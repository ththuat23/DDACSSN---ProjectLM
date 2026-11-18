import unittest
import torch

from decoding.greedy import greedy_from_logits

from decoding.temperature import (
    temperature_probs,
    temperature_from_logits
)

from decoding.top_k import (
    top_k_filter_logits,
    top_k_probs,
    top_k_from_logits
)

from decoding.top_p import (
    top_p_filter_logits,
    top_p_probs,
    top_p_from_logits
)


class TestLogits(unittest.TestCase):

    def setUp(self):
        # Vector logits giả lập
        self.logits = torch.tensor([
            [4.0, 3.0, 2.0, 1.0]
        ])

    def test_greedy(self):
        token = greedy_from_logits(self.logits)

        self.assertEqual(token.item(), 0)

    def test_temperature(self):
        low_temp = temperature_probs(
            self.logits, 0.5
        )

        high_temp = temperature_probs(
            self.logits, 2.0
        )

        # Tổng xác suất phải bằng 1
        self.assertTrue(
            torch.allclose(
                low_temp.sum(dim=-1),
                torch.tensor([1.0])
            )
        )

        # Nhiệt độ thấp tạo phân phối tập trung hơn
        self.assertGreater(
            low_temp[0, 0].item(),
            high_temp[0, 0].item()
        )

    def test_top_k(self):
        filtered = top_k_filter_logits(
            self.logits, 2
        )

        probs = top_k_probs(
            self.logits, 2
        )

        # Chỉ còn 2 token được giữ
        self.assertEqual(
            torch.isfinite(filtered).sum().item(),
            2
        )

        # Tổng xác suất bằng 1
        self.assertAlmostEqual(
            probs.sum().item(),
            1.0,
            places=5
        )

        # Token bị loại có xác suất bằng 0
        self.assertEqual(probs[0, 2].item(), 0.0)
        self.assertEqual(probs[0, 3].item(), 0.0)

    def test_top_p(self):
        filtered = top_p_filter_logits(
            self.logits, 0.8
        )

        probs = top_p_probs(
            self.logits, 0.8
        )

        # Với vector này, top-p=0.8 giữ lại 2 token
        self.assertEqual(
            torch.isfinite(filtered).sum().item(),
            2
        )

        self.assertAlmostEqual(
            probs.sum().item(),
            1.0,
            places=5
        )

    def test_fixed_seed(self):
        # Kiểm tra Temperature
        torch.manual_seed(42)
        result1 = temperature_from_logits(
            self.logits, 1.0
        )

        torch.manual_seed(42)
        result2 = temperature_from_logits(
            self.logits, 1.0
        )

        self.assertTrue(torch.equal(result1, result2))

        # Kiểm tra Top-k
        torch.manual_seed(42)
        result1 = top_k_from_logits(
            self.logits, 2
        )

        torch.manual_seed(42)
        result2 = top_k_from_logits(
            self.logits, 2
        )

        self.assertTrue(torch.equal(result1, result2))

        # Kiểm tra Top-p
        torch.manual_seed(42)
        result1 = top_p_from_logits(
            self.logits, 0.8
        )

        torch.manual_seed(42)
        result2 = top_p_from_logits(
            self.logits, 0.8
        )

        self.assertTrue(torch.equal(result1, result2))


if __name__ == "__main__":
    unittest.main()
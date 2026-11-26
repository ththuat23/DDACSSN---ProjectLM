import json
import csv

from pathlib import Path

from models.model_loader import load_medium_model

from evaluation.distinct_n import calculate_distinct_scores
from evaluation.repetition import repetition_rate
from evaluation.perplexity import calculate_perplexity

# ĐƯỜNG DẪN
BASE_DIR = Path(__file__).resolve().parent

RAW_FILE = BASE_DIR / "results" / "raw_results.json"

DETAIL_FILE = (
    BASE_DIR / "results" / "evaluation_details.json"
)

SUMMARY_FILE = (
    BASE_DIR / "results" / "evaluation_summary.csv"
)

# ĐỌC KẾT QUẢ SINH VĂN BẢN
def load_raw_results():

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            "Không tìm thấy raw_results.json. "
            "Hãy chạy run_experiments_2.py trước."
        )

    with open(RAW_FILE, "r", encoding="utf-8") as file:
        results = json.load(file)

    if not results:
        raise ValueError("File raw_results.json không có dữ liệu")

    return results

# ĐÁNH GIÁ TỪNG VĂN BẢN
def evaluate_text(
    eval_model,
    eval_tokenizer,
    prompt,
    generated_text
):

    # 1. Distinct-n
    distinct = calculate_distinct_scores(
        generated_text
    )

    # 2. Repetition Rate
    repetition = repetition_rate(
        generated_text,
        n=3
    )

    # 3. Perplexity
    perplexity = calculate_perplexity(
        eval_model,
        eval_tokenizer,
        prompt,
        generated_text
    )

    return {
        "distinct_1": distinct["distinct_1"],
        "distinct_2": distinct["distinct_2"],
        "distinct_3": distinct["distinct_3"],
        "repetition_rate": repetition,
        "perplexity": perplexity
    }

# TỔNG HỢP THEO PHƯƠNG PHÁP
def calculate_average(values):

    valid_values = [
        value for value in values
        if isinstance(value, (int, float))
    ]

    if not valid_values:
        return None

    return sum(valid_values) / len(valid_values)


def summarize_results(results):

    grouped = {}

    for item in results:

        method = item["method"]

        grouped.setdefault(method, []).append(item)

    metric_names = [
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "repetition_rate",
        "perplexity",

        # Performance
        "generation_time_sec",
        "generated_tokens",
        "throughput_tokens_sec",
        "ram_before_mb",
        "ram_after_mb",
        "ram_delta_mb",
        "vram_peak_mb",
        "vram_extra_peak_mb"
    ]

    summary = []

    for method, records in grouped.items():

        row = {
            "method": method,
            "samples": len(records)
        }

        for metric in metric_names:

            values = []

            for record in records:

                if metric in record:
                    value = record[metric]

                else:
                    value = record.get(
                        "performance", {}
                    ).get(metric)

                values.append(value)

            row[metric] = calculate_average(values)

        summary.append(row)

    return summary

# LƯU KẾT QUẢ
def save_details(results):

    DETAIL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(DETAIL_FILE, "w", encoding="utf-8") as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )


def save_summary(summary):

    if not summary:
        return

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys()
        )

        writer.writeheader()
        writer.writerows(summary)

# CHƯƠNG TRÌNH CHÍNH
def main():

    results = load_raw_results()

    print("Number of generated samples:", len(results))

    # Chỉ tải GPT-2 Medium ở giai đoạn Evaluation
    print("\nLoading GPT-2 Medium for Perplexity...")

    eval_model, eval_tokenizer = load_medium_model()

    evaluated_results = []

    for index, item in enumerate(results, start=1):

        print(
            f"\nEvaluating [{index}/{len(results)}]: "
            f"{item['method']}"
        )

        generated_text = item["generated_text"]

        if not generated_text.strip():
            print("Warning: Văn bản sinh ra đang rỗng")
            print("Skipping this sample.")
            continue

        metrics = evaluate_text(
            eval_model,
            eval_tokenizer,
            item["prompt"],
            generated_text
        )

        print(
            f"Distinct-1: {metrics['distinct_1']:.4f}"
        )

        print(
            f"Distinct-2: {metrics['distinct_2']:.4f}"
        )

        print(
            f"Repetition Rate: "
            f"{metrics['repetition_rate']:.2f}%"
        )

        print(
            f"Perplexity: {metrics['perplexity']:.4f}"
        )

        # Giữ dữ liệu gốc và bổ sung các chỉ số đánh giá
        record = item.copy()

        record.update(metrics)

        evaluated_results.append(record)

    # Tổng hợp kết quả
    summary = summarize_results(
        evaluated_results
    )

    # Lưu hai loại kết quả
    save_details(evaluated_results)
    save_summary(summary)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    for row in summary:

        print("\nMethod:", row["method"])
        print("Samples:", row["samples"])

        for metric in [
            "distinct_1",
            "distinct_2",
            "distinct_3",
            "repetition_rate",
            "perplexity",
            "generation_time_sec",
            "generated_tokens",
            "throughput_tokens_sec",
            "ram_delta_mb",
            "vram_peak_mb",
            "vram_extra_peak_mb"
        ]:

            value = row[metric]

            if value is not None:
                print(f"{metric}: {value:.4f}")

    print("\nSaved files:")
    print(DETAIL_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
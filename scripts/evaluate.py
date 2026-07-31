"""Evaluate PlainMed base and fine-tuned model readability."""

import csv
import json
import os
from pathlib import Path
from statistics import mean
from typing import Dict, List

import textstat


TEST_FILE = Path("data/processed/test.jsonl")
BASE_MODEL = "gpt-4o-mini"
MODEL_ID_FILE = Path("models/model_id.txt")
OUTPUT_DIR = Path("data/outputs")
RESULTS_FILE = OUTPUT_DIR / "eval_results.csv"
CHART_FILE = OUTPUT_DIR / "readability_chart.png"
CSV_COLUMNS = [
    "original",
    "reference_simple",
    "base_output",
    "finetuned_output",
    "fk_original",
    "fk_base",
    "fk_finetuned",
]


def get_predict_function():
    """Load the shared prediction helper for package or direct script execution."""
    try:
        from scripts.model import predict
    except ModuleNotFoundError:
        from model import predict
    return predict


def load_jsonl(path: Path = TEST_FILE) -> List[Dict[str, str]]:
    """Load held-out expert and simple pairs from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_model_id(path: Path = MODEL_ID_FILE) -> str:
    """Read the fine-tuned model ID from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Fine-tuned model ID not found: {path}")
    model_id = path.read_text(encoding="utf-8").strip()
    if not model_id.startswith("ft:"):
        raise ValueError(f"Expected a fine-tuned model ID starting with 'ft:', got: {model_id}")
    return model_id


def fk_grade(text: str) -> float:
    """Compute Flesch-Kincaid grade level for text."""
    return round(float(textstat.flesch_kincaid_grade(text)), 2)


def evaluate_rows(test_rows: List[Dict[str, str]], finetuned_model_id: str) -> List[Dict[str, object]]:
    """Run base and fine-tuned models over all held-out rows and collect readability metrics."""
    results = []
    predict = get_predict_function()

    for index, row in enumerate(test_rows, start=1):
        original = row["expert"]
        reference_simple = row["simple"]
        print(f"Evaluating row {index}/{len(test_rows)}")

        base_output = predict(original, BASE_MODEL)
        finetuned_output = predict(original, finetuned_model_id)

        results.append(
            {
                "original": original,
                "reference_simple": reference_simple,
                "base_output": base_output,
                "finetuned_output": finetuned_output,
                "fk_original": fk_grade(original),
                "fk_base": fk_grade(base_output),
                "fk_finetuned": fk_grade(finetuned_output),
            }
        )

    return results


def save_results_csv(rows: List[Dict[str, object]], path: Path = RESULTS_FILE) -> None:
    """Save evaluation rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved evaluation CSV to {path}")


def mean_scores(rows: List[Dict[str, object]]) -> Dict[str, float]:
    """Calculate mean FK grades for original, base, and fine-tuned outputs."""
    return {
        "original": round(mean(float(row["fk_original"]) for row in rows), 2),
        "base": round(mean(float(row["fk_base"]) for row in rows), 2),
        "fine_tuned": round(mean(float(row["fk_finetuned"]) for row in rows), 2),
    }


def print_summary(rows: List[Dict[str, object]]) -> None:
    """Print mean FK grades and three before/after examples."""
    scores = mean_scores(rows)
    print("\nMean Flesch-Kincaid grade")
    print("--------------------------")
    for label, value in scores.items():
        print(f"{label:12} {value}")

    print("\nExamples")
    print("--------")
    for row in rows[:3]:
        print(f"Original: {row['original']}")
        print(f"Base: {row['base_output']}")
        print(f"Fine-tuned: {row['finetuned_output']}")
        print()

    if scores["fine_tuned"] >= scores["original"]:
        print("Note: mean fk_finetuned is not lower than fk_original.")


def save_chart(rows: List[Dict[str, object]], path: Path = CHART_FILE) -> None:
    """Save a bar chart of mean FK grade levels."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/plainmed_matplotlib")
    import matplotlib.pyplot as plt

    scores = mean_scores(rows)
    labels = ["Original", "Base", "Fine-tuned"]
    values = [scores["original"], scores["base"], scores["fine_tuned"]]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, values, color=["#6b7280", "#2563eb", "#16a34a"])
    plt.ylabel("Mean FK grade")
    plt.title("PlainMed Readability Comparison")
    plt.ylim(0, max(values) + 2)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(value), ha="center")

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved readability chart to {path}")


def main() -> None:
    """Run the full evaluation workflow."""
    test_rows = load_jsonl()
    finetuned_model_id = load_model_id()
    results = evaluate_rows(test_rows, finetuned_model_id)
    save_results_csv(results)
    print_summary(results)
    save_chart(results)


if __name__ == "__main__":
    main()

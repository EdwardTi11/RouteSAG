import csv
import re
import time
import statistics
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dspy.utils.exceptions import AdapterParseError
from model_design.model import init_dspy_model
from model_design.router import RouteSAGRouter

CSV_PATH = "FreshQA_v042126.csv"
EVAL_SPLIT = "TEST"  # None = all rows
YEAR_THRESHOLD = 2025

LABELS = ["static", "search"]
ALL_LABELS = ["static", "search", "invalid"]

def expected_route(fact_type, effective_year):
    if fact_type == "never-changing":
        return "static"
    if fact_type == "fast-changing":
        return "search"
    if fact_type == "slow-changing":
        match = re.search(r"\d{4}", str(effective_year))
        if not match:
            raise ValueError(f"Invalid effective_year: {effective_year}")
        year = int(match.group())
        if str(effective_year).strip().lower().startswith("before "):
            year -= 1
        return "static" if year < YEAR_THRESHOLD else "search"

    raise ValueError(f"Unknown fact_type: {fact_type}")

def load_data():
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        f.readline()
        f.readline()
        reader = csv.DictReader(f)

        samples = []
        for row in reader:
            if EVAL_SPLIT and row["split"].strip().upper() != EVAL_SPLIT:
                continue

            question = (row["question"] or "").strip()
            if not question:
                continue

            samples.append({
                "question": question,
                "fact_type": row["fact_type"].strip(),
                "effective_year": row["effective_year"].strip(),
                "expected": expected_route(
                    row["fact_type"].strip(),
                    row["effective_year"],
                ),
            })

    return samples

def main():
    samples = load_data()
    print(f"Loaded {len(samples)} questions ({EVAL_SPLIT or 'ALL'} split).")

    init_dspy_model()
    router = RouteSAGRouter()

    expected, predicted, latencies = [], [], []
    category_stats = defaultdict(lambda: [0, 0])

    for i, sample in enumerate(samples, 1):
        start = time.perf_counter()

        try:
            raw = str(router(question=sample["question"])).strip().lower()
            prediction = raw.strip(" '\".,-")
            if prediction not in LABELS:
                prediction = "invalid"
        except AdapterParseError:
            prediction = "invalid"

        latency = time.perf_counter() - start
        target = sample["expected"]

        expected.append(target)
        predicted.append(prediction)
        latencies.append(latency)

        stats = category_stats[sample["fact_type"]]
        stats[1] += 1
        stats[0] += prediction == target

        print(
            f"[{i}/{len(samples)}] "
            f"Expected: {target:6} | Predicted: {prediction:7} | "
            f"Type: {sample['fact_type']:15} | "
            f"Year: {sample['effective_year']} | {latency:.2f}s"
        )

    invalid = predicted.count("invalid")

    print("\n========== Results ==========")
    print(f"Accuracy:          {accuracy_score(expected, predicted):.4f}")
    print(f"Macro F1:          {f1_score(expected, predicted, labels=LABELS, average='macro', zero_division=0):.4f}")
    print(f"Search precision:  {precision_score(expected, predicted, labels=['search'], average='micro', zero_division=0):.4f}")
    print(f"Search recall:     {recall_score(expected, predicted, labels=['search'], average='micro', zero_division=0):.4f}")
    print(f"Invalid rate:      {invalid / len(samples):.2%}")
    print(f"Median latency:    {statistics.median(latencies):.3f}s")

    print("\n========== Accuracy by Fact Type ==========")
    for category, (correct, total) in sorted(category_stats.items()):
        print(f"{category:16} {correct / total:.4f} ({correct}/{total})")

    print("\n========== Classification Report ==========")
    print(classification_report(expected, predicted, labels=ALL_LABELS, zero_division=0))

    print("========== Confusion Matrix ==========")
    print(f"Label order: {ALL_LABELS}")
    print(confusion_matrix(expected, predicted, labels=ALL_LABELS))

if __name__ == "__main__":
    main()
import random
import time

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from inspect_evals.simpleqa import simpleqa_verified
from inspect_evals.browse_comp import browse_comp
from router import RouteSAGRouter

SEED = 42
LIMIT = None  # Set to 10 for quick testing

def main():
    router = RouteSAGRouter()
    samples = []

    for benchmark, name, label in [
        (simpleqa_verified(), "SimpleQA", "local"),
        (browse_comp(), "BrowseComp", "search"),
    ]:
        data = list(benchmark.dataset)
        random.Random(SEED).shuffle(data)

        for sample in data[:LIMIT]:
            samples.append((sample.input, label, name))

    random.Random(SEED).shuffle(samples)

    expected, predicted, latencies = [], [], []
    results = {"SimpleQA": [0, 0], "BrowseComp": [0, 0]}

    for question, label, source in samples:
        start = time.perf_counter()
        output = str(router(question)).strip().lower()
        latencies.append(time.perf_counter() - start)

        prediction = output if output in ("local", "search") else "invalid"
        expected.append(label)
        predicted.append(prediction)

        results[source][1] += 1
        results[source][0] += prediction == label

    print(f"\nSamples: {len(samples)}")
    print(f"Accuracy: {accuracy_score(expected, predicted):.2%}")
    print(f"Macro-F1: {f1_score(expected, predicted, labels=['local', 'search'], average='macro', zero_division=0):.4f}")
    print(f"Average latency: {sum(latencies) / len(latencies):.4f}s")

    print("\nClassification Report:")
    print(classification_report(expected, predicted, labels=["local", "search"], zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(expected, predicted, labels=["local", "search"]))

    print("\nBenchmark Accuracy:")
    for name, (correct, total) in results.items():
        print(f"{name}: {correct}/{total} ({correct / total:.2%})" if total else f"{name}: No samples")


if __name__ == "__main__":
    main()
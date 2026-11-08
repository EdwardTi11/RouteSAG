import random
import time
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from router import RouteSAGRouter
from model import init_dspy_model

SEED, LABELS = 42, ["local", "search"]

def load_data():
    samples = []
    configs = [
        ("google/simpleqa-verified", "eval", "SimpleQA Verified", "local"),
        ("smolagents/browse_comp", "test", "BrowseComp", "search")
    ]
    for name, split, source_name, intent in configs:
        try:
            data = list(load_dataset(name, split=split))
            random.Random(SEED).shuffle(data)
            for item in data:
                row = dict(item)
                q = row.get("problem") or row.get("question") or row.get("input")
                if q:
                    samples.append({"question": str(q), "expected": intent, "source": source_name})
        except Exception:
            pass
    random.Random(SEED).shuffle(samples)
    return samples

def main():
    init_dspy_model()
    router = RouteSAGRouter()
    samples = load_data()
    
    expected, predicted, latencies = [], [], []
    breakdown = {"SimpleQA Verified": {"correct": 0, "total": 0}, "BrowseComp": {"correct": 0, "total": 0}}

    for i, s in enumerate(samples, 1):
        start = time.perf_counter()
        raw = str(router(s["question"])).strip().lower()
        latencies.append(time.perf_counter() - start)
        
        pred = raw if raw in LABELS else "invalid"
        expected.append(s["expected"])
        predicted.append(pred)
        
        if s["source"] in breakdown:
            breakdown[s["source"]]["total"] += 1
            if pred == s["expected"]:
                breakdown[s["source"]]["correct"] += 1
            
        print(f"[{i}/{len(samples)}] {'✓' if pred == s['expected'] else '✗'} [{s['source']}] {s['expected']} -> {pred}")

    print(f"\nAccuracy: {accuracy_score(expected, predicted):.2%}")
    print(f"Macro-F1: {f1_score(expected, predicted, labels=LABELS, average='macro'):.4f}")
    print(f"Avg Latency: {sum(latencies)/len(latencies):.4f}s\n")
    print(classification_report(expected, predicted, labels=LABELS, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(expected, predicted, labels=LABELS))

if __name__ == "__main__":
    main()
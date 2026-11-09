import random
import time
import statistics
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from router import RouteSAGRouter
from model import init_dspy_model
from dspy.utils.exceptions import AdapterParseError

SEED, LABELS = 42, ["static", "search"]

def load_data():
    samples = []
    configs = [
        ("google/simpleqa-verified", "eval", "SimpleQA Verified", "static"),
        ("smolagents/browse_comp", "test", "BrowseComp", "search")
    ]
    for name, split_str, source_name, intent in configs:
        try:
            data = list(load_dataset(name, split=split_str))
            random.Random(SEED).shuffle(data)
            for item in data:
                row = dict(item)
                q = row.get("problem") or row.get("question") or row.get("input")
                if q and isinstance(q, str):
                    clean_q = q.strip()
                    # Filter out base64-like blobs, extremely long gibberish, or empty strings
                    if len(clean_q) > 3 and not clean_q.startswith("Y0cJn"): 
                        samples.append({"question": clean_q, "expected": intent, "source": source_name})
        except Exception as e:
            print(f"Error loading {name} ({split_str}): {e}")
            pass
    random.Random(SEED).shuffle(samples)
    return samples

def main():
    init_dspy_model()
    router = RouteSAGRouter()
    
    samples = load_data()
    print(f"Loaded {len(samples)} total samples across datasets.\n")
    
    expected, predicted, latencies = [], [], []
    
    # Per-dataset statistics tracking
    breakdown = {
        "SimpleQA Verified": {"correct": 0, "total": 0}, 
        "BrowseComp": {"correct": 0, "total": 0}
    }

    for i, s in enumerate(samples, 1):
        start = time.perf_counter()
        try:
            raw = str(router(s["question"])).strip().lower()
            # Strip common LLM punctuation/quotes to prevent false "invalid" flags
            clean_raw = raw.strip(" '\".,-")
            pred = clean_raw if clean_raw in LABELS else "invalid"
        except AdapterParseError:
            pred = "invalid"  # Catch LLM parsing breakdowns gracefully
            
        latencies.append(time.perf_counter() - start)
        
        expected.append(s["expected"])
        predicted.append(pred)
        
        # Track per-dataset statistics
        source = s["source"]
        if source in breakdown:
            breakdown[source]["total"] += 1
            if pred == s["expected"]:
                breakdown[source]["correct"] += 1
            
        print(f"[{i}/{len(samples)}] {'✓' if pred == s['expected'] else '✗'} [{source}] Expected: {s['expected']} -> Predicted: {pred}")

    # --- Overall Metrics ---
    print("\n" + "="*40)
    print("GLOBAL EVALUATION RESULTS")
    print("="*40)
    print(f"Overall Accuracy: {accuracy_score(expected, predicted):.2%}")
    print(f"Macro-F1: {f1_score(expected, predicted, labels=LABELS, average='macro'):.4f}")
    print(f"Median Latency: {statistics.median(latencies):.4f}s\n")

    # --- Per-Dataset Statistics Breakdown ---
    print("="*40)
    print("PER-DATASET BREAKDOWN STATISTICS")
    print("="*40)
    for source_name, stats in breakdown.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = (correct / total) if total > 0 else 0.0
        print(f"* **{source_name}**:")
        print(f"  - Correct / Total: {correct} / {total}")
        print(f"  - Accuracy: {acc:.2%}")

    print("\n" + "="*40)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*40)
    print(classification_report(expected, predicted, labels=LABELS, zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(expected, predicted, labels=LABELS))

if __name__ == "__main__":
    main()
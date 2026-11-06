Here is your updated `README.md` reflecting the migration to `duckduckgo-search`, the removal of `requests`, the robust `Predict`-based regex sanitization in `router.py`, and the simplified direct connection setup.

```markdown
# RouteSAG 🌿🤖

**RouteSAG** (Route-based System for Agentic Generation) is a lightweight, hybrid AI assistant that combines a local GGUF model with dynamic query routing and live web search via **duckduckgo-search**. 

Instead of relying on heavy external infrastructure or API keys, RouteSAG uses **DSPy** to route queries intelligently—using fast local inference for general prompts and a keyless, privacy-oriented web search only when up-to-date data is required.

---

## 📐 Architecture & Workflow

```text
User Query ──> [router.py] (intent: search or local)
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       (local)               (search)
          │                     │
          │                [search.py]
          │             (top 3 snippets via DDGS)
          │                     │
          └──────────┬──────────┘
                     ▼
        [model.py] Qwen3-4B generates the answer
                     ▼
                Final Answer

```

| Module | Role |
| --- | --- |
| `main.py` | CLI entry point and execution loop. |
| `model.py` | Initializes local GGUF inference via `llama-cpp-python` and configures DSPy. |
| `router.py` | DSPy `Predict`-based intent classifier with regex tag sanitization (`search` vs `local`). |
| `search.py` | Keyless, direct DuckDuckGo search integration via `duckduckgo-search`. |

---

## 🚀 Quick Start

### 1. Prerequisites

* **Python 3.10+**
* A compatible GGUF model (e.g., Qwen3-4B-Q4_K_M)

### 2. Installation

```bash
git clone [https://github.com/your-username/RouteSAG.git](https://github.com/your-username/RouteSAG.git)
cd RouteSAG

python -m venv venv
# Activate virtual environment:
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install dspy llama-cpp-python duckduckgo-search

```

### 3. Setup Model Weights

Place your target GGUF file in the `./models/` directory:

```text
RouteSAG/
├── models/
│   └── Qwen3-4B-Q4_K_M.gguf

```

### 4. Run the Assistant

```bash
python main.py

```

---

## 🛠️ Tech Stack

* **Orchestration:** [DSPy](https://github.com/stanfordnlp/dspy)
* **Inference Engine:** [llama.cpp](https://github.com/ggml-org/llama.cpp) via `llama-cpp-python`
* **Local Model:** Qwen3-4B (Q4_K_M GGUF)
* **Search Engine:** [duckduckgo-search](https://pypi.org/project/duckduckgo-search/) (Keyless, lightweight, privacy-focused search)

---

## 📋 Limitations & Roadmap

**Limitations**

* **Router accuracy is unmeasured.** A misrouted query can return a stale or wrong answer.
* **Search rate-limiting.** Rapid, repetitive automated queries may temporarily trigger DuckDuckGo rate limits.
* **Search snippet count.** Only three result snippets are returned to maintain low latency and fit local context windows.
* **Small-model parsing quirks.** Smaller local models sometimes generate extra XML/text wrappers, which are cleaned via regex post-processing.
* **Stateless.** No conversation memory.

**Roadmap:** Router evaluation script (`evaluate.py`) on a labeled dataset, conversational memory, streaming responses, and a simple web GUI.

---

## 📜 License

Distributed under the MIT License.
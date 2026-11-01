# RouteSAG 🌿🤖

**RouteSAG** (Route-based System for Agentic Generation) is a lightweight, hybrid AI assistant that combines a local GGUF model with dynamic query routing and live web search via **SearXNG**. 

Instead of relying on heavy cloud data centers, RouteSAG uses **DSPy** to route queries intelligently—using local inference for general prompts and a search-enabled ReAct agent only when up-to-date data is required.

---

## 📐 Architecture & Workflow

```text
User Query ──> [router.py] (intent: search or local)
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
      (local)               (search)
         │                     │
         │              [search.py]
         │            (top 3 snippets)
         │                     │
         └──────────┬──────────┘
                    ▼
        [model.py] Qwen3-4B generates the answer
                    ▼
              Final Answer
```

| Module | Role |
| :--- | :--- |
| `main.py` | CLI entry point and execution loop. |
| `model.py` | Initializes local GGUF inference via `llama-cpp-python` and configures DSPy. |
| `router.py` | DSPy-based intent classifier (`search` vs `local`). |
| `search.py` | Fallback-enabled SearXNG web search integration. |

---

## 🚀 Quick Start

### 1. Prerequisites
* **Python 3.10+**
* A compatible GGUF model (e.g., Qwen3-4B-Q4_K_M)

### 2. Installation
```bash
git clone https://github.com/your-username/RouteSAG.git
cd RouteSAG

python -m venv venv
# Activate virtual environment:
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install dspy llama-cpp-python requests
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
* **Search:** SearXNG (Privacy-oriented metasearch)

---

## 📋 Limitations & Roadmap

**Limitations**
* **Router accuracy is unmeasured.** A misrouted query can return a stale or wrong answer.
* **Search is not private.** On the search path, the query text is sent to public SearXNG instances. Self-hosting an instance avoids this.
* **Search quality is limited.** Only three result snippets are returned, and public instances may rate-limit or go down.
* **The agent may skip the search.** On the search path, a small model can answer from memory without calling the tool.
* **Small-model reliability.** A 4B model can fail at structured output and tool use.
* **Stateless.** No conversation memory.

**Roadmap:** router evaluation on a labeled query set, conversation history, streaming responses, a self-hosted search setup, and a simple web GUI.
---

## 📜 License
Distributed under the MIT License.
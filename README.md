# RouteSAG 🌿🤖
### Route-Based System for Agentic Generation

**RouteSAG** is a lightweight hybrid AI assistant built with **DSPy** that combines local Qwen3 inference with intelligent query routing and live **DuckDuckGo** search.

It automatically determines whether a query can be answered locally or requires up-to-date web information.

RouteSAG explores how lightweight language models can decide when to rely on internal knowledge and when to retrieve external information.

## 📐 Architecture

```text
       User Query
           |
           v
   [model_design/router.py]
           |
       +---+---+
       |       |
       v       v
     STATIC  SEARCH
       |       |
       | [model_design/search.py]
       |       |
       +---+---+
           |
           v
  [model_design/model.py]
       Local Qwen3
           |
           v
       Final Answer
```

### Project Structure

```text
RouteSAG/
├── evaluation/
│   ├── evaluate.py
│   └── FreshQA_v042126.csv
├── model_design/
│   ├── model.py
│   ├── router.py
│   └── search.py
├── models/
│   └── Qwen3-4B-Q4_K_M.gguf
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

| Module | Description |
|---|---|
| `main.py` | Application entry point and execution. |
| `model_design/model.py` | Local GGUF model initialization and DSPy configuration. |
| `model_design/router.py` | DSPy-based intent classification with regex sanitization. |
| `model_design/search.py` | DuckDuckGo web search integration. |
| `evaluation/evaluate.py` | Evaluates router classification performance using FreshQA. |
| `evaluation/FreshQA_v042126.csv` | Dataset used for router evaluation. |

## 🚀 Installation

**Requirements:**
- Python 3.10+
- [uv](https://astral.sh) package manager
- Compatible GGUF model
- Internet connection for web searches

**1. Clone the repository**

```bash
git clone https://github.com/EdwardTi11/RouteSAG.git
cd RouteSAG
```

**2. Install dependencies**

This creates a `.venv` virtual environment and installs exact dependency versions from `uv.lock`:

```bash
uv sync
```

**3. Add your model**

Place your GGUF model in `models/`:

```text
models/
└── Qwen3-4B-Q4_K_M.gguf
```

**4. Run RouteSAG**

```bash
uv run python main.py
```

## 🛠️ Tech Stack

- **[DSPy](https://github.com/stanfordnlp/dspy):** Agent orchestration and query routing.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp):** Local model inference.
- **[Qwen3](https://github.com/QwenLM/Qwen3):** Local language model.
- **[ddgs](https://pypi.org/project/ddgs/):** Keyless web search.
- **[uv](https://astral.sh):** Dependency and environment management.
- **Python:** Core implementation.

## 📊 Router Evaluation

The router was evaluated on **500 questions from the FreshQA TEST split** to assess whether queries require local inference (`static`) or external web search (`search`).

The evaluation script and dataset are located in the `evaluation/` directory.

The evaluation was conducted on Kaggle using a T4 GPU.

### Evaluation Methodology

Expected routes were assigned using FreshQA's `fact_type` and `effective_year` metadata:

- `never-changing` → `static`
- `fast-changing` → `search`
- `slow-changing` → `static` if the effective year is before 2025; otherwise `search`

The 2025 threshold serves as an operational approximation of the model's knowledge boundary.

Latency was measured using a Kaggle T4 GPU.

### Running the Evaluation

From the project root:

```bash
uv run python evaluation/evaluate.py
```

### Overall Results

| Metric | Result |
|---|---:|
| Accuracy | **61.60%** |
| Macro F1 | **0.6154** |
| Search precision | 45.86% |
| Search recall | 86.75% |
| Invalid prediction rate | **0.00%** |
| Median latency | **3.357 seconds** |

### Accuracy by Question Category

| Category | Accuracy |
|---|---:|
| Fast-changing | 90.91% (120/132) |
| Never-changing | 56.91% (107/188) |
| Slow-changing | 45.00% (81/180) |

### Confusion Matrix

| Expected Route | Predicted Static | Predicted Search | Invalid |
|---|---:|---:|---:|
| Static | 164 | 170 | 0 |
| Search | 22 | 144 | 0 |

### Limitations

- **Approximate knowledge boundary:** The 2025 threshold is an operational assumption, not a verified cutoff for the model's training knowledge.
- **Proxy labels:** FreshQA metadata provides a practical labeling approach but does not establish the correct routing decision for every individual question.
- **Slow-changing questions:** This category has the lowest accuracy, indicating that distinguishing older information from newer information remains challenging.
- **Search trade-off:** The router favors search coverage, achieving 86.75% search recall at the cost of additional unnecessary searches.
- **Benchmark coverage:** FreshQA does not represent every type of real-world user query.

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.

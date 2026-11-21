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
     LOCAL   SEARCH
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
├──
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
- Compatible GGUF model
- Internet connection for web searches

**1. Clone the repository**

```bash
git clone https://github.com/your-username/RouteSAG.git
cd RouteSAG
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Add your model**

Place your GGUF model in `models/`:

```text
models/
└── Qwen3-4B-Q4_K_M.gguf
```

**5. Run RouteSAG**

```bash
python main.py
```

## 🛠️ Tech Stack

- **[DSPy](https://github.com/stanfordnlp/dspy):** Agent orchestration and query routing.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp):** Local model inference.
- **[Qwen3](https://github.com/QwenLM/Qwen3):** Local language model.
- **[ddgs](https://pypi.org/project/ddgs/):** Keyless web search.
- **Python:** Core implementation.

## 📊 Router Evaluation

The router was evaluated on **500 questions from the FreshQA TEST split** to assess whether queries require local inference (`static`) or external web search (`search`).

The evaluation script and dataset are located in the `evaluation/` directory.

### Evaluation Methodology

Expected routes were assigned using FreshQA's `fact_type` and `effective_year` metadata:

- `never-changing` → `static`
- `fast-changing` → `search`
- `slow-changing` → `static` if the effective year is before 2025; otherwise `search`

The 2025 threshold serves as an operational approximation of the model's knowledge boundary.

For latency calculations, the numbers come from the T4 GPU, provided by Kaggle.

### Running the Evaluation

From the project root:

```bash
python evaluation/evaluate.py
```

### Overall Results

| Metric | Result |
|---|---:|
| Accuracy | **68.00%** |
| Macro F1 | **0.6832** |
| Search precision | 53.04% |
| Search recall | 78.92% |
| Invalid prediction rate | 3.20% |
| Median latency | 7.124 seconds |

### Accuracy by Question Category

| Category | Accuracy |
|---|---:|
| Fast-changing | 82.58% (109/132) |
| Never-changing | 71.81% (135/188) |
| Slow-changing | 53.33% (96/180) |

### Confusion Matrix

| Expected Route | Predicted Static | Predicted Search | Invalid |
|---|---:|---:|---:|
| Static | 209 | 116 | 9 |
| Search | 28 | 131 | 7 |

### Limitations

- **Approximate knowledge boundary:** The 2025 threshold is an operational assumption, not a verified cutoff for the model's training knowledge.
- **Proxy labels:** FreshQA metadata provides a practical labeling approach, but does not establish the correct routing decision for every individual question.
- **Slow-changing questions:** This category currently has the lowest accuracy, suggesting that distinguishing older information from newer information remains challenging.
- **Invalid outputs:** Some predictions do not match either expected route, indicating remaining output-consistency issues.
- **Benchmark coverage:** FreshQA does not represent every type of real-world user query.

## 🗺️ Roadmap

- [x] Initial router evaluation and benchmarking.
- [x] FreshQA-based router evaluation.
- [x] Temporal routing classification.
- [ ] Improve routing accuracy and output consistency.
- [ ] Conversational memory.
- [ ] Streaming responses.
- [ ] Web-based interface.
- [ ] Improved search and retrieval.
- [ ] Configurable model settings.

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.

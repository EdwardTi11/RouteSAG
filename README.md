# RouteSAG 🌿🤖
### Route-Based System for Agentic Generation

**RouteSAG** is a lightweight hybrid AI assistant built with **DSPy** that combines local Qwen3 inference with intelligent query routing and live **DuckDuckGo** search.

It automatically determines whether a query can be answered locally or requires up-to-date web information.

## 📐 Architecture

```text
       User Query
            |
            v
      [router.py]
            |
       +----+----+
       |         |
       v         v
     LOCAL     SEARCH
       |         |
       |    [search.py]
       |         |
       +----+----+
            |
            v
        [model.py]
        Local Qwen3
            |
            v
       Final Answer
```

| Module | Description |
|---|---|
| `main.py` | Application entry point and execution. |
| `model.py` | Local GGUF model initialization and DSPy configuration. |
| `router.py` | DSPy-based intent classification with regex sanitization. |
| `search.py` | DuckDuckGo web search integration. |

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
pip install dspy llama-cpp-python duckduckgo-search
```

**4. Add your model**

Place your GGUF model in `models/`:

```text
RouteSAG/
├── models/
│   └── Qwen3-4B-Q4_K_M.gguf
├── main.py
├── model.py
├── router.py
└── search.py
```

**5. Run**

```bash
python main.py
```

## 🛠️ Tech Stack

- **[DSPy](https://github.com/stanfordnlp/dspy):** Agent orchestration and query routing.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp):** Local model inference.
- **[Qwen3](https://github.com/QwenLM/Qwen3):** Local language model.
- **[duckduckgo-search](https://pypi.org/project/duckduckgo-search/):** Keyless web search.
- **Python:** Core implementation.

## 📊 Router Evaluation

The router was evaluated on **2,255 samples** from two datasets. The benchmark assigned `static` as the expected route for SimpleQA Verified and `search` for BrowseComp.

| Metric | Result |
|---|---:|
| Overall accuracy | **36.19%** |
| Macro-F1 | **0.3787** |
| Median latency | **8.6948 seconds** |
| SimpleQA Verified accuracy | 63.10% (631/1,000) |
| BrowseComp accuracy | 14.74% (185/1,255) |

### Classification results

| Expected route | Predicted static | Predicted search | Invalid |
|---|---:|---:|---:|
| Static (1,000 samples) | 631 | 353 | 16 |
| Search (1,255 samples) | 659 | 185 | 411 |

The evaluation produced **427 invalid predictions** that did not match either route label. They are counted as incorrect in the overall accuracy.

## 📋 Limitations

- **Benchmark labels are proxies:** Each dataset was assigned a single expected route (`static` for SimpleQA Verified and `search` for BrowseComp). This is a useful initial test, but it does not establish the correct route for every individual question.
- **Routing performance is limited:** The benchmark achieved 36.19% overall accuracy. BrowseComp routing was particularly weak at 14.74%, indicating that the router often failed to identify questions assigned to the search category.
- **Invalid outputs:** 427 of 2,255 predictions did not match either expected label, suggesting that output formatting or label compliance remains an issue.
- **Limited benchmark coverage:** Two datasets cannot represent the full range of real user queries or establish general real-world routing performance.
- DuckDuckGo may impose rate limits.
- Search retrieves only the top three results.
- Smaller models may produce inconsistent output formatting.
- No conversational memory or persistent storage.
- Performance depends on available hardware.

## 🗺️ Roadmap

- [x] Initial router evaluation and benchmarking.
- [ ] Improve routing accuracy and output consistency.
- [ ] Conversational memory.
- [ ] Streaming responses.
- [ ] Web-based interface.
- [ ] Improved search and retrieval.
- [ ] Configurable model settings.

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.

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

## 📋 Limitations

- Router accuracy has not been formally evaluated.
- DuckDuckGo may impose rate limits.
- Search retrieves only the top three results.
- Smaller models may produce inconsistent output formatting.
- No conversational memory or persistent storage.
- Performance depends on available hardware.

## 🗺️ Roadmap

- [ ] Router evaluation and benchmarking.
- [ ] Conversational memory.
- [ ] Streaming responses.
- [ ] Web-based interface.
- [ ] Improved search and retrieval.
- [ ] Configurable model settings.

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.
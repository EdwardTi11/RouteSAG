# RouteSAG 🌿🤖

**RouteSAG** (Route-based System for Agentic Generation) is an edge-first, privacy-focused AI framework designed to deliver specialized agentic intelligence without relying on massive, energy-intensive cloud data centers or Mixture of Experts (MoE) architectures.

By pairing a lightweight local model with declarative **DSPy** signatures, dynamic query routing, and real-time open search via **SearXNG**, RouteSAG achieves high-quality specialized reasoning at zero API cost and minimal resource footprint.

## 🌟 Key Features & Philosophy

* **Data Center Alternative:** Built to prove that effective AI specialization can be achieved through clever software architecture (routing and prompt engineering) rather than brute-force hardware scaling.

* **100% Local & Private:** Powered locally by `llama-cpp-python` using quantized GGUF models (\~3B to 4B parameters), consuming \~2.0 GB RAM/VRAM.

* **Declarative Prompt Orchestration:** Leverages **DSPy** for structured input/output contracts and ReAct agentic reasoning loops.

* **Dynamic Query Routing:** Acts as a lightweight software-level "router" that categorizes tasks (coding, math, local reasoning, live web search) to optimize execution and avoid wasted compute.

* **No API Keys Required:** Uses SearXNG for live web search snippet retrieval without tracking, database caching, or third-party service fees.

## 📐 System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        1. USER INPUT / QUERY                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        2. DSPY ROUTER MODULE                           │
│  - Inspects intent: Coding, Math, Direct Reasoning, or Web Search      │
└───────────┬────────────────────────────────────────────────┬───────────┘
            │                                                │
   [Direct / Local Intent]                          [Needs Live Web Context]
            │                                                │
            ▼                                                ▼
┌─────────────────────────┐               ┌──────────────────────────────┐
│  LOCAL MODEL REASONING  │               │   3. SEARXNG SEARCH AGENT    │
│  - Optimized prompt     │               │ - Queries open JSON API      │
│  - Low latency execution│               │ - Retrieves top 3 snippets   │
└───────────┬─────────────┘               └──────────────┬───────────────┘
            │                                            │
            └───────────────────────┬────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   4. LOCAL GGUF INFERENCE ENGINE                       │
│  - 3B/4B quantized local GGUF model (Qwen / Llama)                     │
│  - Handled via llama-cpp-python (~2.0 GB Memory Footprint)             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   5. RESPONSE & EVALUATION LOOP                        │
│  - Formatted final answer returned directly                            │
└────────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology | Footprint / Resource Usage | Role | 
| ----- | ----- | ----- | ----- | 
| **Logic & Control** | [DSPy](https://github.com/stanfordnlp/dspy) | Minimal Python Process | Orchestrates routing, agent ReAct loops, and signatures | 
| **Inference Engine** | `llama-cpp-python` | \~2.0 GB VRAM / System RAM | Executes local GGUF model inference | 
| **Local Model** | Qwen 2.5 3B / Qwen 3 4B (Q4_K_M GGUF) | \~2.5 - 3.0 GB Storage | Lightweight local reasoning engine | 
| **Search Engine** | [SearXNG](https://docs.searxng.org/) | Lightweight HTTP requests | Live, privacy-preserving web search integration | 

## 📁 Repository Structure

```
route_sag/
├── main.py            # CLI interface and main execution loop
├── model.py           # Local GGUF initialization and DSPy LM wrapper
├── router.py          # DSPy classification & dynamic prompt routing
├── tools.py           # SearXNG search tool integration & fallback handling
├── agent.py           # ReAct module and agent signature definitions
├── models/            # Folder containing local .gguf model files
└── README.md          # Project documentation
```

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have Python 3.10+ installed and a C/C++ compiler set up for `llama-cpp-python`.

```bash
git clone https://github.com/your-username/route-sag.git
cd route-sag
```

### 2. Install Dependencies

```bash
pip install dspy-ai llama-cpp-python requests
```

### 3. Download Model Weights

Place your target GGUF file inside the `./models/` directory (e.g., `Qwen3-4B-Q4_K_M.gguf` or `Qwen2.5-3B-Instruct-Q4_K_M.gguf`).

### 4. Run RouteSAG

```bash
python main.py
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
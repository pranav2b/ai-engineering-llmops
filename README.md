# 🧠 AI Engineering — LLMOps & AgentOps Demos

> Practical, runnable Python demos for teaching LLMOps and AgentOps concepts.
> Used in classroom sessions and the companion YouTube playlist.

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| OpenAI GPT-4o | LLM backend for all demos |
| MLflow | Experiment tracking, prompt versioning, model registry |
| LangSmith | Agent tracing & observability *(upcoming)* |
| LangChain | Agent orchestration *(upcoming)* |
| Python 3.10+ | All scripts |

---

## 📂 Demo Index

| # | Folder | Concept | Tools |
|---|--------|---------|-------|
| 01 | `01_prompt_versioning/` | Prompt Versioning | OpenAI, MLflow |
| 02 | `02_evaluations/` | LLM Evaluations & Evals | *(coming soon)* |
| 03 | `03_observability/` | Observability & Tracing | *(coming soon)* |
| 04 | `04_rag_pipeline/` | RAG Pipeline | *(coming soon)* |
| 05 | `05_guardrails/` | Guardrails & Safety | *(coming soon)* |
| 06 | `06_tool_use/` | Tool Use / Function Calling | *(coming soon)* |
| 07 | `07_react_agent/` | ReAct Agent Loop | *(coming soon)* |
| 08 | `08_multi_agent/` | Multi-Agent Systems | *(coming soon)* |
| 09 | `09_memory/` | Agent Memory Types | *(coming soon)* |
| 10 | `10_human_in_loop/` | Human-in-the-Loop | *(coming soon)* |

---

## ⚙️ Setup

```bash
# Clone
git clone https://github.com/pranav2b/ai-engineering-llmops.git
cd ai-engineering-llmops

# Install base dependencies
pip install openai mlflow python-dotenv

# Create .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

Each demo folder has its own `requirements.txt` and `README.md`.

---

## ▶️ Running a Demo

```bash
cd 01_prompt_versioning
python prompt_versioning_demo.py

# View MLflow UI
mlflow ui   # → http://127.0.0.1:5000
```

---

## 📺 YouTube Playlist

> Link will be added once the playlist is live.

---

## 📄 License

MIT

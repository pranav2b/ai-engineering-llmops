# LLMOps Demo #1 — Prompt Versioning with MLflow

## What this demo teaches
Prompts are code. They evolve, break, and need rollbacks — just like software.
This demo shows how to version prompts like a production engineering team would.

## Concepts covered
| Concept | How it's shown |
|---|---|
| Prompt versioning | 3 explicit versions (v1→v2→v3) of a customer support prompt |
| Structured params | model, temperature, max_tokens tracked per version |
| Commit messages | Each version has a reason for the change |
| Quality metrics | empathy score, actionability score, format compliance |
| Operational metrics | latency, token usage, estimated cost |
| MLflow tracking | Every run logged with params + metrics + artifact |
| Version comparison | Side-by-side table with composite score |
| Deploy + Rollback | Simulate promoting and reverting a production prompt |

## Setup
```bash
pip install openai mlflow python-dotenv
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

## Run the demo
```bash
python prompt_versioning_demo.py
```

## View results in MLflow UI
```bash
mlflow ui
# Open http://127.0.0.1:5000
# Go to experiment: llmops-prompt-versioning
```

## What to show in class / on YouTube

### Step 1 — Show the 3 prompt versions side by side
- v1: one-liner, vague
- v2: structured goals, better tone rules
- v3: strict output format with SLA language

**Key point:** "Same task, same model — but the prompt completely changes the output quality."

### Step 2 — Run the script
Watch it hit the API for each version × each test case.
Point out the live metrics printing.

### Step 3 — Open MLflow UI
- Show the Experiment view with all runs grouped
- Click into a run: show params, metrics, and the JSON artifact
- Use "Compare runs" to show a bar chart of empathy_score across v1/v2/v3
- Show how v3 format_score jumps to 1.0

### Step 4 — Composite scoring
Walk through the weighted formula:
```
score = 0.35×empathy + 0.30×action + 0.20×format + 0.15×(speed bonus)
```
"This is how teams decide which version to promote — not gut feel."

### Step 5 — Deploy & Rollback
Show the registry JSON update.
Simulate a rollback and ask: "What happens in production if v3 starts hallucinating?"

## File structure
```
prompt_versioning/
├── prompt_versioning_demo.py   # Main demo script
├── README.md                   # This file
└── mlruns/                     # Auto-created by MLflow
    └── ...
```

## Next demo in the playlist
**Demo #2 — LLM Observability with MLflow Tracing**
Instrument a RAG pipeline and watch every call traced in real time.

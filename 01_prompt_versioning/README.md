# Demo 01 — Prompt Versioning with MLflow

## Overview

Prompts are not static configuration. They evolve, regress, and need to be
audited just like application code. This demo shows how to treat prompts as
versioned artifacts: track every change, measure the impact on output quality,
and promote or roll back versions with confidence.

---

## Concepts Covered

| Concept | How it is demonstrated |
|---------|----------------------|
| Prompt versioning | Three explicit versions of a customer support prompt, v1 through v3 |
| Structured parameters | Model, temperature, and max_tokens tracked per version |
| Commit messages | Each version documents the reason for the change |
| Quality metrics | Empathy score, actionability score, format compliance |
| Operational metrics | Latency, token usage, estimated cost per run |
| MLflow tracking | Every run logged with full params, metrics, and artifacts |
| Version comparison | Side-by-side table with a weighted composite score |
| Deploy and rollback | Simulates promoting a version to production and reverting it |

---

## Setup

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## Running the Demo

```bash
python prompt_versioning_demo.py
```

Then open the MLflow UI to inspect results:

```bash
mlflow ui
# Navigate to http://127.0.0.1:5000
# Select experiment: llmops-prompt-versioning
```

---

## What to Show in Class

### Step 1 — Walk through the three prompt versions

Show v1 (one-liner, vague), v2 (structured goals, tone rules), and v3
(strict output format with SLA language) side by side in the script.

Key point: same task, same model, same test input — the prompt is the
only variable, and it completely changes output quality.

### Step 2 — Run the script

Watch each version run against three fixed test cases. Point out the
metrics printing live: latency, tokens, cost, empathy score, action score.

### Step 3 — Open the MLflow UI

- Show the Experiments view with all runs grouped by version tag.
- Click into a single run and show params, metrics, and the JSON artifact.
- Use Compare Runs to display a bar chart of empathy_score across v1, v2, v3.
- Show how format_score jumps to 1.0 only for v3.

### Step 4 — Walk through the composite scoring formula

```
score = 0.35 x empathy + 0.30 x action + 0.20 x format + 0.15 x speed_bonus
```

This is how teams decide which version to promote rather than relying on
intuition. The weights are adjustable based on business priorities.

### Step 5 — Deploy and rollback simulation

Show the registry JSON being updated on deploy and reverted on rollback.
Discussion point: what happens in production if v3 starts producing
malformed output at scale?

---

## File Structure

```
01_prompt_versioning/
    prompt_versioning_demo.py    Main demo script
    requirements.txt             Python dependencies
    README.md                    This file
    mlruns/                      Auto-created by MLflow on first run
```

---

## Next Demo

Demo 02 — LLM Evaluations

Build an automated evaluation harness with LLM-as-judge scoring.
Measure factuality, tone, and format compliance across model versions.

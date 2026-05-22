"""
Demo 01 - Prompt Versioning with MLflow
Prompts are not static configuration. They evolve, regress, and need to be
audited just like application code. This script demonstrates how to:

    1. Store prompt versions as structured artifacts in MLflow
    2. Run each version against the same fixed test inputs
    3. Track quality metrics: empathy, actionability, format compliance
    4. Track operational metrics: latency, token usage, estimated cost
    5. Compare versions with a weighted composite score
    6. Simulate promoting a version to production and rolling it back

Setup:
    pip install -r requirements.txt
    Add OPENAI_API_KEY to a .env file in this directory.

Run:
    python prompt_versioning_demo.py

View results:
    mlflow ui
    Open http://127.0.0.1:5000 and select experiment: llmops-prompt-versioning
"""

import os
import json
import time
import mlflow
import mlflow.artifacts
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI client ─────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── MLflow experiment ─────────────────────────────────────
EXPERIMENT_NAME = "llmops-prompt-versioning"
mlflow.set_tracking_uri("mlruns")          # local folder; swap for remote URI in prod
mlflow.set_experiment(EXPERIMENT_NAME)


# ---
# SECTION 1 — Define prompt versions
# ---
# Each version is a dict that captures everything needed to
# reproduce a run: system prompt, user template, and model params.
# Think of this as your "prompt registry".

PROMPT_VERSIONS = {
    "v1": {
        "version":         "v1",
        "description":     "Basic — vague instructions, no structure",
        "system_prompt":   "You are a customer support agent. Help users with their problems.",
        "user_template":   "Customer: {message}",
        "model":           "gpt-4o",
        "temperature":     0.8,
        "max_tokens":      200,
        "commit_message":  "Initial prompt. Baseline.",
    },
    "v2": {
        "version":         "v2",
        "description":     "Improved tone + structured goals",
        "system_prompt":   (
            "You are a friendly, empathetic customer support specialist for an e-commerce platform.\n"
            "Your goals:\n"
            "1. Acknowledge the customer's frustration with genuine empathy.\n"
            "2. Clearly state what you can do to help.\n"
            "3. Offer concrete next steps or a solution.\n"
            "4. End with a reassurance.\n\n"
            "Tone: warm, professional, solution-focused. "
            "Never say 'I cannot' — say what you CAN do instead."
        ),
        "user_template":   "Customer message: {message}\n\nRespond helpfully:",
        "model":           "gpt-4o",
        "temperature":     0.5,
        "max_tokens":      350,
        "commit_message":  "Added empathy structure + tone rules. Reduced temp for consistency.",
    },
    "v3": {
        "version":         "v3",
        "description":     "Structured output with SLA language",
        "system_prompt":   (
            "You are a senior customer success specialist for ShopEase.\n\n"
            "Always structure your reply in EXACTLY this format:\n"
            "[EMPATHY]   — One sentence acknowledging the customer's emotion.\n"
            "[ACTION]    — What you will do immediately.\n"
            "[TIMELINE]  — Specific timeframe (e.g., '24–48 hours', not 'soon').\n"
            "[OFFER]     — Any compensation or goodwill gesture.\n\n"
            "Rules:\n"
            "- Use specific timeframes, never vague words like 'soon' or 'shortly'.\n"
            "- Offer an escalation path if needed.\n"
            "- Sign off as: 'Alex — ShopEase Customer Success'"
        ),
        "user_template":   "Customer: {message}",
        "model":           "gpt-4o",
        "temperature":     0.3,
        "max_tokens":      400,
        "commit_message":  "Structured [EMPATHY/ACTION/TIMELINE/OFFER] format. Lower temp for format compliance.",
    },
}


# ---
# SECTION 2 — Test inputs (fixed test set for fair comparison)
# ---
TEST_INPUTS = [
    "My order hasn't arrived in 2 weeks and I want a refund immediately!",
    "I was charged twice for the same item. This is unacceptable.",
    "The product I received is completely different from what I ordered.",
]


# ---
# SECTION 3 — Simple scoring heuristics
# (In a real pipeline, use an LLM-as-judge or human eval)
# ---

def score_empathy(response: str) -> float:
    """Check for empathetic language keywords."""
    empathy_words = [
        "sorry", "understand", "frustrat", "apologize",
        "concern", "feel", "difficult", "inconvenien"
    ]
    hits = sum(1 for w in empathy_words if w in response.lower())
    return min(hits / 3.0, 1.0)  # normalise to [0, 1]


def score_actionability(response: str) -> float:
    """Check for concrete action words."""
    action_words = [
        "will", "refund", "replace", "contact", "investigate",
        "escalate", "resolve", "fix", "send", "process"
    ]
    hits = sum(1 for w in action_words if w in response.lower())
    return min(hits / 3.0, 1.0)


def score_format_compliance(response: str, version: str) -> float:
    """v3 requires specific section headers."""
    if version != "v3":
        return 1.0  # not applicable
    required = ["[EMPATHY]", "[ACTION]", "[TIMELINE]", "[OFFER]"]
    found = sum(1 for tag in required if tag in response)
    return found / len(required)


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """GPT-4o pricing as of mid-2025: $5/1M input, $15/1M output."""
    return (prompt_tokens * 5 + completion_tokens * 15) / 1_000_000


# ---
# SECTION 4 — Run a single prompt version against one input
# ---

def run_prompt(prompt_cfg: dict, user_message: str) -> dict:
    """
    Calls the OpenAI API with the given prompt version config,
    measures latency, tokens, and cost, returns a result dict.
    """
    system = prompt_cfg["system_prompt"]
    user   = prompt_cfg["user_template"].format(message=user_message)

    start = time.perf_counter()
    response = client.chat.completions.create(
        model       = prompt_cfg["model"],
        temperature = prompt_cfg["temperature"],
        max_tokens  = prompt_cfg["max_tokens"],
        messages    = [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ],
    )
    latency = time.perf_counter() - start

    output         = response.choices[0].message.content.strip()
    prompt_tokens  = response.usage.prompt_tokens
    compl_tokens   = response.usage.completion_tokens
    cost           = estimate_cost_usd(prompt_tokens, compl_tokens)

    return {
        "output":           output,
        "latency_s":        round(latency, 3),
        "prompt_tokens":    prompt_tokens,
        "completion_tokens": compl_tokens,
        "total_tokens":     prompt_tokens + compl_tokens,
        "cost_usd":         round(cost, 6),
        "empathy_score":    round(score_empathy(output), 3),
        "action_score":     round(score_actionability(output), 3),
        "format_score":     round(score_format_compliance(output, prompt_cfg["version"]), 3),
    }


# ---
# SECTION 5 — Log everything to MLflow
# ---

def run_and_log_version(prompt_cfg: dict):
    """
    Runs a prompt version against all test inputs and logs
    each run as a separate MLflow run with full params + metrics.
    """
    version = prompt_cfg["version"]
    print(f"\n{'='*60}")
    print(f"  Running prompt {version}: {prompt_cfg['description']}")
    print("-" * 60)

    all_results = []

    for i, test_input in enumerate(TEST_INPUTS, 1):
        print(f"  [{i}/{len(TEST_INPUTS)}] Input: {test_input[:60]}...")

        result = run_prompt(prompt_cfg, test_input)
        all_results.append(result)

        # ── Log each test case as its own MLflow run ──────────
        with mlflow.start_run(run_name=f"{version}_test{i}"):

            # Tags (searchable metadata)
            mlflow.set_tags({
                "prompt_version":    version,
                "test_case":         str(i),
                "commit_message":    prompt_cfg["commit_message"],
                "description":       prompt_cfg["description"],
            })

            # Parameters (what we controlled)
            mlflow.log_params({
                "model":             prompt_cfg["model"],
                "temperature":       prompt_cfg["temperature"],
                "max_tokens":        prompt_cfg["max_tokens"],
                "prompt_version":    version,
            })

            # Metrics (what we measured)
            mlflow.log_metrics({
                "latency_s":         result["latency_s"],
                "prompt_tokens":     result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens":      result["total_tokens"],
                "cost_usd":          result["cost_usd"],
                "empathy_score":     result["empathy_score"],
                "action_score":      result["action_score"],
                "format_score":      result["format_score"],
            })

            # Artifacts (the actual prompt text + output)
            artifact_data = {
                "prompt_config": {
                    "system_prompt": prompt_cfg["system_prompt"],
                    "user_template": prompt_cfg["user_template"],
                },
                "test_input":    test_input,
                "model_output":  result["output"],
            }
            artifact_path = f"/tmp/prompt_{version}_test{i}.json"
            with open(artifact_path, "w") as f:
                json.dump(artifact_data, f, indent=2)
            mlflow.log_artifact(artifact_path, artifact_path="prompt_artifacts")

        print(f"     latency={result['latency_s']}s  "
              f"tokens={result['total_tokens']}  "
              f"cost=${result['cost_usd']}  "
              f"empathy={result['empathy_score']}  "
              f"action={result['action_score']}  "
              f"format={result['format_score']}")
        print(f"     Output preview: {result['output'][:120]}...\n")

    # ── Aggregate metrics across test cases ───────────────────
    avg = lambda key: round(sum(r[key] for r in all_results) / len(all_results), 4)

    print(f"\n  ── Aggregate for {version} ──")
    print(f"     avg latency:       {avg('latency_s')}s")
    print(f"     avg total_tokens:  {avg('total_tokens')}")
    print(f"     avg cost_usd:      ${avg('cost_usd')}")
    print(f"     avg empathy:       {avg('empathy_score')}")
    print(f"     avg action:        {avg('action_score')}")
    print(f"     avg format:        {avg('format_score')}")

    return all_results


# ---
# SECTION 6 — Compare versions and pick the winner
# ---

def compare_versions(summary: dict):
    """
    Print a side-by-side comparison table of all versions,
    then declare the winner based on a weighted composite score.
    """
    print(f"\n{'='*70}")
    print("  VERSION COMPARISON SUMMARY")
    print("-" * 70)
    header = f"{'Version':<8} {'Latency':>9} {'Tokens':>8} {'Cost$':>9} {'Empathy':>9} {'Action':>9} {'Format':>8} {'SCORE':>8}"
    print(header)
    print("-" * 70)

    scores = {}
    for version, metrics in summary.items():
        # Weighted composite: quality matters more than speed for support bot
        composite = (
            0.35 * metrics["empathy_score"] +
            0.30 * metrics["action_score"] +
            0.20 * metrics["format_score"] +
            0.15 * (1 - min(metrics["latency_s"] / 5.0, 1.0))  # penalise slow
        )
        scores[version] = composite
        print(
            f"{version:<8} "
            f"{metrics['latency_s']:>8.2f}s "
            f"{int(metrics['total_tokens']):>8} "
            f"${metrics['cost_usd']:>8.5f} "
            f"{metrics['empathy_score']:>9.3f} "
            f"{metrics['action_score']:>9.3f} "
            f"{metrics['format_score']:>8.3f} "
            f"{composite:>8.3f}"
        )

    winner = max(scores, key=scores.get)
    print(f"\n  Winner: {winner}  (composite score: {scores[winner]:.3f})")
    print(f"      → This is the version you would DEPLOY to production.\n")
    return winner


# ---
# SECTION 7 — Demo: Deploy & Rollback simulation
# ---

def demo_deploy_rollback(winner: str):
    """
    Simulates tagging a version as 'production' in MLflow,
    and rolling back to the previous version.
    """
    print("-" * 60)
    print("  DEPLOY & ROLLBACK DEMO")
    print("-" * 60)

    # In a real system you'd use mlflow.register_model() +
    # MlflowClient().transition_model_version_stage().
    # Here we simulate with a simple JSON file to keep it clear.

    registry_path = "/tmp/prompt_registry.json"

    # Load or create registry
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"production": None, "history": []}

    prev = registry["production"]

    # Deploy winner
    registry["history"].append(prev)
    registry["production"] = winner
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"  Deployed  : {winner}  → production")
    print(f"  Previous  : {prev or 'none'}")

    # Simulate a rollback
    print(f"\n  Simulating production issue — rolling back...")
    registry["production"] = prev or list(PROMPT_VERSIONS.keys())[0]
    registry["history"].append(winner)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"  Rolled back to: {registry['production']}")
    print(f"  Registry state: {json.dumps(registry, indent=4)}")


# ---
# MAIN
# ---

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LLMOps Demo: Prompt Versioning with MLflow")
    print("=" * 60)
    print(f"  Experiment : {EXPERIMENT_NAME}")
    print(f"  Versions   : {list(PROMPT_VERSIONS.keys())}")
    print(f"  Test cases : {len(TEST_INPUTS)}")
    print(f"  Total runs : {len(PROMPT_VERSIONS) * len(TEST_INPUTS)}")
    print("=" * 60)

    # Run all versions
    summary = {}
    for version_key, prompt_cfg in PROMPT_VERSIONS.items():
        results = run_and_log_version(prompt_cfg)
        # Aggregate for comparison
        avg = lambda key: round(sum(r[key] for r in results) / len(results), 4)
        summary[version_key] = {
            "latency_s":     avg("latency_s"),
            "total_tokens":  avg("total_tokens"),
            "cost_usd":      avg("cost_usd"),
            "empathy_score": avg("empathy_score"),
            "action_score":  avg("action_score"),
            "format_score":  avg("format_score"),
        }

    # Compare and pick winner
    winner = compare_versions(summary)

    # Demo deploy/rollback
    demo_deploy_rollback(winner)

    print("\n" + "=" * 60)
    print("  All runs logged to MLflow.")
    print("  Run:  mlflow ui")
    print("  Then open: http://127.0.0.1:5000")
    print("  Filter by experiment: llmops-prompt-versioning")
    print("  Compare runs, inspect artifacts, view metric charts.")
    print("=" * 60 + "\n")

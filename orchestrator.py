import json, re, subprocess
from openai import OpenAI
from pathlib import Path

try:
    from kaggle_secrets import UserSecretsClient
    api_key = UserSecretsClient().get_secret("NVIDIA_API_KEY")
except ImportError:
    import os
    api_key = os.environ["NVIDIA_API_KEY"]

# --- Config ---
PATIENCE   = 4        # rounds without meaningful improvement before stopping
REL_DELTA  = 0.001    # 0.1% relative improvement threshold
SLOPE_STOP = -0.0005  # linear trend slope: if flatter than this, count as stale
MAX_ROUNDS = 20       # hard cap for free-tier budget
LOG_FILE   = Path("results.json")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


# ---------------------------------------------------------------------------
# Plateau Detection (three independent signals — any two triggers a stop)
# ---------------------------------------------------------------------------

def _relative_improvement(log: list) -> bool:
    if len(log) < PATIENCE:
        return False
    best_before = min(r["loss"] for r in log[:-PATIENCE])
    best_recent = min(r["loss"] for r in log[-PATIENCE:])
    return (best_before - best_recent) / best_before < REL_DELTA


def _linear_trend(log: list) -> bool:
    window = 6
    if len(log) < window:
        return False
    losses = [r["loss"] for r in log[-window:]]
    n = len(losses)
    xs = list(range(n))
    x_mean, y_mean = sum(xs) / n, sum(losses) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, losses))
    den = sum((x - x_mean) ** 2 for x in xs)
    slope = num / den if den else 0
    return slope > SLOPE_STOP


def _consecutive_failures(log: list) -> bool:
    if len(log) < PATIENCE:
        return False
    all_time_best = min(r["loss"] for r in log)
    return all(r["loss"] >= all_time_best for r in log[-PATIENCE:])


def plateau_reached(log: list) -> tuple[bool, str]:
    valid = [r for r in log if "loss" in r]
    signals = {
        "relative_improvement": _relative_improvement(valid),
        "linear_trend_flat":    _linear_trend(valid),
        "consecutive_failures": _consecutive_failures(valid),
    }
    triggered = [k for k, v in signals.items() if v]
    if len(triggered) >= 2:
        return True, f"Plateau: {', '.join(triggered)}"
    return False, ""


# ---------------------------------------------------------------------------
# Core Loop Helpers
# ---------------------------------------------------------------------------

def load_log() -> list:
    if LOG_FILE.exists():
        return json.loads(LOG_FILE.read_text())
    return []


def save_log(log: list):
    LOG_FILE.write_text(json.dumps(log, indent=2))


def propose_improvement(code: str, log: list) -> str:
    history = json.dumps(log[-6:], indent=2)
    prompt = f"""You are an ML researcher optimizing a small MLP language model.
Here is the current training code:
```python
{code}
```
Here are the last experiment results (loss lower = better):
{history}

Suggest ONE specific, minimal code change to improve final loss or generation quality.
Think about your hypothesis first, then return ONLY the modified Python code."""
    resp = client.chat.completions.create(
        model="z-ai/glm-5.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=4096, stream=False
    )
    return resp.choices[0].message.content


def extract_code(response: str) -> str:
    match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    return match.group(1) if match else response


def run_on_kaggle(code: str) -> dict:
    Path("current_experiment.py").write_text(code)
    subprocess.run(["kaggle", "kernels", "push", "-p", "."], check=True)
    subprocess.run(["kaggle", "kernels", "output", "YOUR_KERNEL_SLUG"], check=True)
    output = Path("output.txt").read_text()
    last_line = [l for l in output.splitlines() if "Loss:" in l][-1]
    loss = float(re.search(r"Loss: ([\d.]+)", last_line).group(1))
    acc  = float(re.search(r"Acc: ([\d.]+)%", last_line).group(1)) / 100
    return {"loss": loss, "acc": acc, "output": output[-500:]}


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def main():
    log  = load_log()
    code = Path("mlp_lm.py").read_text()
    round_num = len(log)
    print(f"Resuming from round {round_num}")

    while round_num < MAX_ROUNDS:
        round_num += 1
        print(f"\n=== Round {round_num} / {MAX_ROUNDS} ===")

        stopped, reason = plateau_reached(log)
        if stopped:
            print(reason)
            break

        new_code = extract_code(propose_improvement(code, log))

        try:
            metrics = run_on_kaggle(new_code)
        except Exception as e:
            print(f"Experiment failed: {e}")
            log.append({"round": round_num, "error": str(e)})
            save_log(log)
            continue

        valid_losses = [r["loss"] for r in log if "loss" in r]
        best_so_far  = min(valid_losses) if valid_losses else float("inf")
        improved     = metrics["loss"] < best_so_far

        entry = {
            "round":        round_num,
            "loss":         metrics["loss"],
            "acc":          metrics["acc"],
            "improved":     improved,
            "code_snippet": new_code[:300],
        }
        log.append(entry)
        save_log(log)

        print(f"Loss: {metrics['loss']:.4f} | Acc: {metrics['acc']:.1%} | {'✓ improved' if improved else '✗ no gain'}")

        if improved:
            code = new_code
            Path("mlp_lm.py").write_text(code)
    else:
        print(f"Reached MAX_ROUNDS ({MAX_ROUNDS}). Stopping.")


if __name__ == "__main__":
    main()

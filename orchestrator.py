import json, os, re, subprocess, sys, time
from pathlib import Path
from openai import OpenAI

print(f"[init] Working directory: {Path.cwd()}")
print(f"[init] Python: {sys.executable}")

try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("NVIDIA_API_KEY")  # Read Kaggle secret
    print("[init] API key loaded from Kaggle secrets")
except ImportError:
    API_KEY = os.environ["NVIDIA_API_KEY"]                       # Read local environment variable
    print("[init] API key loaded from environment variable")

MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 20))               # Set MAX_ROUNDS env var to override (e.g. 1 for debug)
PATIENCE = 4                                                     # Stop after this many failed experiments
LOG_FILE = Path("results.json")                                  # Saved experiment history

BASELINE = {"round": 0, "idea": "baseline", "loss": 2.6551, "acc": 0.2598, "improved": False}  # mlp_lm_base.py result

print(f"[init] MAX_ROUNDS={MAX_ROUNDS} | PATIENCE={PATIENCE} | LOG_FILE={LOG_FILE}")
print(f"[init] Baseline: loss={BASELINE['loss']} acc={BASELINE['acc']:.1%}")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",              # NVIDIA API endpoint
    api_key=API_KEY,                                             # NVIDIA API key
    timeout=120,                                                 # 120s hard timeout on API calls
)

# --- Helpers ---
def load_log():
    if LOG_FILE.exists():
        log = json.loads(LOG_FILE.read_text())
        successes = sum(1 for r in log if "loss" in r)
        failures = sum(1 for r in log if "error" in r)
        print(f"[log] Resumed {len(log)} previous experiments ({successes} succeeded, {failures} failed) from {LOG_FILE}")
        return log
    log = [BASELINE]                                             # Seed log with hardcoded baseline as round 0
    save_log(log)
    print(f"[log] No previous results found — seeded log with baseline (loss={BASELINE['loss']})")
    return log

def save_log(log):
    LOG_FILE.write_text(json.dumps(log, indent=2))               # Save results after each experiment

def plateau(log):
    losses = [run["loss"] for run in log if "loss" in run and run.get("round", 1) > 0]  # Exclude baseline
    return len(losses) > PATIENCE and min(losses[-PATIENCE:]) >= min(losses[:-PATIENCE])  # No new best loss

def ask_model(code, log):
    print("[llm] Sending streaming request to GLM-5.2...", flush=True)
    prompt = f"""Improve this small MLP language model with ONE small change.

Current code:
```python
{code}
```

Recent results:
{json.dumps(log[-5:], indent=2)}

Return exactly this format:

IDEA: one short sentence describing the single change
```python
complete replacement code
```

Rules:
- Keep TRAIN_SECONDS = 60
- Keep torch.manual_seed(42)
- Keep the FINAL | Loss: ... | Acc: ... output line
- Change one idea only
"""
    t0 = time.time()
    chunks = []
    tokens = 0
    stream = client.chat.completions.create(
        model="z-ai/glm-5.2",                                    # NVIDIA-hosted GLM-5.2
        messages=[{"role": "user", "content": prompt}],          # Send code + past results
        temperature=0.5,                                         # Prefer focused suggestions
        max_tokens=8192,                                         # Enough room for full script
        stream=True,                                             # Stream tokens as they arrive
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        chunks.append(delta)
        tokens += 1
        if tokens % 50 == 0:                                     # Print progress every 50 tokens
            print(f"[llm] ...{tokens} tokens ({time.time()-t0:.0f}s)", flush=True)

    text = "".join(chunks)
    print(f"[llm] Response complete: {tokens} tokens in {time.time()-t0:.1f}s ({len(text)} chars)", flush=True)

    idea_match = re.search(r"IDEA:\s*(.+)", text)
    code_match = re.search(r"```python\s*(.*?)```", text, re.S)
    if not idea_match:
        raise ValueError(f"[llm] No IDEA: line found in response. Raw response:\n{text[:500]}")
    if not code_match:
        raise ValueError(f"[llm] No python code block found in response. Raw response:\n{text[:500]}")

    idea = idea_match.group(1).strip()
    code = code_match.group(1).strip()
    print(f"[llm] Idea: {idea}")
    print(f"[llm] Extracted code ({len(code)} chars)")
    return idea, code

def run(code):
    train_seconds = int(re.search(r"TRAIN_SECONDS\s*=\s*(\d+)", code).group(1))  # Read candidate time limit
    if train_seconds != 60:
        raise ValueError("TRAIN_SECONDS must remain 60.")        # Reject unfair time budget

    Path("current_experiment.py").write_text(code)               # Save proposed experiment
    print(f"[run] Saved current_experiment.py ({len(code)} chars)", flush=True)
    print(f"[run] Launching subprocess: {sys.executable} current_experiment.py", flush=True)

    result = subprocess.run(
        [sys.executable, "current_experiment.py"],                # Run in current Kaggle session
        capture_output=True,
        text=True,
        timeout=90,                                               # 60s training + startup margin
    )

    print(f"[run] Subprocess finished (returncode={result.returncode})", flush=True)
    print(result.stdout)
    if result.stderr:
        print(f"[run] stderr:\n{result.stderr[-500:]}")
    if result.returncode:
        raise RuntimeError(result.stderr[-500:])                  # Keep useful error ending

    final = [line for line in result.stdout.splitlines() if line.startswith("FINAL")][-1]  # Find metrics
    loss = float(re.search(r"Loss: ([\d.]+)", final).group(1))    # Parse final loss
    acc = float(re.search(r"Acc: ([\d.]+)%", final).group(1)) / 100  # Parse final accuracy
    print(f"[run] Parsed: loss={loss:.4f} acc={acc:.1%}")
    return loss, acc

# --- Research loop ---
def main():
    log = load_log()                                              # Resume or seed with baseline

    mlp_path = Path("mlp_lm.py")
    print(f"[main] Reading base model from {mlp_path.resolve()}")
    if not mlp_path.exists():
        raise FileNotFoundError(f"[main] {mlp_path} not found — are you in the AutoResearch directory?")
    best_code = mlp_path.read_text()                              # Start from current best code
    print(f"[main] Base model loaded ({len(best_code)} chars)")

    rounds_done = max((r["round"] for r in log), default=0)      # Highest round number attempted (success or fail)
    print(f"[main] Rounds already attempted: {rounds_done} | Remaining: {MAX_ROUNDS - rounds_done}")

    for round_num in range(rounds_done + 1, MAX_ROUNDS + 1):
        print(f"\n{'='*50}")
        print(f"[main] Starting Round {round_num}/{MAX_ROUNDS}", flush=True)

        if plateau(log):
            print(f"[main] Plateau reached — no improvement in last {PATIENCE} rounds.")
            break

        try:
            idea, candidate_code = ask_model(best_code, log)     # Ask GLM for hypothesis + code
            print(f"\nRound {round_num}: {idea}")
            loss, acc = run(candidate_code)                      # Run proposed experiment

            old_losses = [r["loss"] for r in log if "loss" in r] # All losses including baseline
            improved = loss < min(old_losses, default=float("inf"))  # Must beat baseline

            log.append({
                "round": round_num,
                "idea": idea,
                "loss": loss,
                "acc": acc,
                "improved": improved,
            })
            save_log(log)                                        # Persist result before next round

            print(f"Loss: {loss:.4f} | Acc: {acc:.1%}")

            if improved:
                best_code = candidate_code                       # Use winner as next baseline
                Path("mlp_lm.py").write_text(best_code)          # Save latest best code
                print("Accepted.")
            else:
                print("Rejected.")

        except Exception as error:
            print(f"Round {round_num} failed: {error}")
            log.append({"round": round_num, "error": str(error)})
            save_log(log)

if __name__ == "__main__":
    main()

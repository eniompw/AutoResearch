import json, os, re, subprocess, sys, time
from pathlib import Path
from openai import OpenAI

try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("NVIDIA_API_KEY")  # Read Kaggle secret
except ImportError:
    API_KEY = os.environ["NVIDIA_API_KEY"]                       # Read local environment variable

MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 20))               # Set MAX_ROUNDS env var to override (e.g. 1 for debug)
PATIENCE = 4                                                     # Stop after this many rounds without improvement
LOG_FILE = Path("results.json")                                  # Experiment history (round 0 = baseline)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",              # NVIDIA API endpoint
    api_key=API_KEY,
    timeout=120,
)

def load_log():
    log = json.loads(LOG_FILE.read_text())                       # results.json ships with baseline as round 0
    successes = sum(1 for r in log if r.get("status") == "success")
    failures  = sum(1 for r in log if r.get("status") == "failure")
    print(f"[log] {len(log)} entries ({successes} succeeded, {failures} failed)")
    return log

def save_log(log):
    LOG_FILE.write_text(json.dumps(log, indent=2))

def plateau(log):
    losses = [r["loss"] for r in log if r.get("status") == "success"]
    return len(losses) > PATIENCE and min(losses[-PATIENCE:]) >= min(losses[:-PATIENCE])

def ask_model(code, log):
    print("[llm] Sending request...", flush=True)

    failures = [r for r in log if r.get("status") == "failure"]
    recent_failures = failures[-10:]                             # Last 10 failures gives broad avoidance context

    prompt = f"""Improve this small MLP language model with ONE small change.

Current best code (infer what already works from this):
```python
{code}
```

Failed/rejected attempts to AVOID repeating:
{json.dumps(recent_failures, indent=2)}

Return exactly this format:

IDEA: one short sentence describing the single change
```python
complete replacement code
```

Rules:
- Keep TRAIN_SECONDS = 60
- Keep torch.manual_seed(42)
- Keep the FINAL | Loss: ... | Epochs: ... output line
- Change one idea only
"""
    t0 = time.time()
    chunks, tokens = [], 0
    stream = client.chat.completions.create(
        model="z-ai/glm-5.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=8192,
        stream=True,                                             # Stream so we see progress, not a silent hang
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        chunks.append(delta)
        tokens += 1
        if tokens % 50 == 0:
            print(f"[llm] ...{tokens} tokens ({time.time()-t0:.0f}s)", flush=True)

    text = "".join(chunks)
    print(f"[llm] Done: {tokens} tokens in {time.time()-t0:.1f}s", flush=True)

    idea_match = re.search(r"IDEA:\s*(.+)", text)
    code_match = re.search(r"```python\s*(.*?)```", text, re.S)
    if not idea_match or not code_match:
        raise ValueError(f"Bad response format. Raw:\n{text[:500]}")

    return idea_match.group(1).strip(), code_match.group(1).strip()

def run(code):
    if int(re.search(r"TRAIN_SECONDS\s*=\s*(\d+)", code).group(1)) != 60:
        raise ValueError("TRAIN_SECONDS must remain 60.")        # Reject unfair time budget

    Path("current_experiment.py").write_text(code)
    print(f"[run] Launching experiment...", flush=True)
    result = subprocess.run(
        [sys.executable, "current_experiment.py"],
        capture_output=True, text=True, timeout=90,              # 60s training + startup margin
    )
    print(result.stdout)
    if result.stderr:
        print(f"[run] stderr: {result.stderr[-500:]}")
    if result.returncode:
        raise RuntimeError(result.stderr[-500:])

    final = [l for l in result.stdout.splitlines() if l.startswith("FINAL")][-1]
    loss   = float(re.search(r"Loss: ([\d.]+)",  final).group(1))
    epochs = int(re.search(r"Epochs: (\d+)",     final).group(1))
    return loss, epochs

def main():
    log = load_log()
    best_code = Path("mlp_lm.py").read_text()                    # Current best model code

    rounds_done = max(r["round"] for r in log)                   # Resume from last attempted round
    print(f"[main] Working directory: {Path.cwd()}")
    print(f"[main] Rounds done: {rounds_done} | Remaining: {MAX_ROUNDS - rounds_done}")

    for round_num in range(rounds_done + 1, MAX_ROUNDS + 1):
        print(f"\n{'='*50}")
        print(f"[main] Round {round_num}/{MAX_ROUNDS}", flush=True)

        if plateau(log):
            print("[main] Plateau reached.")
            break

        try:
            idea, candidate_code = ask_model(best_code, log)
            print(f"Idea: {idea}")
        except Exception as e:
            print(f"Round {round_num} LLM failed: {e}")
            log.append({"round": round_num, "status": "failure", "reason": f"llm: {e}"})
            save_log(log)
            continue

        try:
            loss, epochs = run(candidate_code)
        except Exception as e:
            print(f"Round {round_num} code crashed: {e}")
            log.append({"round": round_num, "status": "failure", "idea": idea, "reason": f"run: {e}"})
            save_log(log)
            continue

        best_loss = min(r["loss"] for r in log if "loss" in r)   # Includes baseline
        if loss < best_loss:
            log.append({"round": round_num, "status": "success", "idea": idea, "loss": loss, "epochs": epochs})
            save_log(log)
            print(f"Loss: {loss:.4f} | Epochs: {epochs} | Accepted")
            best_code = candidate_code
            Path("mlp_lm.py").write_text(best_code)              # Persist new best
        else:
            log.append({"round": round_num, "status": "failure", "idea": idea, "reason": f"no improvement | loss: {loss:.4f}"})
            save_log(log)
            print(f"Loss: {loss:.4f} | Epochs: {epochs} | Rejected")

if __name__ == "__main__":
    main()

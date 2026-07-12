import json, os, re, subprocess, sys
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

print(f"[init] MAX_ROUNDS={MAX_ROUNDS} | PATIENCE={PATIENCE} | LOG_FILE={LOG_FILE}")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",              # NVIDIA API endpoint
    api_key=API_KEY,                                             # NVIDIA API key
)

# --- Helpers ---
def load_log():
    if LOG_FILE.exists():
        log = json.loads(LOG_FILE.read_text())
        print(f"[log] Resumed {len(log)} previous experiments from {LOG_FILE}")
        return log
    print("[log] No previous results found, starting fresh")
    return []

def save_log(log):
    LOG_FILE.write_text(json.dumps(log, indent=2))               # Save results after each experiment

def plateau(log):
    losses = [run["loss"] for run in log if "loss" in run]       # Ignore failed experiments
    return len(losses) > PATIENCE and min(losses[-PATIENCE:]) >= min(losses[:-PATIENCE])  # No new best loss

def ask_model(code, log):
    print("[llm] Sending request to GLM-5.2...")
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
    response = client.chat.completions.create(
        model="z-ai/glm-5.2",                                    # NVIDIA-hosted GLM-5.2
        messages=[{"role": "user", "content": prompt}],          # Send code + past results
        temperature=0.5,                                         # Prefer focused suggestions
        max_tokens=8192,                                         # Enough room for full script
    )

    text = response.choices[0].message.content                   # Read model reply
    print(f"[llm] Response received ({len(text)} chars)")
    idea = re.search(r"IDEA:\s*(.+)", text).group(1).strip()     # Extract one-line hypothesis
    code = re.search(r"```python\s*(.*?)```", text, re.S).group(1).strip()  # Extract candidate code
    print(f"[llm] Idea: {idea}")
    print(f"[llm] Extracted code ({len(code)} chars)")
    return idea, code

def run(code):
    train_seconds = int(re.search(r"TRAIN_SECONDS\s*=\s*(\d+)", code).group(1))  # Read candidate time limit
    if train_seconds != 60:
        raise ValueError("TRAIN_SECONDS must remain 60.")        # Reject unfair time budget

    Path("current_experiment.py").write_text(code)               # Save proposed experiment
    print(f"[run] Saved current_experiment.py ({len(code)} chars)")
    print(f"[run] Launching subprocess: {sys.executable} current_experiment.py")

    result = subprocess.run(
        [sys.executable, "current_experiment.py"],                # Run in current Kaggle session
        capture_output=True,
        text=True,
        timeout=90,                                               # 60s training + startup margin
    )

    print(f"[run] Subprocess finished (returncode={result.returncode})")
    print(result.stdout)                                          # Show training progress
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
    log = load_log()                                              # Resume existing experiment history

    mlp_path = Path("mlp_lm.py")
    print(f"[main] Reading base model from {mlp_path.resolve()}")
    if not mlp_path.exists():
        raise FileNotFoundError(f"[main] {mlp_path} not found — are you in the AutoResearch directory?")
    best_code = mlp_path.read_text()                              # Start from current best code
    print(f"[main] Base model loaded ({len(best_code)} chars)")

    for round_num in range(len(log) + 1, MAX_ROUNDS + 1):
        print(f"\n{'='*50}")
        print(f"[main] Starting Round {round_num}/{MAX_ROUNDS}")

        if plateau(log):
            print("[main] Plateau reached — no improvement in last {PATIENCE} rounds.")
            break

        try:
            idea, candidate_code = ask_model(best_code, log)     # Ask GLM for hypothesis + code
            print(f"\nRound {round_num}: {idea}")                # Show tested idea
            loss, acc = run(candidate_code)                      # Run proposed experiment

            old_losses = [r["loss"] for r in log if "loss" in r] # Earlier successful losses
            improved = loss < min(old_losses, default=float("inf"))  # Compare with best result

            log.append({
                "round": round_num,
                "idea": idea,
                "loss": loss,
                "acc": acc,
                "improved": improved,
            })
            save_log(log)                                        # Persist result before next round

            print(f"Loss: {loss:.4f} | Acc: {acc:.1%}")          # Short experiment summary

            if improved:
                best_code = candidate_code                       # Use winner as next baseline
                Path("mlp_lm.py").write_text(best_code)          # Save latest best code
                print("Accepted.")
            else:
                print("Rejected.")

        except Exception as error:
            print(f"Round {round_num} failed: {error}")          # Continue after invalid proposals
            log.append({"round": round_num, "error": str(error)})
            save_log(log)

if __name__ == "__main__":
    main()

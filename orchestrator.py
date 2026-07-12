import json, os, re, subprocess, sys
from pathlib import Path
from openai import OpenAI

try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("NVIDIA_API_KEY")  # Read Kaggle secret
except ImportError:
    API_KEY = os.environ["NVIDIA_API_KEY"]  # Read local environment variable

MAX_ROUNDS = 20         # Maximum experiments in one run
PATIENCE = 4            # Stop after this many failed experiments
LOG_FILE = Path("results.json")  # Saved experiment history

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",  # NVIDIA OpenAI-compatible endpoint
    api_key=API_KEY,  # NVIDIA API key
)

# --- Helpers ---
def load_log():
    return json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []  # Load prior results

def save_log(log):
    LOG_FILE.write_text(json.dumps(log, indent=2))  # Save results after every experiment

def plateau(log):
    losses = [run["loss"] for run in log if "loss" in run]  # Ignore failed runs
    if len(losses) <= PATIENCE:
        return False  # Need results before checking plateau
    return min(losses[-PATIENCE:]) >= min(losses[:-PATIENCE])  # No recent best loss

def ask_model(code, log):
    prompt = f"""Improve this small MLP language model with ONE small change.

Current code:
```python
{code}
```

Recent results:
{json.dumps(log[-5:], indent=2)}

Return only the full replacement Python code in one ```python block.

Rules:
- Keep NUM_STORIES = 100
- Keep CONTEXT_SIZE = 32
- Keep TRAIN_SECONDS = 60
- Keep torch.manual_seed(42)
- Keep the FINAL | Loss: ... | Acc: ... output line
- Change one idea only
"""
    response = client.chat.completions.create(
        model="z-ai/glm-5.2",  # NVIDIA hosted GLM model
        messages=[{"role": "user", "content": prompt}],  # Send code + results
        temperature=0.5,  # Prefer focused changes
        max_tokens=8192,  # Enough room for full script
    )

    text = response.choices.message.content  # Read GLM response
    match = re.search(r"```python\s*(.*?)```", text, re.S)  # Extract Python block
    return match.group(1).strip() if match else text.strip()  # Fall back to raw response

def run(code):
    Path("current_experiment.py").write_text(code)  # Save proposed experiment

    result = subprocess.run(
        [sys.executable, "current_experiment.py"],  # Run candidate in this Kaggle session
        capture_output=True,
        text=True,
        timeout=90,  # 60s training + startup margin
    )

    print(result.stdout)  # Show candidate training progress

    if result.returncode:
        raise RuntimeError(result.stderr[-500:])  # Log useful end of error message

    final = [line for line in result.stdout.splitlines() if line.startswith("FINAL")][-1]  # Find final metrics
    loss = float(re.search(r"Loss: ([\d.]+)", final).group(1))  # Parse lower-is-better loss
    acc = float(re.search(r"Acc: ([\d.]+)%", final).group(1)) / 100  # Parse percentage accuracy
    return loss, acc

# --- Research loop ---
def main():
    log = load_log()  # Continue previous Kaggle runs if results.json exists
    best_code = Path("mlp_lm.py").read_text()  # Start from current best code

    for round_num in range(len(log) + 1, MAX_ROUNDS + 1):
        if plateau(log):
            print("Plateau reached.")  # Stop after PATIENCE non-improving runs
            break

        try:
            candidate_code = ask_model(best_code, log)  # Ask GLM for one new idea
            loss, acc = run(candidate_code)  # Test it locally on Kaggle GPU

            old_losses = [run["loss"] for run in log if "loss" in run]  # Earlier successful losses
            improved = loss < min(old_losses, default=float("inf"))  # Compare with best result

            log.append({
                "round": round_num,
                "loss": loss,
                "acc": acc,
                "improved": improved,
            })
            save_log(log)  # Persist result before next round

            print(f"Round {round_num} | Loss: {loss:.4f} | Acc: {acc:.1%}")  # Short summary

            if improved:
                best_code = candidate_code  # Use winner as next starting point
                Path("mlp_lm.py").write_text(best_code)  # Save new best experiment
                print("Accepted.")
            else:
                print("Rejected.")

        except Exception as error:
            print(f"Round {round_num} failed: {error}")  # Continue after bad proposals
            log.append({"round": round_num, "error": str(error)})
            save_log(log)

if __name__ == "__main__":
    main()

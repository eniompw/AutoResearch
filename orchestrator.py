import json
import os
import re
import subprocess
import sys
from pathlib import Path

from openai import OpenAI

try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("NVIDIA_API_KEY")
except ImportError:
    API_KEY = os.environ["NVIDIA_API_KEY"]

MAX_ROUNDS = 20
PATIENCE = 4
LOG_FILE = Path("results.json")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY,
)


def load_log():
    return json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []


def save_log(log):
    LOG_FILE.write_text(json.dumps(log, indent=2))


def plateau(log):
    runs = [x for x in log if "loss" in x]
    if len(runs) < PATIENCE:
        return False

    old_best = min(x["loss"] for x in runs[:-PATIENCE])
    recent_best = min(x["loss"] for x in runs[-PATIENCE:])
    return recent_best >= old_best


def ask_model(code, log):
    prompt = f"""Improve this MLP language-model experiment.

Current code:
```python
{code}
```

Recent results:
{json.dumps(log[-5:], indent=2)}

Return only the complete replacement Python code in a ```python block.
Make ONE small change. Keep SEED, and
TRAIN_SECONDS unchanged. The script must print:
FINAL | Loss: ... | Acc: ...%
"""

    response = client.chat.completions.create(
        model="z-ai/glm-5.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=8192,
    )

    text = response.choices.message.content
    match = re.search(r"```python\s*(.*?)```", text, re.S)
    return match.group(1) if match else text


def run(code):
    Path("current_experiment.py").write_text(code)

    result = subprocess.run(
        [sys.executable, "current_experiment.py"],
        text=True,
        capture_output=True,
        timeout=180,
    )

    print(result.stdout)

    if result.returncode:
        raise RuntimeError(result.stderr[-500:])

    final = [x for x in result.stdout.splitlines() if x.startswith("FINAL")][-1]
    loss = float(re.search(r"Loss: ([\d.]+)", final).group(1))
    acc = float(re.search(r"Acc: ([\d.]+)%", final).group(1)) / 100
    return loss, acc


def main():
    log = load_log()
    best_code = Path("mlp_lm.py").read_text()

    for round_num in range(len(log) + 1, MAX_ROUNDS + 1):
        if plateau(log):
            print("Plateau reached.")
            break

        try:
            candidate = ask_model(best_code, log)
            loss, acc = run(candidate)

            best_loss = min((x["loss"] for x in log if "loss" in x), default=float("inf"))
            improved = loss < best_loss

            log.append({
                "round": round_num,
                "loss": loss,
                "acc": acc,
                "improved": improved,
            })
            save_log(log)

            print(f"Round {round_num} | Loss {loss:.4f} | Acc {acc:.1%}")

            if improved:
                best_code = candidate
                Path("mlp_lm.py").write_text(candidate)
                print("Accepted.")
            else:
                print("Rejected.")

        except Exception as error:
            print(f"Round {round_num} failed: {error}")
            log.append({"round": round_num, "error": str(error)})
            save_log(log)


if __name__ == "__main__":
    main()

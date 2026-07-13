import json, os, re, subprocess, sys
from pathlib import Path
from api import ask_model

MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 20))               # Set MAX_ROUNDS env var to override (e.g. 1 for debug)
PATIENCE = 4                                                     # Stop after this many rounds without improvement
LOG_FILE = Path("results.json")                                  # Experiment history (round 0 = baseline)
SAMPLE_LEN = 128                                                 # Chars of generated text to save per entry

def plateau(log):
    losses = [r["loss"] for r in log if r.get("status") == "success"]
    return len(losses) > PATIENCE and min(losses[-PATIENCE:]) >= min(losses[:-PATIENCE])

def run(code):
    if int(re.search(r"TRAIN_SECONDS\s*=\s*(\d+)", code).group(1)) != 60:
        raise ValueError("TRAIN_SECONDS must remain 60.")        # Reject unfair time budget

    Path("current_experiment.py").write_text(code)
    print("[run] Launching experiment...", flush=True)
    result = subprocess.run(
        [sys.executable, "current_experiment.py"],
        capture_output=True, text=True, timeout=90,              # 60s training + startup margin
    )
    print(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stderr[-500:])

    final = [l for l in result.stdout.splitlines() if l.startswith("FINAL")][-1]
    loss  = float(re.search(r"Loss: ([\d.]+)",  final).group(1))
    steps = int(re.search(r"Steps: (\d+)",      final).group(1))

    lines = result.stdout.splitlines()
    final_idx = max(i for i, l in enumerate(lines) if l.startswith("FINAL"))
    sample_lines = [l for l in lines[final_idx+1:] if l.strip()]
    sample = " ".join(sample_lines)[:SAMPLE_LEN] if sample_lines else ""

    return loss, steps, sample

def main():
    log = json.loads(LOG_FILE.read_text())
    print(f"[log] {len(log)} entries ({sum(1 for r in log if r.get('status')=='success')} ok, {sum(1 for r in log if r.get('status')=='failure')} failed)")

    best_code   = Path("mlp_lm.py").read_text()                  # Current best model code
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
            LOG_FILE.write_text(json.dumps(log, indent=2))
            continue

        try:
            loss, steps, sample = run(candidate_code)
        except Exception as e:
            print(f"Round {round_num} code crashed: {e}")
            log.append({"round": round_num, "status": "failure", "idea": idea, "reason": f"run: {e}"})
            LOG_FILE.write_text(json.dumps(log, indent=2))
            continue

        best_loss = min(r["loss"] for r in log if "loss" in r)   # Includes baseline
        if loss < best_loss:
            log.append({"round": round_num, "status": "success", "idea": idea, "loss": loss, "steps": steps, "sample": sample})
            LOG_FILE.write_text(json.dumps(log, indent=2))
            print(f"Loss: {loss:.4f} | Steps: {steps} | Accepted")
            best_code = candidate_code
            Path("mlp_lm.py").write_text(best_code)              # Persist new best
        else:
            log.append({"round": round_num, "status": "failure", "idea": idea, "reason": f"no improvement | loss: {loss:.4f} | steps: {steps}", "sample": sample})
            LOG_FILE.write_text(json.dumps(log, indent=2))
            print(f"Loss: {loss:.4f} | Steps: {steps} | Rejected")

if __name__ == "__main__":
    main()

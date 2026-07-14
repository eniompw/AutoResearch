import json, os, re, subprocess, sys
from pathlib import Path
from api import ask_model  # LLM wrapper that proposes code improvements

MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 20))  # Total optimization rounds; override via env var
PATIENCE = 4                                         # Early-stop if no improvement in last 4 rounds
LOG_FILE = Path("results.json")                      # Persistent experiment log (round 0 = baseline)
SAMPLE_LEN = 128                                     # Max chars of generated text to store per round


def plateau(log):
    # Extract losses only from successful rounds
    losses = [r["loss"] for r in log if r.get("status") == "success"]
    # True if we have enough data AND recent rounds haven't beaten the historical best
    return len(losses) > PATIENCE and min(losses[-PATIENCE:]) >= min(losses[:-PATIENCE])


def run(code):
    # Guard: reject any code that tampers with the 60s training budget
    if int(re.search(r"TRAIN_SECONDS\s*=\s*(\d+)", code).group(1)) != 60:
        raise ValueError("TRAIN_SECONDS must remain 60.")

    Path("current_experiment.py").write_text(code)   # Write candidate code to disk
    print("[run] Launching experiment...", flush=True)
    result = subprocess.run(
        [sys.executable, "current_experiment.py"],   # Run it in a subprocess using same Python
        capture_output=True, text=True, timeout=180,  # 60s training + ~120s for dataset download/cache on first run
    )
    print(result.stdout)
    if result.returncode:                            # Non-zero exit = crash
        raise RuntimeError(result.stderr[-500:])     # Surface last 500 chars of stderr

    # Parse the last FINAL line from stdout (format: "FINAL Loss: X Steps: Y")
    final = [l for l in result.stdout.splitlines() if l.startswith("FINAL")][-1]
    loss  = float(re.search(r"Loss: ([\d.]+)",  final).group(1))
    steps = int(re.search(r"Steps: (\d+)",      final).group(1))

    # Grab any non-empty lines after the last FINAL line as the generated text sample
    lines = result.stdout.splitlines()
    final_idx = max(i for i, l in enumerate(lines) if l.startswith("FINAL"))
    sample_lines = [l for l in lines[final_idx+1:] if l.strip()]
    sample = " ".join(sample_lines)[:SAMPLE_LEN] if sample_lines else ""

    return loss, steps, sample


def main():
    log = json.loads(LOG_FILE.read_text())           # Load existing experiment history
    # Print summary: total entries, successes, failures
    print(f"[log] {len(log)} entries ({sum(1 for r in log if r.get('status')=='success')} ok, {sum(1 for r in log if r.get('status')=='failure')} failed)")

    best_code   = Path("mlp_lm.py").read_text()      # Load current best model code
    rounds_done = max(r["round"] for r in log)       # Resume safely from last attempted round

    for round_num in range(rounds_done + 1, MAX_ROUNDS + 1):
        print(f"\n{'='*50}")
        print(f"[main] Round {round_num}/{MAX_ROUNDS}", flush=True)

        if plateau(log):                             # Early exit if stuck
            print("[main] Plateau reached.")
            break

        try:
            idea, candidate_code = ask_model(best_code, log)  # Ask LLM for next improvement
            print(f"Idea: {idea}")
        except Exception as e:
            # LLM call failed; log and skip to next round
            log.append({"round": round_num, "status": "failure", "reason": f"llm: {e}"})
            LOG_FILE.write_text(json.dumps(log, indent=2))
            continue

        try:
            loss, steps, sample = run(candidate_code)  # Execute the proposed code
        except Exception as e:
            # Code crashed; log idea + error and skip
            log.append({"round": round_num, "status": "failure", "idea": idea, "reason": f"run: {e}"})
            LOG_FILE.write_text(json.dumps(log, indent=2))
            continue

        best_loss = min(r["loss"] for r in log if "loss" in r)  # Best loss including baseline
        if loss < best_loss:
            # Improvement found: accept, persist new best code
            log.append({"round": round_num, "status": "success", "idea": idea, "loss": loss, "steps": steps, "sample": sample})
            LOG_FILE.write_text(json.dumps(log, indent=2))
            print(f"Loss: {loss:.4f} | Steps: {steps} | Accepted")
            best_code = candidate_code
            Path("mlp_lm.py").write_text(best_code)  # Overwrite best model file on disk
        else:
            # No improvement: reject but store loss/steps as fields for proper logging
            log.append({"round": round_num, "status": "failure", "idea": idea, "loss": loss, "steps": steps, "reason": f"no improvement | loss: {loss:.4f} | steps: {steps}", "sample": sample})
            LOG_FILE.write_text(json.dumps(log, indent=2))
            print(f"Loss: {loss:.4f} | Steps: {steps} | Rejected")


if __name__ == "__main__":
    main()

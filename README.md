# AutoResearch

A small automated ML research loop.

[GLM-5.2](https://build.nvidia.com/z-ai/glm-5.2) suggests one improvement to a TinyStories MLP, Kaggle tests it for 60 seconds, and the loop keeps the change only when training loss improves.

## How it works

1. GLM-5.2 reads `mlp_lm.py` and recent experiment results
2. It suggests one small code change and explains the idea
3. The candidate trains for 60 seconds on Kaggle GPU
4. The idea, loss, and epochs are saved to `results.json`
5. Better candidates replace `mlp_lm.py`
6. The loop stops after 4 experiments without a new best loss

## Files

| File | Purpose |
|---|---|
| `mlp_lm.py` | Current best model (updated each accepted round) |
| `mlp_lm_base.py` | Original unmodified baseline model |
| `tinystories_dataset.py` | Loads TinyStories and creates context-target pairs |
| `orchestrator.py` | Calls GLM-5.2, runs experiments, and saves results |
| `results.json` | Experiment history — pre-seeded with round 0 baseline |

## Run on Kaggle

### 1. Get an NVIDIA API key

Register at [GLM-5.2 on NVIDIA Build](https://build.nvidia.com/z-ai/glm-5.2) to get a free API key.

### 2. Create a Kaggle notebook

1. Enable a **T4 GPU** accelerator in **Notebook options** (see [GPU note](#gpu-compatibility) below).
2. Enable **Internet** in **Notebook options**.
3. Add your key as `NVIDIA_API_KEY` in **Add-ons -> Secrets**.

### 3. Run this cell

```python
import os
os.environ["MAX_ROUNDS"] = "1"   # Increase for longer runs

%cd /kaggle/working
!git clone https://github.com/eniompw/AutoResearch.git  # First run only — comment out after
#!git stash && git pull --ff-only origin main && git stash pop  # Uncomment for subsequent runs
%cd /kaggle/working/AutoResearch
!pip install -q openai  # First run only — comment out after
!python orchestrator.py
```

`results.json` and `mlp_lm.py` persist across Kaggle sessions. The stash/pull/pop pattern updates source files while preserving your local experiment results.

Always `%cd /kaggle/working` before `git clone` so the kernel's working directory exists regardless of which cell was run last. Using absolute paths for `%cd` avoids `getcwd` errors if the notebook was previously inside `AutoResearch`.

## GPU Compatibility

> **Do not use the P100.** Kaggle's default PyTorch environment uses CUDA 12.8+, which dropped support for the P100's Pascal architecture (SM 6.0). This causes:
> ```
> torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
> ```
> Use the **T4** (or newer) instead. T4 is Turing architecture (SM 7.5) and is fully supported.

## Settings

Edit `mlp_lm.py`:

```python
TRAIN_SECONDS = 60   # Training time for every experiment
LOG_EVERY     = 1000 # Print metrics every N epochs
```

Override `orchestrator.py` settings via environment variables:

```python
import os
os.environ["MAX_ROUNDS"] = "3"   # Default: 20
```

Keep `TRAIN_SECONDS` fixed during one run. Otherwise, a candidate could appear better simply because it trained longer.

## Results

Each experiment is saved to `results.json` with a `status` of `success` or `failure`. Round 0 is the pre-seeded baseline:

```json
[
  {"round": 0, "status": "success", "idea": "baseline", "loss": 2.6551, "epochs": 17701}
]
```

Lower training loss is better. A candidate is accepted only when its loss beats all previous experiments including the baseline. The `epochs` field shows how many training epochs completed in 60 seconds — a low value means the change made training significantly slower.

## Debugging

Check experiment history:

```python
import json
print(json.load(open('/kaggle/working/AutoResearch/results.json')))
```

If `results.json` has a git merge conflict, restore the clean version:

```bash
git checkout main -- results.json
```

Reset to baseline and start fresh:

```python
# Move out of the folder before deleting it
%cd /kaggle/working

import shutil
shutil.rmtree("AutoResearch", ignore_errors=True)

!git clone https://github.com/eniompw/AutoResearch.git
%cd /kaggle/working/AutoResearch

!pip install -q openai
!python orchestrator.py
```

Do not delete `/kaggle/working/AutoResearch` while the notebook is currently inside that directory. First run `%cd /kaggle/working`; otherwise the kernel's working directory no longer exists, and `git clone`, `pip`, and Python can fail with `getcwd` errors.

## Notes

- `results.json` and `mlp_lm.py` persist across Kaggle sessions — no need to push to git.
- Experiments run sequentially inside one Kaggle notebook session.
- This is an educational project, not a rigorous benchmark.
- Inspect accepted code before relying on it.

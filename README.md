# 🤖 AutoResearch

A small automated ML research loop — free LLM API, free GPU.

An LLM ([GLM-5.2](https://build.nvidia.com/z-ai/glm-5.2)) proposes one code change to a TinyStories MLP, [Kaggle](https://www.kaggle.com) trains it for 60 seconds on a free T4 GPU, and the loop keeps the change only when training loss improves. Repeat.

## 📚 Background

After 65+ manual experiments across five models (MLP → BPE transformer) documented in [TinyLM/BENCHMARKS.md](https://github.com/eniompw/TinyLM/blob/main/BENCHMARKS.md), a clear pattern emerged: every round is just *read model → propose one change → train → keep if loss improves*. That's a loop an LLM can run.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), but built entirely on free resources — the [NVIDIA GLM-5.2 API](https://build.nvidia.com/z-ai/glm-5.2) for suggestions and a [Kaggle](https://www.kaggle.com) T4 for GPU training.

The **MLP** is the deliberate starting point: no attention, minimal code, easy to follow for anyone without a transformer background — yet it still surfaces the core research challenges (memorization, capacity limits, speed/accuracy trade-offs).

## ⚙️ How it works

1. 📖 GLM-5.2 reads `mlp_lm.py` and recent experiment results
2. 💡 It suggests one small code change and explains the idea
3. ⏱️ The candidate trains for 60 seconds on Kaggle GPU
4. 💾 The idea, loss, steps, and a 128-char generated text sample are saved to `results.json`
5. ✅ Better candidates replace `mlp_lm.py`
6. 🛑 The loop stops after 4 experiments without a new best loss

## 📂 Files

| File | Purpose |
|---|---|
| [`mlp_lm.py`](mlp_lm.py) | 🏆 Current best model (updated each accepted round) |
| [`mlp_lm_base.py`](mlp_lm_base.py) | 📌 Original unmodified baseline model |
| [`tinystories_dataset.py`](tinystories_dataset.py) | 📚 Loads TinyStories and creates context-target pairs |
| [`api.py`](api.py) | 🔌 NVIDIA API client setup and `ask_model()` — all LLM interaction |
| [`orchestrator.py`](orchestrator.py) | 🎼 Main research loop — runs experiments and saves results |
| [`results.json`](results.json) | 📊 Experiment history — pre-seeded with round 0 baseline |

## 🚀 Run on Kaggle

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
%cd /kaggle/working/AutoResearch
#!git stash && git pull --ff-only origin main && git stash pop  # Uncomment for subsequent runs
!pip install -q openai  # First run only — comment out after
!python orchestrator.py
```

`results.json` and `mlp_lm.py` persist across Kaggle sessions. The stash/pull/pop pattern updates source files while preserving your local experiment results.

Always `%cd /kaggle/working` before `git clone` so the kernel's working directory exists regardless of which cell was run last. Using absolute paths for `%cd` avoids `getcwd` errors if the notebook was previously inside `AutoResearch`.

## ⚠️ GPU Compatibility

> **Do not use the P100.** Kaggle's default PyTorch environment uses CUDA 12.8+, which dropped support for the P100's Pascal architecture (SM 6.0). This causes:
> ```
> torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
> ```
> Use the **T4** (or newer) instead. T4 is Turing architecture (SM 7.5) and is fully supported.

## 🚀 Run on Google Colab

### 1. Get an NVIDIA API key

Register at [GLM-5.2 on NVIDIA Build](https://build.nvidia.com/z-ai/glm-5.2) to get a free API key.

### 2. Create a Colab notebook

1. Enable a **T4 GPU** via **Runtime → Change runtime type → T4 GPU**.
2. Add your key as `NVIDIA_API_KEY` in **Secrets** (🔑 icon in the left sidebar), and enable notebook access.

### 3. Run this cell

```python
import os
os.environ["MAX_ROUNDS"] = "1"   # Increase for longer runs

from google.colab import userdata
os.environ["NVIDIA_API_KEY"] = userdata.get("NVIDIA_API_KEY")

%cd /content
!git clone https://github.com/eniompw/AutoResearch.git  # First run only — comment out after
%cd /content/AutoResearch
#!git stash && git pull --ff-only origin main && git stash pop  # Uncomment for subsequent runs
!pip install -q openai  # First run only — comment out after
!python orchestrator.py
```

> **Note:** Colab sessions reset when the runtime disconnects — `results.json` and `mlp_lm.py` are lost unless you save them manually (e.g. mount Google Drive or push to git).

## 🔧 Settings

Edit `mlp_lm.py`:

```python
TRAIN_SECONDS = 60   # Training time for every experiment
LOG_EVERY     = 1000 # Print metrics every N steps
```

Override `orchestrator.py` settings via environment variables:

```python
import os
os.environ["MAX_ROUNDS"] = "3"   # Default: 20
```

Keep `TRAIN_SECONDS` fixed during one run. Otherwise, a candidate could appear better simply because it trained longer.

## 📊 Results

Each experiment is saved to `results.json` with a `status` of `success` or `failure`. Round 0 is the pre-seeded baseline:

```json
[
  {
    "round": 0,
    "status": "success",
    "idea": "baseline",
    "loss": 2.271498,
    "steps": 31016,
    "sample": " ald there poimyo.\" \"Wayor way and plirth ild fver a tomard. an the mat Rhe hant tamlerit. \"Day anel hponme and no xonds. They t"
  }
]
```

Lower training loss is better. A candidate is accepted only when its loss beats all previous experiments including the baseline. The `steps` field shows how many gradient steps completed in 60 seconds — a low value means the change made training significantly slower. The `sample` field holds the first 128 characters of generated text, giving a quick qualitative check of output coherence alongside the loss metric.

## 🐛 Debugging

Check experiment history as a table:

```python
import json
import pandas as pd

data = json.load(open('/kaggle/working/AutoResearch/results.json'))
df = pd.DataFrame(data)
df
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

## 📝 Notes

- `results.json` and `mlp_lm.py` persist across Kaggle sessions — no need to push to git.
- Experiments run sequentially inside one Kaggle notebook session.
- This is an educational project, not a rigorous benchmark.
- Inspect accepted code before relying on it.

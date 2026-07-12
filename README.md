# AutoResearch

A small automated ML research loop.

[GLM-5.2](https://build.nvidia.com/z-ai/glm-5.2) suggests one improvement to a TinyStories MLP, Kaggle tests it for 60 seconds, and the loop keeps the change only when training loss improves.

## How it works

1. GLM-5.2 reads `mlp_lm.py` and recent experiment results
2. It suggests one small code change and explains the idea
3. The candidate trains for 60 seconds on Kaggle GPU
4. The idea, loss, and accuracy are saved to `results.json`
5. Better candidates replace `mlp_lm.py`
6. The loop stops after 4 experiments without a new best loss

## Files

| File | Purpose |
|---|---|
| `mlp_lm.py` | Small character-level MLP trained on TinyStories |
| `tinystories_dataset.py` | Loads TinyStories and creates context-target pairs |
| `orchestrator.py` | Calls GLM-5.2, runs experiments, and saves results |
| `results.json` | Experiment history, created automatically |

## Run on Kaggle

1. Create a Kaggle notebook.
2. Enable a GPU accelerator in **Notebook options**.
3. Enable **Internet** in **Notebook options**.
4. Add `NVIDIA_API_KEY` in **Add-ons -> Secrets**.
5. Run this cell:

```python
!git clone https://github.com/eniompw/AutoResearch.git
%cd AutoResearch
!pip install -q openai
!python orchestrator.py
```

Get an NVIDIA API key from [GLM-5.2 on NVIDIA Build](https://build.nvidia.com/z-ai/glm-5.2).

## Settings

Edit `mlp_lm.py`:

```python
TRAIN_SECONDS = 60  # Training time for every experiment
```

Override `orchestrator.py` settings via environment variables (useful on Kaggle where editing files is awkward):

```python
import os
os.environ["MAX_ROUNDS"] = "1"   # Default: 20 — set to 1 for a quick debug run
os.environ["PATIENCE"] = "2"     # Default: 4 — stop after this many non-improving experiments
```

Or edit `orchestrator.py` directly:

```python
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 20))  # Maximum experiments per run
PATIENCE = 4                                         # Stop after this many non-improving experiments
```

Keep `TRAIN_SECONDS` fixed during one run. Otherwise, a candidate could appear better simply because it trained longer.

## Results

Each successful experiment is saved to `results.json`:

```json
{
  "round": 3,
  "idea": "Replace ReLU with GELU to improve hidden-layer gradients.",
  "loss": 1.8234,
  "acc": 0.412,
  "improved": true
}
```

Lower training loss is better. A candidate is accepted only when its loss is lower than every previous successful experiment.

## Notes

- Experiments run sequentially inside one Kaggle notebook session.
- This is an educational project, not a rigorous benchmark.
- The baseline uses training loss and training accuracy; it does not yet use a validation split.
- Inspect accepted code before relying on it.

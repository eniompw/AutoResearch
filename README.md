# AutoResearch

An automated ML research loop that uses [GLM-5.2](https://build.nvidia.com/z-ai/glm-5.2) (free via NVIDIA API) to iteratively suggest and test improvements to a small MLP language model trained on TinyStories. The loop runs on free Kaggle GPU compute and stops automatically when improvements plateau.

## How It Works

1. GLM-5.2 reads the current `mlp_lm.py` and experiment history, then suggests one improvement
2. The modified code is tested on a Kaggle GPU notebook
3. Results (loss, accuracy) are logged to `results.json`
4. If 2 of 3 plateau signals fire, the loop stops — otherwise repeat

## Files

| File | Purpose |
|---|---|
| `mlp_lm.py` | Base MLP language model being optimized |
| `tinystories_dataset.py` | TinyStories data loader |
| `orchestrator.py` | Main research loop (GLM-5.2 + plateau detection) |
| `results.json` | Auto-generated experiment log |

## Setup

### 1. NVIDIA API Key (free)

- Sign up at [build.nvidia.com](https://build.nvidia.com/z-ai/glm-5.2)
- Generate a free API key (1000 free credits included)

### 2. Kaggle Secrets

- Go to **Kaggle → Account → Add-ons → Secrets**
- Add a new secret: `NVIDIA_API_KEY` = your key from step 1

### 3. Kaggle Notebook

- Create a new Kaggle notebook with **GPU T4 x2** accelerator enabled
- Clone this repo:
  ```bash
  !git clone https://github.com/eniompw/AutoResearch.git
  %cd AutoResearch
  ```
- Install dependency:
  ```bash
  !pip install -q openai
  ```
- Run the loop:
  ```bash
  !python orchestrator.py
  ```

## Plateau Detection

The loop stops when **2 of 3** signals agree there is no more progress:

- **Relative improvement** — best loss hasn't improved by >0.1% over last 4 rounds
- **Linear trend** — OLS slope of last 6 losses is too flat
- **Consecutive failures** — last 4 rounds all failed to beat the all-time best

A hard cap of `MAX_ROUNDS = 20` also prevents runaway usage of free compute.

## Results

Each round is logged to `results.json`:

```json
{
  "round": 3,
  "loss": 1.823,
  "acc": 0.412,
  "improved": true,
  "code_snippet": "..."
}
```

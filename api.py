import json, os, re, time
from openai import OpenAI, APIStatusError, APITimeoutError
import httpx

MODEL = "z-ai/glm-5.2"  # swap model here

if "NVIDIA_API_KEY" not in os.environ:
    raise EnvironmentError("Missing NVIDIA_API_KEY — set it with: export NVIDIA_API_KEY=your_key")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",              # NVIDIA API endpoint
    api_key=os.environ["NVIDIA_API_KEY"],
    timeout=httpx.Timeout(connect=30, read=120, write=30, pool=30),
)

def ask_model(code, log):
    print("[llm] Sending request...", flush=True)

    recent_successes = [r for r in log if r.get("status") == "success"][-5:]  # last 5 wins for context
    recent_failures  = [r for r in log if r.get("status") == "failure"][-5:]  # last 5 losses to avoid repeating

    def slim(entries):
        return [{"idea": e.get("idea"), "loss": e.get("loss"), "steps": e.get("steps"), "reason": e.get("reason")} for e in entries]

    prompt = f"""Improve this PyTorch language model training script with ONE change.
The change can be to any aspect of the code: model, training loop, or data.

IMPORTANT: All experiments run with a fixed 60-second training budget (TRAIN_SECONDS = 60).
The 'steps' field shows how many gradient steps completed in that time.
A low step count means the change made training much slower — avoid ideas that reduce steps significantly.

Current best code (infer what already works from this):
```python
{code}
```

Last 5 successful improvements (use steps to judge training speed):
{json.dumps(slim(recent_successes), indent=2)}

Last 5 failed/rejected attempts to AVOID repeating:
{json.dumps(slim(recent_failures), indent=2)}

Return exactly this format:

IDEA: one short sentence describing the single change
```python
complete replacement code
```

Rules:
- Keep TRAIN_SECONDS = 60
- Keep torch.manual_seed(42)
- Keep the FINAL | Loss: ... | Steps: ... output line
- Change one idea only
"""
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,                                     # moderate creativity; lower = more deterministic
            max_tokens=2048,                                     # MLP code doesn't need more than 2k tokens
        )
        text = resp.choices[0].message.content
        print(f"[llm] Done in {time.time()-t0:.1f}s ({len(text)} chars)", flush=True)

    except APITimeoutError:
        raise RuntimeError(f"[llm] Timeout after {time.time()-t0:.1f}s")
    except APIStatusError as e:
        raise RuntimeError(f"[llm] HTTP {e.status_code}: {e.message}")  # surface HTTP error code and message

    idea_match = re.search(r"IDEA:\s*(.+)", text)                # parse the one-line improvement description
    code_match = re.search(r"```python\s*(.*?)```", text, re.S)  # extract code block (re.S = dotall for multiline)
    if not idea_match or not code_match:
        raise ValueError(f"Bad response format. Raw:\n{text[:500]}")  # show first 500 chars to aid debugging

    return idea_match.group(1).strip(), code_match.group(1).strip()  # return (idea string, code string)

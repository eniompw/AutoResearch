import json, os, re, time
from openai import OpenAI, APIStatusError, APITimeoutError
import httpx

try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("NVIDIA_API_KEY")  # Read Kaggle secret
except ImportError:
    API_KEY = os.environ["NVIDIA_API_KEY"]                       # Read local environment variable

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",              # NVIDIA API endpoint
    api_key=API_KEY,
    timeout=httpx.Timeout(connect=30, read=120, write=30, pool=30),  # read timeout caps stream duration
)

def ask_model(code, log):
    print("[llm] Sending request...", flush=True)

    recent_successes = [r for r in log if r.get("status") == "success"][-5:]  # last 5 wins for context
    recent_failures  = [r for r in log if r.get("status") == "failure"][-5:]  # last 5 losses to avoid repeating

    prompt = f"""Improve this small MLP language model with ONE small change.

IMPORTANT: All experiments run with a fixed 60-second training budget (TRAIN_SECONDS = 60).
The 'steps' field shows how many gradient steps completed in that time.
A low step count means the change made training much slower — avoid ideas that reduce steps significantly.

Current best code (infer what already works from this):
```python
{code}
```

Last 5 successful improvements (use steps to judge training speed):
{json.dumps(recent_successes, indent=2)}

Last 5 failed/rejected attempts to AVOID repeating:
{json.dumps(recent_failures, indent=2)}

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
    t0 = time.time()  # record start time to measure latency
    try:
        stream = client.chat.completions.create(
            model="z-ai/glm-5.2",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,                                     # moderate creativity; lower = more deterministic
            max_tokens=2048,                                     # reduced from 8192; MLP code doesn't need 8k tokens
            stream=True,                                         # stream so we see progress, not a silent hang
        )
        chunks = []
        first_token = True
        for c in stream:
            token = c.choices[0].delta.content or ""            # extract text delta, default "" if None
            if token and first_token:
                print(f"[llm] First token in {time.time()-t0:.1f}s", flush=True)  # time-to-first-token metric
                first_token = False
            chunks.append(token)
        text = "".join(chunks)                                   # reassemble full response from stream
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

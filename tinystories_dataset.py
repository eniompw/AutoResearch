import torch, json

LOCAL_FILE = "tinystories_5k.jsonl"

def load_tinystories(num_stories=500, context_size=4):
    """
    Loads TinyStories from a local JSONL file and prepares it for a character-level language model.
    Returns: input_ids, target_ids, idx_to_char (dict), encoded (list), vocab_size (int)
    """
    print(f"[data] Loading {num_stories} stories from {LOCAL_FILE}...", flush=True)
    with open(LOCAL_FILE) as f:
        text = ''.join(json.loads(line)["text"] for _, line in zip(range(num_stories), f))
    print(f"[data] Ready — {len(text):,} characters", flush=True)

    vocab = sorted(set(text))                                        # ordered list of unique characters
    char_to_id = {c: i for i, c in enumerate(vocab)}                 # char → integer id
    idx_to_char = {i: c for i, c in enumerate(vocab)}                # integer id → char
    encoded = [char_to_id[c] for c in text]                          # full text as integer sequence

    if context_size == 1:
        return [], [], idx_to_char, encoded, len(vocab)

    inputs  = [encoded[i:i+context_size] for i in range(len(encoded)-context_size)]
    targets = encoded[context_size:]

    return torch.tensor(inputs), torch.tensor(targets), idx_to_char, encoded, len(vocab)

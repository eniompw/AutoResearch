import torch, warnings, itertools, os
from datasets import load_dataset
warnings.filterwarnings('ignore')

def load_tinystories(num_stories=500, context_size=4):
    """
    Fetches TinyStories and prepares it for a character-level language model.
    Returns: input_ids, target_ids, idx_to_char (dict), encoded (list), vocab_size (int)
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    cached = os.path.exists(os.path.join(cache_dir, "karpathy___tinystories-gpt4-clean"))
    print(f"[data] {'Loading from cache' if cached else 'Downloading dataset (first run)'}...", flush=True)

    dataset = load_dataset('karpathy/tinystories-gpt4-clean', split='train')  # cached locally after first download
    print(f"[data] Ready \u2014 using {num_stories} stories", flush=True)

    text = ''.join(s['text'] for s in itertools.islice(dataset, num_stories))

    vocab = sorted(set(text))                                        # ordered list of unique characters
    char_to_id = {c: i for i, c in enumerate(vocab)}                 # char → integer id
    idx_to_char = {i: c for i, c in enumerate(vocab)}                # integer id → char
    encoded = [char_to_id[c] for c in text]                          # full text as integer sequence

    if context_size == 1:
        return [], [], idx_to_char, encoded, len(vocab)

    inputs  = [encoded[i:i+context_size] for i in range(len(encoded)-context_size)]
    targets = encoded[context_size:]

    return torch.tensor(inputs), torch.tensor(targets), idx_to_char, encoded, len(vocab)

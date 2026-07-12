import torch
import torch.nn as nn
from tinystories_dataset import load_tinystories

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Hyperparameters ---
NUM_STORIES  = 100   # number of TinyStories to load (more = better, but slower)
CONTEXT_SIZE = 32    # how many characters the model looks back at
HIDDEN_SIZE  = 64    # number of neurons in the hidden layer
EPOCHS       = 500   # how many times to loop through the training data
LR           = 0.01  # learning rate: how big each weight update step is
LOG_EVERY    = 100   # print progress every N epochs
GEN_CHARS    = 200   # number of characters to generate after training

# --- Data ---
inputs, targets, idx_to_char, _, vocab_size = load_tinystories(NUM_STORIES, CONTEXT_SIZE)
inputs = inputs.float()  # convert token ids to float for the linear layer

# --- Model ---
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(CONTEXT_SIZE, HIDDEN_SIZE),  # input: context window of character ids
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, vocab_size),    # output: score for each character in vocab
)

# --- Train ---
for epoch in range(EPOCHS):
    logits = model(inputs)
    loss = nn.functional.cross_entropy(logits, targets)
    loss.backward()

    with torch.no_grad():
        for p in model.parameters(): p -= LR * p.grad; p.grad.zero_()

    if epoch % LOG_EVERY == 0:
        acc = (logits.argmax(1) == targets).float().mean()
        print(f"Epoch {epoch:4d} | Loss: {loss:.3f} | Acc: {acc:.1%}")

# --- Generate ---
context = inputs[0].tolist()  # start with the first training example as seed
for _ in range(GEN_CHARS):
    inp = torch.tensor(context[-CONTEXT_SIZE:], dtype=torch.float).unsqueeze(0)
    next_char_id = model(inp).argmax(1).item()
    context.append(next_char_id)

print(''.join(idx_to_char[i] for i in context[CONTEXT_SIZE:]))

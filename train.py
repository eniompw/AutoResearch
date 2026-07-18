import time, torch                                                        # Timing + tensors/GPU tools
import torch.nn as nn                                                     # Neural-network layers
from tinystories_dataset import load_tinystories                          # TinyStories character dataset loader

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu') # Put new tensors on GPU when available

# --- Hyperparameters ---
NUM_STORIES   = 100      # Number of TinyStories to load
CONTEXT_SIZE  = 4        # Previous characters used to predict the next one
EMBED_SIZE    = 8        # Embedding dimensions per character
HIDDEN_SIZE   = 16       # Neurons in the hidden layer
BATCH_SIZE    = 8        # Training pairs per gradient step
LR            = 1e-3     # Adam learning rate
ACT           = nn.ReLU  # Activation function
TRAIN_SECONDS = 60       # Fixed budget so all experiments are comparable
LOG_EVERY     = 1000     # Print training metrics every N steps
GEN_CHARS     = 128      # Characters to generate after training
TEMP          = 1.0      # Sampling temperature (lower = more focused)

# --- Data ---
inputs, targets, idx_to_char, _, vocab_size = load_tinystories(NUM_STORIES, CONTEXT_SIZE) # Context-target pairs
# inputs is already int64 (torch.tensor of Python ints); .long() is a no-op but kept as defensive habit
inputs = inputs.long()

# --- Model ---
torch.manual_seed(42)                                                     # Same initialisation each experiment
model = nn.Sequential(
    nn.Embedding(vocab_size, EMBED_SIZE),                                 # Learn a dense vector per character
    nn.Flatten(),                                                         # (context, embed) -> (context*embed,)
    nn.Linear(CONTEXT_SIZE * EMBED_SIZE, HIDDEN_SIZE),                   # Map context to hidden features
    ACT(),                                                                # Non-linearity
    nn.Linear(HIDDEN_SIZE, vocab_size),                                   # Score every possible next character
)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)                  # Adam handles learning-rate scaling

# --- Train ---
start_time = time.time()
n, step = inputs.size(0), 0

while time.time() - start_time < TRAIN_SECONDS:
    step += 1
    idx  = torch.randint(0, n, (BATCH_SIZE,))                            # Random mini-batch indices
    loss = nn.functional.cross_entropy(model(inputs[idx]), targets[idx]) # Predict next character
    optimizer.zero_grad(); loss.backward(); optimizer.step()             # Standard gradient-descent step

    if step % LOG_EVERY == 0:
        print(f"Step  {step:6d} | Loss: {loss:.3f}")                     # Show learning progress

# --- Final metrics (parsed by orchestrator) ---
print(f"FINAL | Loss: {loss:.6f} | Steps: {step}")

# --- Generate ---
context = inputs[0].tolist()                                             # First training context as generation seed
with torch.no_grad():                                                    # Inference: no gradients needed (faster)
    for _ in range(GEN_CHARS):
        inp   = torch.tensor(context[-CONTEXT_SIZE:]).unsqueeze(0)      # Shape: (1, CONTEXT_SIZE)
        probs = torch.softmax(model(inp)[0] / TEMP, dim=-1)             # Temperature-scaled probabilities
        context.append(torch.multinomial(probs, 1).item())              # Sample next character

sample = ''.join(idx_to_char[i] for i in context[CONTEXT_SIZE:])       # Decode generated IDs back to text
print(sample.replace(chr(10), ' '))                                      # Plain text; collected by orchestrator after FINAL

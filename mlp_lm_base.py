import time, torch                                                        # Timing + tensors/GPU tools
import torch.nn as nn                                                     # Neural-network layers
from tinystories_dataset import load_tinystories                          # TinyStories character dataset loader

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu') # Put new tensors on GPU when available

# --- Hyperparameters ---
NUM_STORIES   = 100      # Number of TinyStories to load
CONTEXT_SIZE  = 32       # Previous characters used to predict the next one
EMBED_SIZE    = 32       # Embedding dimensions per character
HIDDEN_SIZE   = 64       # Neurons in the MLP hidden layer
BATCH_SIZE    = 256      # Training pairs per gradient step
LR            = 3e-3     # Adam learning rate
ACT           = nn.ReLU  # Activation function: try nn.Tanh, nn.GELU, nn.SiLU
TRAIN_SECONDS = 60       # Fixed budget so all experiments are comparable
LOG_EVERY     = 1000     # Print training metrics every N steps
GEN_CHARS     = 200      # Characters to generate after training
TEMP          = 0.8      # Sampling temperature (lower = more focused)

# --- Data ---
inputs, targets, idx_to_char, _, vocab_size = load_tinystories(NUM_STORIES, CONTEXT_SIZE) # Context-target pairs
inputs = inputs.long()                                                    # Embedding expects integer indices

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

for step in range(1_000_000):
    idx  = torch.randint(0, n, (BATCH_SIZE,))                            # Random mini-batch indices
    loss = nn.functional.cross_entropy(model(inputs[idx]), targets[idx]) # Predict next character
    optimizer.zero_grad(); loss.backward(); optimizer.step()             # Standard gradient-descent step

    if step % LOG_EVERY == 0:
        print(f"Step  {step:6d} | Loss: {loss:.3f}")                     # Show learning progress

    if time.time() - start_time >= TRAIN_SECONDS:                        # Stop at shared experiment budget
        break

# --- Final metrics (parsed by orchestrator) ---
print(f"FINAL | Loss: {loss:.6f} | Steps: {step+1}")

# --- Generate ---
context = inputs[0].tolist()                                             # First training context as generation seed
for _ in range(GEN_CHARS):
    inp   = torch.tensor(context[-CONTEXT_SIZE:]).unsqueeze(0)          # Shape: (1, CONTEXT_SIZE)
    probs = torch.softmax(model(inp)[0] / TEMP, dim=-1)                 # Temperature-scaled probabilities
    context.append(torch.multinomial(probs, 1).item())                  # Sample next character

print(''.join(idx_to_char[i] for i in context[CONTEXT_SIZE:]))          # Decode generated IDs back to text

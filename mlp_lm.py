import time, torch                                                        # Timing + tensors/GPU tools
import torch.nn as nn                                                     # Neural-network layers
from tinystories_dataset import load_tinystories                          # TinyStories character dataset loader

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu') # Put new tensors on GPU when available

# --- Hyperparameters ---
NUM_STORIES   = 100       # Number of TinyStories to load
CONTEXT_SIZE  = 32        # Previous characters used to predict the next one
HIDDEN_SIZE   = 64        # Neurons in the MLP hidden layer
EPOCHS        = 100_000   # Safety cap; training normally stops at 60 seconds
LR            = 0.01      # Step size for each weight update
LOG_EVERY     = 1000      # Print training metrics every N epochs
GEN_CHARS     = 200       # Characters to generate after training
TRAIN_SECONDS = 60        # Fixed budget so all experiments are comparable

# --- Data ---
inputs, targets, idx_to_char, _, vocab_size = load_tinystories(NUM_STORIES, CONTEXT_SIZE) # Create context-target training pairs
inputs = inputs.float()                                                  # Linear layers require float inputs

# --- Model ---
torch.manual_seed(42)                                                    # Same initialization each experiment
model = nn.Sequential(
    nn.Linear(CONTEXT_SIZE, HIDDEN_SIZE),                                # Map character-ID context to hidden features
    nn.ReLU(),                                                           # Add non-linearity
    nn.Linear(HIDDEN_SIZE, vocab_size),                                  # Score every possible next character
)

# --- Train ---
start_time = time.time()                                                 # Begin fixed experiment timer

for epoch in range(EPOCHS):
    logits = model(inputs)                                               # Predict next-character scores
    loss = nn.functional.cross_entropy(logits, targets)                  # Compare predictions with true characters
    loss.backward()                                                      # Compute gradients for every parameter

    with torch.no_grad():                                                # Update weights without tracking gradients
        for p in model.parameters():
            p -= LR * p.grad                                             # Gradient-descent parameter update
            p.grad.zero_()                                               # Clear gradients before next epoch

    if epoch % LOG_EVERY == 0:
        acc = (logits.argmax(1) == targets).float().mean()               # Fraction of correct next-character predictions
        print(f"Epoch {epoch:5d} | Loss: {loss:.3f} | Acc: {acc:.1%}")  # Show learning progress

    if time.time() - start_time >= TRAIN_SECONDS:                        # Stop at the shared experiment budget
        break

# --- Final metrics ---
acc = (logits.argmax(1) == targets).float().mean()                       # Final training accuracy
print(f"FINAL | Loss: {loss:.6f} | Acc: {acc:.2%}")                      # Parsed by orchestrator

# --- Generate ---
context = inputs[0].tolist()                                              # First training context is generation seed

for _ in range(GEN_CHARS):
    inp = torch.tensor(context[-CONTEXT_SIZE:], dtype=torch.float).unsqueeze(0) # Convert latest context into one model input
    next_char_id = model(inp).argmax(1).item()                            # Select most likely next character
    context.append(next_char_id)                                          # Slide context forward with generated character

print(''.join(idx_to_char[i] for i in context[CONTEXT_SIZE:]))            # Convert generated IDs back to text

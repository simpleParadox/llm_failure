import torch
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer

# Load GPT-2 small model and tokenizer
model_name = 'gpt2'
model = GPT2Model.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to compute SVD (Singular Value Decomposition)
def compute_svd(matrix):
    # Convert the matrix to numpy array
    matrix_np = matrix.detach().cpu().numpy()
    
    # Compute the singular value decomposition (SVD)
    U, S, Vh = np.linalg.svd(matrix_np, full_matrices=False)
    
    return U, S, Vh

# Hook function to capture and perform SVD on activations during forward pass
def hook_fn(module, input, output, name):
    # Check if output has hidden states and extract them
    if hasattr(output, 'last_hidden_state'):
        activation = output.last_hidden_state  # Extract the hidden states
    else:
        activation = output  # Fallback in case the structure is different

    # Flatten the activation for SVD if it's not 2D
    import pdb; pdb.set_trace()
    activation_flat = activation.view(activation.shape[0], -1)
    
    U, S, Vh = compute_svd(activation_flat)
    
    print(f"\nProcessing activations at layer: {name}")
    print(f"Singular values: {S}")

# Register forward hooks for each layer in the model
hooks = []
for name, layer in model.named_modules():
    hooks.append(layer.register_forward_hook(lambda module, input, output, name=name: hook_fn(module, input, output, name)))

# Example of input text
input_text = "hello world"
input_ids = tokenizer(input_text, return_tensors='pt')['input_ids']

# Pass the input through the model to trigger the hooks
outputs = model(input_ids)

# Remove hooks after the forward pass
for hook in hooks:
    hook.remove()

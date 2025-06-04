import torch
import torch.nn as nn
import math

def train_loop(data_loader, optimizer, criterion_train, model, clip_value=5):
    """
    Training loop for one epoch with gradient clipping.
    This function handles the forward pass, loss computation, backpropagation and parameter updates.
    """
    model.train() 
    loss_array = []
    number_of_tokens = []

    for batch in data_loader:
        optimizer.zero_grad()

        # Forward pass through the model
        model_output = model(batch["source"])
        predictions_raw = model_output[0] if isinstance(model_output, tuple) else model_output

        # Safety check: ensure model output has correct dimensions (batch_size, vocab_size, sequence_length)
        if predictions_raw.dim() != 3:
            print(f"ERROR in train_loop: Model output 'predictions_raw' is not 3D! Shape: {predictions_raw.shape}. Skipping batch.")
            continue

        # Compute cross-entropy loss
        loss = criterion_train(predictions_raw, batch["target"])
        
        # Store weighted loss (by number of tokens) for proper averaging
        loss_array.append(loss.item() * batch["number_tokens"])
        number_of_tokens.append(batch["number_tokens"])

        # Backpropagation and gradient clipping to prevent exploding gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

    # Return average loss per token across the entire epoch
    if sum(number_of_tokens) == 0:
        return 0.0
    return sum(loss_array) / sum(number_of_tokens)

def eval_loop(data_loader, criterion_eval, model):
    """
    Evaluation loop that computes perplexity and average loss.
    Perplexity is the exponential of the average loss and measures how well the model predicts sequences.
    """
    model.eval() 
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch in data_loader:
            model_output = model(batch["source"])
            predictions_raw = model_output[0] if isinstance(model_output, tuple) else model_output

            # Safety check for tensor dimensions
            if predictions_raw.dim() != 3:
                print(f"ERROR in eval_loop: Model output 'predictions_raw' is not 3D! Shape: {predictions_raw.shape}. Skipping batch.")
                continue

            # Compute loss (using sum reduction for proper perplexity calculation)
            loss = criterion_eval(predictions_raw, batch["target"])
           
            loss_array.append(loss.item())
            number_of_tokens.append(batch["number_tokens"])

    if sum(number_of_tokens) == 0:
        return float('inf'), float('inf')

    # Calculate average loss per token
    avg_loss = sum(loss_array) / sum(number_of_tokens)
    
    # Convert to perplexity (lower is better)
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float('inf') 
    return ppl, avg_loss

def init_weights(module):
    """
    Custom weight initialization for different layer types.
    Proper initialization is crucial for LSTM training stability.
    """
    # Special initialization for recurrent layers (LSTM, GRU, RNN)
    if type(module) in [nn.GRU, nn.LSTM, nn.RNN]:
        for name, param in module.named_parameters():
            if 'weight_ih' in name:  # Input-to-hidden weights
                for idx in range(4):
                    mul = param.shape[0]//4
                    torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
            elif 'weight_hh' in name:  # Hidden-to-hidden weights  
                for idx in range(4):
                    mul = param.shape[0]//4
                    torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
            elif 'bias' in name:  # Bias terms
                param.data.fill_(0)
    else:
        # Standard initialization for linear layers
        if type(module) in [nn.Linear]:
            torch.nn.init.uniform_(module.weight, -0.01, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
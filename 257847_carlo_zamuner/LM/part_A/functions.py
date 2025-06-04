import torch
import torch.nn as nn
import math

def train_loop(data_loader, optimizer, criterion_train, model, clip_value=5):
    """
    Training function: trains the model for one epoch and returns average loss per token
    """
    model.train() 
    loss_array = []
    number_of_tokens = []

    for batch in data_loader:
        optimizer.zero_grad()

        # Forward pass through the model
        model_output = model(batch["source"])
        predictions_raw = model_output[0] if isinstance(model_output, tuple) else model_output

        # Check model output dimensions (batch_size, vocab_size, sequence_length)
        if predictions_raw.dim() != 3:
            print(f"ERROR in train_loop: Model output 'predictions_raw' is not 3D! Shape: {predictions_raw.shape}. Skipping batch.")
            continue

        # Calculate loss and accumulate weighted by number of tokens
        loss = criterion_train(predictions_raw, batch["target"])
        loss_array.append(loss.item() * batch["number_tokens"])
        number_of_tokens.append(batch["number_tokens"])

        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

    if sum(number_of_tokens) == 0:
        return 0.0
    return sum(loss_array) / sum(number_of_tokens)

def eval_loop(data_loader, criterion_eval, model):
    """
    Evaluation function: evaluates model without updating weights.
    Returns perplexity and average loss.
    """
    model.eval() 
    loss_array = []
    number_of_tokens = []

    with torch.no_grad(): 
        for batch in data_loader:
            model_output = model(batch["source"])
            predictions_raw = model_output[0] if isinstance(model_output, tuple) else model_output

            if predictions_raw.dim() != 3:
                print(f"ERROR in eval_loop: Model output 'predictions_raw' is not 3D! Shape: {predictions_raw.shape}. Skipping batch.")
                continue

            loss = criterion_eval(predictions_raw, batch["target"])
            loss_array.append(loss.item())
            number_of_tokens.append(batch["number_tokens"])

    if sum(number_of_tokens) == 0:
        return float('inf'), float('inf')

    # Calculate perplexity: exp(average_loss)
    avg_loss = sum(loss_array) / sum(number_of_tokens)
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float('inf') 
    return ppl, avg_loss

def init_weights(module):
    """
    Initialize model weights using best practices for RNN and Linear layers
    """
    # Special initialization for RNN layers (LSTM, GRU, RNN)
    if type(module) in [nn.GRU, nn.LSTM, nn.RNN]:
        for name, param in module.named_parameters():
            if 'weight_ih' in name:  # Input-to-hidden weights: Xavier uniform
                for idx in range(4):
                    mul = param.shape[0]//4
                    torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
            elif 'weight_hh' in name:  # Hidden-to-hidden weights: Orthogonal
                for idx in range(4):
                    mul = param.shape[0]//4
                    torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
            elif 'bias' in name:
                param.data.fill_(0)
    else:
        # Linear layers: small uniform initialization
        if type(module) in [nn.Linear]:
            torch.nn.init.uniform_(module.weight, -0.01, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
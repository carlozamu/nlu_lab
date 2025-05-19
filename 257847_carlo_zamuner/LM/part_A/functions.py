
import torch
import torch.nn as nn
import math

# ==============================================================================
# --- TRAINING LOOP FUNCTION ---
# ==============================================================================
def train_loop(data_loader, optimizer, criterion_train, model, clip_value=5):
    """
    Executes one epoch of training.

    Args:
        data_loader: DataLoader for the training data.
        optimizer: The optimizer instance.
        criterion_train: The loss function for training (e.g., CrossEntropyLoss).
        model: The neural network model.
        clip_value (float): Max norm for gradient clipping.
    Returns:
        float: Average loss per token for the epoch.
    """
    model.train() # Set model to training mode
    total_loss_weighted = 0
    total_tokens_processed = 0

    for batch in data_loader:
        optimizer.zero_grad()

        # Forward pass
        # Model output might be a tuple (predictions, hidden_state) or just predictions
        model_output = model(batch["source"])
        predictions_raw = model_output[0] if isinstance(model_output, tuple) else model_output

        # Reshape for CrossEntropyLoss:
        # Input: (N, C, ...) Target: (N, ...)
        # predictions_raw is expected to be [Batch, VocabSize, SeqLen]
        # target is expected to be [Batch, SeqLen]
        if predictions_raw.dim() != 3:
            # This check is CRITICAL. If output is not 3D, permute will fail.
            print(f"ERROR in train_loop: Model output 'predictions_raw' is not 3D! Shape: {predictions_raw.shape}. Skipping batch.")
            continue

        vocab_size = predictions_raw.size(1)
        # Permute to [Batch, SeqLen, VocabSize] then reshape to [Batch*SeqLen, VocabSize]
        predictions_flat = predictions_raw.permute(0, 2, 1).reshape(-1, vocab_size)
        # Reshape target to [Batch*SeqLen]
        target_flat = batch["target"].reshape(-1)

        loss = criterion_train(predictions_flat, target_flat)

        # Accumulate loss, weighted by the number of actual (non-padding) tokens.
        # This correctly calculates average loss if criterion_train.reduction is 'mean'.
        # 'number_tokens' should come from collate_fn and represent actual tokens in the batch.
        if criterion_train.reduction == 'mean':
             current_batch_loss_sum = loss.item() * batch["number_tokens"]
        elif criterion_train.reduction == 'sum':
             current_batch_loss_sum = loss.item()
        else: # 'none' or other cases - this is an approximation
             print(f"Warning: criterion_train.reduction is '{criterion_train.reduction}'. Loss averaging might be approximate.")
             current_batch_loss_sum = loss.item() * batch["number_tokens"] / target_flat.size(0) if target_flat.size(0) > 0 else 0


        total_loss_weighted += current_batch_loss_sum
        total_tokens_processed += batch["number_tokens"]

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

    if total_tokens_processed == 0:
        return 0.0
    return total_loss_weighted / total_tokens_processed

# ==============================================================================
# --- EVALUATION LOOP FUNCTION ---
# ==============================================================================
def eval_loop(data_loader, criterion_eval, model):
    """
    Evaluates the model on a dataset.

    Args:
        data_loader: DataLoader for the evaluation data.
        criterion_eval: The loss function for evaluation (e.g., CrossEntropyLoss with reduction='sum').
        model: The neural network model.
    Returns:
        tuple: (Perplexity, Average Loss per token).
    """
    model.eval() # Set model to evaluation mode
    total_loss_sum = 0
    total_tokens_processed = 0

    with torch.no_grad(): # Disable gradient calculations
        for batch in data_loader:
            # Forward pass
            model_output = model(batch["source"])
            predictions_raw = model_output[0] if isinstance(model_output, tuple) else model_output

            if predictions_raw.dim() != 3:
                print(f"ERROR in eval_loop: Model output 'predictions_raw' is not 3D! Shape: {predictions_raw.shape}. Skipping batch.")
                continue

            # Reshape for CrossEntropyLoss
            vocab_size = predictions_raw.size(1)
            predictions_flat = predictions_raw.permute(0, 2, 1).reshape(-1, vocab_size)
            target_flat = batch["target"].reshape(-1)

            # Calculate loss (criterion_eval should have reduction='sum' for PPL)
            loss = criterion_eval(predictions_flat, target_flat)
            total_loss_sum += loss.item() # Accumulate sum of losses
            total_tokens_processed += batch["number_tokens"] # Accumulate actual tokens

    if total_tokens_processed == 0:
        return float('inf'), float('inf')

    avg_loss = total_loss_sum / total_tokens_processed
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float('inf') # Handle potential math overflow for very high losses
    return ppl, avg_loss

# ==============================================================================
# --- WEIGHT INITIALIZATION FUNCTION ---
# ==============================================================================
def init_weights(module):
    """
    Initializes weights for different layer types.
    To be used with model.apply(init_weights).

    Args:
        module (nn.Module): A module (or model) whose submodules will be initialized.
    """
    # Specific initialization ranges
    EMB_INIT_RANGE = 0.1  # For Embedding layers
    LINEAR_INIT_RANGE = 0.01 # For Linear layers

    classname = module.__class__.__name__
    if classname.find('Embedding') != -1:
        nn.init.uniform_(module.weight, -EMB_INIT_RANGE, EMB_INIT_RANGE)
        if module.padding_idx is not None:
            with torch.no_grad(): # Ensure padding index is zeroed out
                module.weight[module.padding_idx].fill_(0)
    elif classname.find('LSTM') != -1:
        # Common initialization for LSTMs
        for name, param in module.named_parameters():
            if 'weight_ih' in name: # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name: # Hidden-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name: # Biases
                param.data.fill_(0)
                # Optionally, initialize forget gate bias to 1 for better initial learning
                # n = param.size(0)
                # start, end = n // 4, n // 2
                # param.data[start:end].fill_(1.)
    elif classname.find('Linear') != -1:
        # Uniform initialization for Linear layers
        nn.init.uniform_(module.weight, -LINEAR_INIT_RANGE, LINEAR_INIT_RANGE)
        if module.bias is not None:
            nn.init.uniform_(module.bias, -LINEAR_INIT_RANGE, LINEAR_INIT_RANGE)

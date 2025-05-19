
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

        try:
            # Forward pass
            model_output = model(batch["source"])
            # Handle models that might return (output, hidden_state) tuple
            predictions_raw = model_output[0] if isinstance(model_output, tuple) else model_output

            # CRITICAL: Ensure model output is 3D before permute/reshape
            if predictions_raw.dim() != 3:
                print(f"ERROR in train_loop: Model output 'predictions_raw' is not 3D! "
                      f"Shape: {predictions_raw.shape}. Skipping batch.")
                continue

            # Reshape for CrossEntropyLoss
            # Input to CrossEntropyLoss: (N, C, ...) where N is batch_size, C is num_classes
            # predictions_raw is expected to be [Batch, VocabSize, SeqLen]
            # target is expected to be [Batch, SeqLen]
            vocab_size = predictions_raw.size(1)
            # Permute to [Batch, SeqLen, VocabSize] then reshape to [Batch*SeqLen, VocabSize]
            predictions_flat = predictions_raw.permute(0, 2, 1).reshape(-1, vocab_size)
            # Reshape target to [Batch*SeqLen]
            target_flat = batch["target"].reshape(-1)

            loss = criterion_train(predictions_flat, target_flat)

            # Accumulate loss, correctly weighted by actual (non-padding) tokens.
            # 'number_tokens' from collate_fn is sum of actual lengths in batch.
            if criterion_train.reduction == 'mean':
                 # If loss is already mean, multiply by actual tokens to get sum for this batch
                 current_batch_loss_sum = loss.item() * batch["number_tokens"]
            elif criterion_train.reduction == 'sum':
                 current_batch_loss_sum = loss.item()
            else: # 'none' or other cases
                 # This case is an approximation if reduction is 'none'.
                 # It assumes loss.item() is per-element and tries to scale it.
                 print(f"Warning: criterion_train.reduction is '{criterion_train.reduction}'. "
                       "Loss averaging per token might be approximate if not 'mean'.")
                 # If loss is per-element, its sum would be loss.sum().item().
                 # For now, approximate based on first element if loss is per-element.
                 # A better way if 'none' is used, would be to sum the loss tensor then .item().
                 current_batch_loss_sum = loss.item() * batch["number_tokens"] / target_flat.size(0) if target_flat.size(0) > 0 else 0


            total_loss_weighted += current_batch_loss_sum
            total_tokens_processed += batch["number_tokens"]

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

        except Exception as e:
            print(f"\nError during training loop for a batch: {e}")
            # Optionally, print more details about the batch or error
            continue # Attempt to continue with the next batch

    if total_tokens_processed == 0:
        print("Warning: No tokens processed in train_loop.")
        return 0.0
    return total_loss_weighted / total_tokens_processed

# ==============================================================================
# --- EVALUATION LOOP FUNCTION ---
# ==============================================================================
def eval_loop(data_loader, criterion_eval, model):
    """
    Evaluates the model on a dataset (e.g., development or test set).

    Args:
        data_loader: DataLoader for the evaluation data.
        criterion_eval: Loss function for evaluation (expected to have reduction='sum' for PPL).
        model: The neural network model.
    Returns:
        tuple: (Perplexity, Average Loss per token).
    """
    model.eval() # Set model to evaluation mode
    total_loss_sum = 0
    total_tokens_processed = 0

    with torch.no_grad(): # Disable gradient calculations for evaluation
        for batch in data_loader:
            try:
                # Forward pass
                model_output = model(batch["source"])
                predictions_raw = model_output[0] if isinstance(model_output, tuple) else model_output

                if predictions_raw.dim() != 3:
                    print(f"ERROR in eval_loop: Model output 'predictions_raw' is not 3D! "
                          f"Shape: {predictions_raw.shape}. Skipping batch.")
                    continue

                # Reshape for CrossEntropyLoss
                vocab_size = predictions_raw.size(1)
                predictions_flat = predictions_raw.permute(0, 2, 1).reshape(-1, vocab_size)
                target_flat = batch["target"].reshape(-1)

                # Calculate loss (criterion_eval should have reduction='sum')
                loss = criterion_eval(predictions_flat, target_flat)
                total_loss_sum += loss.item() # Accumulate sum of losses
                total_tokens_processed += batch["number_tokens"] # Accumulate actual tokens
            except Exception as e:
                print(f"\nError during evaluation loop for a batch: {e}")
                continue

    if total_tokens_processed == 0:
        print("Warning: No tokens processed in eval_loop.")
        return float('inf'), float('inf')

    avg_loss = total_loss_sum / total_tokens_processed
    try:
        # Perplexity is e^(average cross-entropy loss)
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float('inf') # Handle potential math overflow for very high losses
    return ppl, avg_loss

# ==============================================================================
# --- WEIGHT INITIALIZATION FUNCTION ---
# ==============================================================================
def init_weights(module: nn.Module):
    """
    Initializes weights for specific layer types within a model.
    Designed to be used with `model.apply(init_weights)`.

    Args:
        module (nn.Module): The module (or submodule) to initialize.
    """
    EMB_INIT_RANGE = 0.1  # Uniform range for Embedding layers
    LINEAR_INIT_RANGE = 0.01 # Uniform range for Linear layers

    classname = module.__class__.__name__
    if isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -EMB_INIT_RANGE, EMB_INIT_RANGE)
        if module.padding_idx is not None:
            with torch.no_grad(): # Ensure padding index embedding is zero
                module.weight[module.padding_idx].fill_(0)
    elif isinstance(module, nn.LSTM):
        # Common initialization scheme for LSTMs
        for name, param in module.named_parameters():
            if 'weight_ih' in name: # Input-to-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name: # Hidden-to-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name: # Biases
                param.data.fill_(0)
                # Optionally, initialize forget gate bias to 1 (common practice)
                # n = param.size(0)
                # forget_gate_start_idx = n // 4
                # forget_gate_end_idx = n // 2
                # param.data[forget_gate_start_idx:forget_gate_end_idx].fill_(1.)
    elif isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -LINEAR_INIT_RANGE, LINEAR_INIT_RANGE)
        if module.bias is not None:
            nn.init.uniform_(module.bias, -LINEAR_INIT_RANGE, LINEAR_INIT_RANGE)
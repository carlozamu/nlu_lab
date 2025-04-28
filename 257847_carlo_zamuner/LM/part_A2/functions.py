import torch
import torch.nn as nn
import math

# Funzione per l'epoca di addestramento (CON RESHAPE NECESSARIO)
def train_loop(data_loader, optimizer, criterion_train, model, clip_value=5): # Usa clip_value
    """Esegue un'epoca di addestramento sul modello."""
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in data_loader:
        optimizer.zero_grad()
        predictions_raw, *_ = model(batch["source"]) # Gestisce output tupla/tensore
        vocab_size = predictions_raw.size(1)
        # --- !!! RESHAPE NECESSARIO QUI !!! ---
        predictions_flat = predictions_raw.permute(0, 2, 1).reshape(-1, vocab_size)
        target_flat = batch["target"].reshape(-1)
        # ------------------------------------
        loss = criterion_train(predictions_flat, target_flat) # Usa reshape

        if criterion_train.reduction == 'mean':
             weighted_loss = loss.item() * batch["number_tokens"]
        else:
             print("Warning: criterion_train reduction non 'mean'. Averaging approssimativo.")
             weighted_loss = loss.item()

        total_loss += weighted_loss
        total_tokens += batch["number_tokens"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) # Usa clip_value
        optimizer.step()

    if total_tokens == 0: return 0.0
    return total_loss / total_tokens

# Funzione per valutazione (CON RESHAPE NECESSARIO)
def eval_loop(data_loader, criterion_eval, model):
    """Valuta il modello su un dataset (PPL e loss)."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            predictions_raw, *_ = model(batch["source"])
            vocab_size = predictions_raw.size(1)
            # --- !!! RESHAPE NECESSARIO QUI !!! ---
            predictions_flat = predictions_raw.permute(0, 2, 1).reshape(-1, vocab_size)
            target_flat = batch["target"].reshape(-1)
            # ------------------------------------
            loss = criterion_eval(predictions_flat, target_flat) # Usa reshape
            total_loss += loss.item()
            total_tokens += batch["number_tokens"]

    if total_tokens == 0: return float('inf'), float('inf')
    avg_loss = total_loss / total_tokens
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float('inf')
    return ppl, avg_loss

# Funzione init_weights (originale o quella pulita, assicurati sia quella voluta)
def init_weights(m):
    """Inizializza pesi per modulo: Emb Unif[-0.1,0.1], LSTM Xavier/Ortho, Linear Unif."""
    EMB_INIT_RANGE = 0.1
    LINEAR_INIT_RANGE = 0.01
    classname = m.__class__.__name__
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -EMB_INIT_RANGE, EMB_INIT_RANGE)
        if m.padding_idx is not None:
            with torch.no_grad(): m.weight[m.padding_idx].fill_(0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name: nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name: nn.init.orthogonal_(param.data)
            elif 'bias' in name: param.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -LINEAR_INIT_RANGE, LINEAR_INIT_RANGE)
        if m.bias is not None: nn.init.uniform_(m.bias, -LINEAR_INIT_RANGE, LINEAR_INIT_RANGE)
import torch
import torch.nn as nn
import math


# Esegue un'epoca di addestramento
def train_loop(data_loader, optimizer, criterion_train, model, clip_value=5):
    model.train()
    total_loss = 0
    total_tokens = 0

    for batch in data_loader:
        optimizer.zero_grad()
        # Gestisce output modello (tupla o tensore)
        model_output = model(batch["source"])
        predictions_raw = (
            model_output[0] if isinstance(model_output, tuple) else model_output
        )

        if predictions_raw.dim() != 3:
            print(
                f"ERRORE train_loop: Output modello non 3D! Shape: {predictions_raw.shape}. Salto batch."
            )
            continue

        # Reshape per CrossEntropyLoss
        vocab_size = predictions_raw.size(1)
        predictions_flat = predictions_raw.permute(0, 2, 1).reshape(-1, vocab_size)
        target_flat = batch["target"].reshape(-1)

        try:
            loss = criterion_train(predictions_flat, target_flat)

            # Accumula loss ponderata
            if criterion_train.reduction == "mean":
                weighted_loss = loss.item() * batch["number_tokens"]
            else:  # Gestione approssimata per 'sum' o 'none'
                weighted_loss = (
                    loss.item() * batch["number_tokens"] / target_flat.size(0)
                    if target_flat.size(0) > 0
                    else 0
                )

            total_loss += weighted_loss
            total_tokens += batch["number_tokens"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
        except Exception as e:
            print(f"\nErrore calcolo loss/backprop (Batch): {e}")
            continue

    if total_tokens == 0:
        return 0.0
    return total_loss / total_tokens


# Esegue la valutazione
def eval_loop(data_loader, criterion_eval, model):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            try:
                model_output = model(batch["source"])
                predictions_raw = (
                    model_output[0] if isinstance(model_output, tuple) else model_output
                )

                if predictions_raw.dim() != 3:
                    print(
                        f"ERRORE eval_loop: Output modello non 3D! Shape: {predictions_raw.shape}. Salto batch."
                    )
                    continue

                # Reshape per CrossEntropyLoss
                vocab_size = predictions_raw.size(1)
                predictions_flat = predictions_raw.permute(0, 2, 1).reshape(
                    -1, vocab_size
                )
                target_flat = batch["target"].reshape(-1)

                loss = criterion_eval(predictions_flat, target_flat)  # reduction='sum'
                total_loss += loss.item()
                total_tokens += batch["number_tokens"]
            except Exception as e:
                print(f"\nErrore durante valutazione (Batch): {e}")
                continue

    if total_tokens == 0:
        return float("inf"), float("inf")
    avg_loss = total_loss / total_tokens
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")
    return ppl, avg_loss


# Inizializzazione pesi
def init_weights(m):
    EMB_INIT_RANGE = 0.1
    LINEAR_INIT_RANGE = 0.01
    classname = m.__class__.__name__
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -EMB_INIT_RANGE, EMB_INIT_RANGE)
        if m.padding_idx is not None:
            with torch.no_grad():
                m.weight[m.padding_idx].fill_(0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -LINEAR_INIT_RANGE, LINEAR_INIT_RANGE)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -LINEAR_INIT_RANGE, LINEAR_INIT_RANGE)

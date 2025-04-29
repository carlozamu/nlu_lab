import torch
import torch.nn as nn
import math


# Funzione per l'epoca di addestramento (CON RESHAPE)
def train_loop(data_loader, optimizer, criterion_train, model, clip_value=5):
    """Esegue un'epoca di addestramento sul modello."""
    model.train()  # Imposta il modello in modalità training
    total_loss = 0
    total_tokens = 0

    for batch in data_loader:
        optimizer.zero_grad()

        # --- Gestione Output Modello e Reshape ---
        try:
            # Chiama il modello
            model_output = model(batch["source"])

            # Estrai le predizioni (il primo elemento se è una tupla, altrimenti l'output stesso)
            predictions_raw = (
                model_output[0] if isinstance(model_output, tuple) else model_output
            )

            # Verifica dimensioni PRIMA del permute (per debug se necessario)
            # print(f"Debug train_loop - predictions_raw shape: {predictions_raw.shape}")
            if predictions_raw.dim() != 3:
                print(
                    f"ERRORE in train_loop: Output del modello non è 3D! Shape: {predictions_raw.shape}"
                )
                # Potresti voler sollevare un errore qui o gestire diversamente
                continue  # Salta questo batch problematico

            # Reshape per CrossEntropyLoss
            vocab_size = predictions_raw.size(1)  # Dimensione C (classi/vocab)
            # Da [B, C, S] a [B * S, C]
            predictions_flat = predictions_raw.permute(0, 2, 1).reshape(-1, vocab_size)
            # Target da [B, S] a [B * S]
            target_flat = batch["target"].reshape(-1)

        except Exception as e:
            print(
                f"Errore durante processamento output modello o reshape in train_loop: {e}"
            )
            print(f"   Input shape: {batch['source'].shape}")
            if "model_output" in locals():
                print(f"   Output modello (tipo): {type(model_output)}")
            if "predictions_raw" in locals():
                print(f"   Predictions_raw shape: {predictions_raw.shape}")
            continue  # Salta il batch

        # --- Calcolo Loss e Aggiornamento ---
        try:
            loss = criterion_train(predictions_flat, target_flat)

            # Accumula loss (considerando 'number_tokens' reali)
            # Assumendo criterion_train.reduction == 'mean'
            if criterion_train.reduction == "mean":
                weighted_loss = loss.item() * batch["number_tokens"]
            else:  # Gestione approssimata per 'sum' o 'none'
                print(
                    f"Warning: criterion_train reduction ({criterion_train.reduction}) non è 'mean'. Averaging approssimativo."
                )
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
            print(f"Errore durante calcolo loss o backprop in train_loop: {e}")
            continue  # Salta il batch

    # Ritorna loss media per token reale
    if total_tokens == 0:
        return 0.0
    return total_loss / total_tokens


# Funzione per valutazione (CON RESHAPE)
def eval_loop(data_loader, criterion_eval, model):
    """Valuta il modello su un dataset (PPL e loss)."""
    model.eval()  # Modalità valutazione
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():  # Disabilita gradienti
        for batch in data_loader:
            # --- Gestione Output Modello e Reshape ---
            try:
                model_output = model(batch["source"])
                predictions_raw = (
                    model_output[0] if isinstance(model_output, tuple) else model_output
                )

                # Verifica dimensioni PRIMA del permute
                # print(f"Debug eval_loop - predictions_raw shape: {predictions_raw.shape}")
                if predictions_raw.dim() != 3:
                    print(
                        f"ERRORE in eval_loop: Output del modello non è 3D! Shape: {predictions_raw.shape}"
                    )
                    continue

                vocab_size = predictions_raw.size(1)
                predictions_flat = predictions_raw.permute(0, 2, 1).reshape(
                    -1, vocab_size
                )
                target_flat = batch["target"].reshape(-1)

            except Exception as e:
                print(
                    f"Errore durante processamento output modello o reshape in eval_loop: {e}"
                )
                print(f"   Input shape: {batch['source'].shape}")
                if "model_output" in locals():
                    print(f"   Output modello (tipo): {type(model_output)}")
                if "predictions_raw" in locals():
                    print(f"   Predictions_raw shape: {predictions_raw.shape}")
                continue

            # --- Calcolo Loss ---
            try:
                # Usa criterion con reduction='sum' per calcolo PPL
                loss = criterion_eval(predictions_flat, target_flat)
                total_loss += loss.item()  # loss.item() è già la somma
                total_tokens += batch["number_tokens"]  # Token reali
            except Exception as e:
                print(f"Errore durante calcolo loss in eval_loop: {e}")
                continue

    # Calcola Perplexity e Loss media
    if total_tokens == 0:
        return float("inf"), float("inf")
    avg_loss = total_loss / total_tokens
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")
    return ppl, avg_loss


# Funzione di inizializzazione pesi (per usare con model.apply)
def init_weights(m):
    """Inizializza pesi per modulo: Emb Unif[-0.1,0.1], LSTM Xavier/Ortho, Linear Unif."""
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

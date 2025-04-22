# Esegue la configurazione migliore (B3 Fine-Tuned) e salva il modello.
# Modello: LSTM+WT+VarDrop, Optim: NT-AvSGD(SGD)
# Params: LR=3.75, BS=16, P=7, T=3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import copy
from functools import partial
import os

# Rimosso: pandas, matplotlib, seaborn, tabulate, time, warnings
from tqdm import tqdm

# Importa funzioni e classi custom
try:
    from functions import train_loop, eval_loop, init_weights
    from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn

    # Importa il modello corretto per B3/FT
    from model import LM_LSTM_VarDropWT
except ImportError as e:
    print(f"Errore importazione moduli: {e}. Assicurati che siano accessibili.")
    exit()

if __name__ == "__main__":
    # --- Impostazioni ---
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    experiment_step = "Part1B_FineTuned_Best"
    model_save_dir = "bin"  # Cartella per salvataggio

    # --- Parametri Fissi Configurazione Migliore (B3 Fine-Tuned) ---
    lr = 3.75
    batch_size = 16
    patience_setting = 7
    trigger_setting = 3
    hid = 300
    emb = 300  # == hid per Weight Tying
    clip = 5
    n_epochs = 100
    dropoute_rate = 0.1
    dropouto_rate = 0.4

    # --- Dati ---
    print("\nLoading data & vocabulary...")
    try:
        train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
        dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
        test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")
        lang = Lang(train_raw, ["<pad>", "<eos>"])
        vocab_len = len(lang.word2id)
        pad_token_id = lang.word2id["<pad>"]
        print("Data loaded.")
    except FileNotFoundError as e:
        print(f"Errore: Dataset non trovato: {e}")
        exit()

    print("Creating Datasets...")
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    print("Datasets created.")

    # --- Esecuzione ---
    print(f"\n--- Running Configuration: {experiment_step} ---")
    print(
        f"   LR={lr}, BS={batch_size}, P={patience_setting}, T={trigger_setting}, Optim=NT-AvSGD(SGD)"
    )

    # --- DataLoader ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id, device=DEVICE),
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size * 2,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id, device=DEVICE),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id, device=DEVICE),
    )

    # --- Modello, Optimizer, Loss ---
    print("Initializing Model, Optimizer, Loss...")
    model = LM_LSTM_VarDropWT(
        emb_size=emb,
        hidden_size=hid,
        output_size=vocab_len,
        pad_index=pad_token_id,
        n_layers=1,
        dropoute=dropoute_rate,
        dropouto=dropouto_rate,
    ).to(DEVICE)
    model.apply(init_weights)  # Inizializzazione

    # Optimizer base è SGD per NT-AvSGD
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="sum")

    # --- Training Loop con NT-AvSGD ---
    print("Starting Training...")
    patience = patience_setting
    avg_trigger_countdown = trigger_setting
    is_averaging = False
    avg_model = None
    avg_step_count = 0
    trigger_epoch = -1
    best_dev_ppl = math.inf
    best_model_state_dict = None
    best_ppl_for_avg_trigger = math.inf
    epochs_trained = 0

    pbar = tqdm(range(1, n_epochs + 1), desc="Epoch", leave=True)
    for epoch in pbar:
        epochs_trained = epoch
        model.train()
        # Usa clip_value come definito in functions.py
        train_loss = train_loop(
            train_loader, optimizer, criterion_train, model, clip_value=clip
        )

        # Averaging (se attivo)
        if is_averaging:
            avg_step_count += 1
            with torch.no_grad():
                for avg_param, param in zip(avg_model.parameters(), model.parameters()):
                    avg_param.data.add_(
                        param.data.to(avg_param.device) - avg_param.data,
                        alpha=1.0 / avg_step_count,
                    )

        # Valutazione (usa modello mediato se attivo)
        eval_model_ref = avg_model if is_averaging else model
        if is_averaging:
            eval_model_ref.to(DEVICE)
        eval_model_ref.eval()
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, eval_model_ref)
        if is_averaging:
            eval_model_ref.to("cpu")

        pbar.set_description(f"Epoch {epoch}/{n_epochs} -> Dev PPL: {ppl_dev:.3f}")

        # Trigger NT-AvSGD
        if not is_averaging:
            if ppl_dev < best_ppl_for_avg_trigger:
                best_ppl_for_avg_trigger = ppl_dev
                avg_trigger_countdown = trigger_setting
            else:
                avg_trigger_countdown -= 1
                if avg_trigger_countdown <= 0:
                    print(
                        f"\n   NT-AvSGD Triggered at Epoch {epoch}! Starting averaging."
                    )
                    is_averaging = True
                    trigger_epoch = epoch
                    avg_model = copy.deepcopy(model).to("cpu")
                    avg_step_count = 1

        # Early Stopping & Miglior Modello
        if ppl_dev < best_dev_ppl:
            best_dev_ppl = ppl_dev
            model_to_save = avg_model if is_averaging else model
            best_model_state_dict = copy.deepcopy(model_to_save.state_dict())
            patience = patience_setting
        else:
            patience -= 1
            if patience <= 0:
                print(f"\nEarly stopping at epoch {epoch}.")
                break
    pbar.close()
    print("--- Training Finished ---")

    # --- Valutazione Finale ---
    final_test_ppl = math.inf
    trigger_info = f"Yes (Epoch {trigger_epoch})" if trigger_epoch != -1 else "No"

    if best_model_state_dict is not None:
        print("\nEvaluating best model on Test Set...")
        # Ricarica il modello corretto
        final_model_to_eval = LM_LSTM_VarDropWT(
            emb_size=emb,
            hidden_size=hid,
            output_size=vocab_len,
            pad_index=pad_token_id,
            n_layers=1,
            dropoute=dropoute_rate,
            dropouto=dropouto_rate,
        ).to(DEVICE)
        final_model_to_eval.load_state_dict(best_model_state_dict)
        final_model_to_eval.eval()
        with torch.no_grad():
            final_test_ppl, _ = eval_loop(
                test_loader, criterion_eval, final_model_to_eval
            )

        print(f"\n--- Final Results ---")
        print(f"   Best Development PPL: {best_dev_ppl:.4f}")
        print(f"   Final Test PPL: {final_test_ppl:.4f}")
        print(f"   Averaging Triggered: {trigger_info}")

        # --- Salvataggio Modello Migliore ---
        print("\nSaving best model state dict...")
        os.makedirs(model_save_dir, exist_ok=True)
        model_filename = f"{experiment_step}_lr{lr}_bs{batch_size}_p{patience_setting}_t{trigger_setting}.pt"
        model_save_path = os.path.join(model_save_dir, model_filename)
        try:
            torch.save(best_model_state_dict, model_save_path)
            print(f"   Model saved to: {model_save_path}")
        except Exception as e:
            print(f"   Error saving model: {e}")
    else:
        print("\nNo best model state saved.")

    print(f"\n--- Script {experiment_step} Finished ---")

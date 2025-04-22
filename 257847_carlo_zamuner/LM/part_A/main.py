# Esegue la configurazione A3 migliore (LSTM+Dropout+AdamW) e salva il modello.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import copy
from functools import partial
import os

from tqdm import tqdm

from functions import train_loop, eval_loop, init_weights
from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
from model import LM_LSTM_With_Dropout  # Modello per A3

if __name__ == "__main__":
    # --- Impostazioni ---
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    experiment_step = "Part1A_Step3_Best"
    model_save_dir = "bin"  # Cartella per salvataggio

    # --- Parametri Fissi Configurazione Migliore A3 ---
    lr = 0.001
    batch_size = 16
    hid = 300
    emb = 200
    clip = 5
    n_epochs = 100
    patience_setting = 3
    emb_dropout_rate = 0.1
    fc_dropout_rate = 0.1

    # --- Dati ---
    print("\nLoading data & vocabulary...")
    try:
        train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
        dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
        test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")
        lang = Lang(train_raw, ["<pad>", "<eos>"])
        vocab_len = len(lang.word2id)
        pad_token_id = lang.word2id["<pad>"]  # per collate_fn
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
    print(f"   LR={lr}, BS={batch_size}, Optim=AdamW")

    # --- DataLoader ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(
            collate_fn, pad_token=pad_token_id, DEVICE=DEVICE
        ),  # Usa 'pad_token' come nell'originale utils.py
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size * 2,
        collate_fn=partial(
            collate_fn, pad_token=pad_token_id, DEVICE=DEVICE
        ),  # Usa 'pad_token'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        collate_fn=partial(
            collate_fn, pad_token=pad_token_id, DEVICE=DEVICE
        ),  # Usa 'pad_token'
    )

    # --- Modello, Optimizer, Loss ---
    print("Initializing Model, Optimizer, Loss...")
    model = LM_LSTM_With_Dropout(
        emb_size=emb,
        hidden_size=hid,
        output_size=vocab_len,
        pad_index=pad_token_id,  # Passa l'ID di padding
        emb_dropout=emb_dropout_rate,
        fc_dropout=fc_dropout_rate,
    ).to(DEVICE)
    model.apply(init_weights)  # Applica init originale

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="sum")

    # --- Training Loop ---
    print("Starting Training...")
    patience = patience_setting
    best_dev_ppl = math.inf
    best_model_state_dict = None  # Salva lo state_dict, non il modello intero

    pbar = tqdm(range(1, n_epochs + 1), desc="Epoch", leave=True)
    for epoch in pbar:
        # --- Training Step ---
        # Usa 'clip' come nell'originale functions.py
        train_loss = train_loop(
            train_loader, optimizer, criterion_train, model, clip=clip
        )

        # --- Evaluation Step ---
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        pbar.set_description(f"Epoch {epoch}/{n_epochs} -> Dev PPL: {ppl_dev:.3f}")

        # --- Early Stopping & Miglior Modello ---
        if ppl_dev < best_dev_ppl:
            best_dev_ppl = ppl_dev
            best_model_state_dict = copy.deepcopy(
                model.state_dict()
            )  # Salva state_dict
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
    if best_model_state_dict is not None:
        print("\nEvaluating best model on Test Set...")
        model.load_state_dict(best_model_state_dict)  # Carica stato migliore
        model.eval()
        with torch.no_grad():
            final_test_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print(f"   Best Dev PPL achieved: {best_dev_ppl:.4f}")
        print(f"   Final Test PPL: {final_test_ppl:.4f}")

        # --- Salvataggio Modello Migliore ---
        print("\nSaving best model state dict...")
        os.makedirs(model_save_dir, exist_ok=True)
        model_filename = f"{experiment_step}_lr{lr}_bs{batch_size}.pt"
        model_save_path = os.path.join(model_save_dir, model_filename)
        try:
            torch.save(best_model_state_dict, model_save_path)
            print(f"   Model saved to: {model_save_path}")
        except Exception as e:
            print(f"   Error saving model: {e}")
    else:
        print("\nNo best model state saved.")

    print(f"\n--- Script {experiment_step} Finished ---")

# Esegue la configurazione migliore (A1: LSTM Base + SGD) e salva il modello.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import copy
from functools import partial
import os
from tqdm import tqdm

# Import Moduli Custom
try:
    # Usa le funzioni standard, ma il modello LM_LSTM_Base
    from functions import train_loop, eval_loop, init_weights
    from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
    # --- MODIFICA PRINCIPALE: Importa il modello BASE ---
    from model import LM_LSTM_Base
except ImportError as e:
    print(f"Errore importazione moduli: {e}. Assicurati che siano accessibili.")
    exit()

if __name__ == "__main__":
    # --- Impostazioni ---
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    # --- Nome per identificare lo step ---
    experiment_step = "Part1A_Step1_Best"
    model_save_dir = "bin"

    # --- Parametri Fissi Configurazione Migliore A1 ---
    lr = 1.5 # LR ottimale per A1
    batch_size = 16 # BS ottimale per A1
    hid = 300
    emb = 200 # Embedding size originale per A1
    clip = 5
    n_epochs = 100
    patience_setting = 3
    # Non ci sono dropout rate da definire per A1

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
    # --- MODIFICA: Indicare SGD ---
    print(f"   LR={lr}, BS={batch_size}, Optim=SGD")

    # --- DataLoader ---
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=partial(collate_fn, pad_token=pad_token_id, DEVICE=DEVICE), # Usa nome originale 'pad_token'
        drop_last=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size * 2,
        collate_fn=partial(collate_fn, pad_token=pad_token_id, DEVICE=DEVICE)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2,
        collate_fn=partial(collate_fn, pad_token=pad_token_id, DEVICE=DEVICE)
    )

    # --- Modello, Optimizer, Loss ---
    print("Initializing Model, Optimizer, Loss...")
    # --- MODIFICA: Usa il modello BASE ---
    model = LM_LSTM_Base(
        emb_size=emb, hidden_size=hid, output_size=vocab_len,
        pad_index=pad_token_id
        # Non ha argomenti dropout
    ).to(DEVICE)
    model.apply(init_weights)

    # --- MODIFICA: Usa l'optimizer SGD ---
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="sum")

    # --- Training Loop ---
    print("Starting Training...")
    patience = patience_setting
    best_dev_ppl = math.inf
    best_model_state_dict = None
    epochs_trained = 0

    pbar = tqdm(range(1, n_epochs + 1), desc="Epoch", leave=True)
    for epoch in pbar:
        epochs_trained = epoch
        model.train()
        # --- MODIFICA: Passa 'clip' come nell'originale functions.py ---
        train_loss = train_loop(
            train_loader, optimizer, criterion_train, model, clip=clip
        )

        model.eval()
        # --- ATTENZIONE: Assicurarsi che eval_loop gestisca correttamente il reshape ---
        # --- La versione originale di functions.py NON lo fa. Assicurati di usare ---
        # --- una versione di functions.py che includa il reshape per la loss. ---
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        pbar.set_description(f"Epoch {epoch}/{n_epochs} -> Dev PPL: {ppl_dev:.3f}")

        if ppl_dev < best_dev_ppl:
            best_dev_ppl = ppl_dev
            best_model_state_dict = copy.deepcopy(model.state_dict())
            patience = patience_setting
        else:
            patience -= 1
            if patience <= 0:
                print(f"\nEarly stopping at epoch {epoch}.")
                break
    pbar.close()
    print("--- Training Finished ---")

    # --- Valutazione Finale & Salvataggio ---
    final_test_ppl = math.inf
    if best_model_state_dict is not None:
        print("\nEvaluating best model on Test Set...")
        model.load_state_dict(best_model_state_dict)
        model.eval()
        with torch.no_grad():
             final_test_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print(f"   Best Dev PPL achieved: {best_dev_ppl:.4f}")
        print(f"   Final Test PPL: {final_test_ppl:.4f}")

        print("\nSaving best model state dict...")
        os.makedirs(model_save_dir, exist_ok=True)
        # --- MODIFICA: Nome file per A1 ---
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
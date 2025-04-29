# Esegue la configurazione A2 migliore (LSTM+Dropout+SGD) e salva il modello.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import copy

# Rimosso: functools.partial
import os
from tqdm import tqdm

# Import Moduli Custom (versioni originali)
from functions import train_loop, eval_loop, init_weights
from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
from model import LM_LSTM_With_Dropout

if __name__ == "__main__":
    # --- Impostazioni ---
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    experiment_step = "Part1A_Step2_Best"
    model_save_dir = "bin"

    # --- Parametri Fissi Configurazione Migliore A2 ---
    lr = 1.5
    batch_size = 64
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
        pad_token_id = lang.word2id["<pad>"]  # Ottieni l'ID padding
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
        f"   LR={lr}, BS={batch_size}, EmbDrop={emb_dropout_rate}, FcDrop={fc_dropout_rate}, Optim=SGD"
    )

    # --- MODIFICA: Funzione Wrapper per Collate ---
    def collate_wrapper(batch_data):
        # Chiama la collate_fn originale passando i valori fissati con i nomi originali
        return collate_fn(batch_data, pad_token=pad_token_id, DEVICE=DEVICE)

    # -------------------------------------------

    # --- DataLoader (usa wrapper) ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_wrapper,  # Usa la funzione wrapper
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size * 2,
        collate_fn=collate_wrapper,  # Usa la funzione wrapper
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        collate_fn=collate_wrapper,  # Usa la funzione wrapper
    )

    # --- Modello, Optimizer, Loss ---
    print("Initializing Model, Optimizer, Loss...")
    model = LM_LSTM_With_Dropout(
        emb_size=emb,
        hidden_size=hid,
        output_size=vocab_len,
        pad_index=pad_token_id,  # Usa l'ID
        emb_dropout=emb_dropout_rate,
        fc_dropout=fc_dropout_rate,
    ).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr)  # SGD per A2
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
        # Usa 'clip' come atteso da functions.py originale
        train_loss = train_loop(
            train_loader, optimizer, criterion_train, model, clip=clip
        )

        model.eval()
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

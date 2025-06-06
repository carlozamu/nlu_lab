import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import copy
from functools import partial
import os
from tqdm import tqdm
import numpy as np

from functions import train_loop, eval_loop, init_weights
from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
from model import LM_LSTM_With_Dropout, LM_LSTM_Base


def evaluate_saved_model(
    model_path, test_loader, lang, emb_size, hid_size, vocab_len, device
):
    """Carica e valuta un modello salvato"""
    print(f"Loading model from: {model_path}")

    # Deduce model type and training config from filename
    if "Step1" in model_path:
        print("  Config: LSTM + SGD")
        model = LM_LSTM_Base(
            emb_size=emb_size,
            hidden_size=hid_size,
            output_size=vocab_len,
            pad_index=lang.word2id["<pad>"],
        ).to(device)
    elif "Step2" in model_path:
        print("  Config: LSTM + Dropout + SGD")
        model = LM_LSTM_With_Dropout(
            emb_size=emb_size,
            hidden_size=hid_size,
            output_size=vocab_len,
            pad_index=lang.word2id["<pad>"],
            emb_dropout=0.1,
            fc_dropout=0.1,
        ).to(device)
    else:  # Step3
        print("  Config: LSTM + Dropout + AdamW")
        model = LM_LSTM_With_Dropout(
            emb_size=emb_size,
            hidden_size=hid_size,
            output_size=vocab_len,
            pad_index=lang.word2id["<pad>"],
            emb_dropout=0.1,
            fc_dropout=0.1,
        ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    ppl, loss = eval_loop(test_loader, criterion_eval, model)
    print(f"Test Perplexity: {ppl:.4f}")
    return ppl, loss


if __name__ == "__main__":
    # ========================
    # CONFIGURATION
    # ========================
    EVAL_ONLY = True

    # Device setup
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    experiment_step = "Part1A_Step3_Best"
    model_save_dir = "bin"

    # Hyperparameters
    lr = 0.001
    batch_size = 16
    hid = 300  # LSTM hidden size
    emb = 200  # Embedding dimension
    clip = 5  # Gradient clipping value
    n_epochs = 100
    patience = 3  # Early stopping patience
    emb_dropout = 0.1  # Embedding dropout rate
    fc_dropout = 0.1  # Output layer dropout rate

    # Load Penn TreeBank dataset
    train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")

    # Create vocabulary with special tokens
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    print("Data loaded.")

    # Create datasets that convert words to IDs
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Create data loaders with padding and batching
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        collate_fn=partial(
            collate_fn, pad_token_id=lang.word2id["<pad>"], device=DEVICE
        ),
    )

    # ========================
    # EVALUATION OR TRAINING
    # ========================
    if EVAL_ONLY:
        print(f"\n--- EVALUATION ONLY MODE ---")
        if os.path.exists("bin/"):
            model_files = [f for f in os.listdir("bin/") if f.endswith(".pt")]
            if model_files:
                print(f"Found {len(model_files)} model(s). Evaluating all...")
                for i, model_file in enumerate(model_files, 1):
                    print(f"\n{i}. {model_file}")
                    model_path = os.path.join("bin", model_file)
                    evaluate_saved_model(
                        model_path, test_loader, lang, emb, hid, vocab_len, DEVICE
                    )
            else:
                print("No .pt files found in bin/")
        else:
            print("bin/ directory not found")
    else:
        print(f"\n--- Running Configuration: {experiment_step} ---")
        print(f"   LR={lr}, BS={batch_size}, Optim=AdamW")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=partial(
                collate_fn, pad_token_id=lang.word2id["<pad>"], device=DEVICE
            ),
            shuffle=True,
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=batch_size * 2,
            collate_fn=partial(
                collate_fn, pad_token_id=lang.word2id["<pad>"], device=DEVICE
            ),
        )

        # Multiple runs for statistical significance
        runs = 5
        test_ppls = []
        
        for run_idx in tqdm(range(0, runs), desc="Total Runs"):
            print(f"\n--- Run {run_idx + 1}/{runs} ---")
            
            # Initialize model, optimizer, and loss functions
            print("Initializing Model, Optimizer, Loss...")
            model = LM_LSTM_With_Dropout(
                emb_size=emb,
                hidden_size=hid,
                output_size=vocab_len,
                pad_index=lang.word2id["<pad>"],
                emb_dropout=emb_dropout,
                fc_dropout=fc_dropout,
            ).to(DEVICE)
            model.apply(init_weights)  # Apply custom weight initialization

            optimizer = optim.AdamW(model.parameters(), lr=lr)
            criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
            criterion_eval = nn.CrossEntropyLoss(
                ignore_index=lang.word2id["<pad>"], reduction="sum"
            )

            # Training loop with early stopping
            print("Starting Training...")
            patience_counter = patience
            best_ppl = math.inf
            best_model = None
            sampled_epochs = []
            losses_train = []
            losses_dev = []

            pbar = tqdm(range(1, n_epochs + 1), desc=f"Run {run_idx + 1} Epochs", leave=False)
            for epoch in pbar:
                # Train for one epoch
                loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

                # Evaluate on development set every epoch
                if epoch % 1 == 0:
                    sampled_epochs.append(epoch)
                    losses_train.append(loss)
                    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                    losses_dev.append(loss_dev)
                    pbar.set_description(f"PPL: {ppl_dev:.4f}")

                    # Early stopping logic: save best model and check patience
                    if ppl_dev < best_ppl:
                        best_ppl = ppl_dev
                        best_model = copy.deepcopy(model).to("cpu")
                        patience_counter = patience
                    else:
                        patience_counter -= 1

                    if patience_counter <= 0:
                        print(f"\nEarly stopping at epoch {epoch}.")
                        break

            # Final evaluation on test set
            if best_model is not None:
                best_model.to(DEVICE)
                final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
                print(f"Run {run_idx + 1} - Test ppl: {final_ppl:.4f}")
                test_ppls.append(final_ppl)

                # Save model for the best run only
                if run_idx == 0:  # Save only first run for now
                    print("\nSaving best model from first run...")
                    os.makedirs(model_save_dir, exist_ok=True)
                    model_filename = f"{experiment_step}_lr{lr}_bs{batch_size}.pt"
                    model_save_path = os.path.join(model_save_dir, model_filename)
                    try:
                        torch.save(best_model.state_dict(), model_save_path)
                        print(f"Model saved to: {model_save_path}")
                    except Exception as e:
                        print(f"Error saving model: {e}")
            else:
                print("\nNo best model found for this run.")
        
        # Calculate statistics across all runs
        test_ppls = np.array(test_ppls)
        print("\n--- FINAL RESULTS ---")
        print(f"Test Perplexity: {test_ppls.mean():.4f} ± {test_ppls.std():.4f}")

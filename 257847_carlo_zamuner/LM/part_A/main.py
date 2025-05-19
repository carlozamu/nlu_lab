
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import copy
from functools import partial
import os
from tqdm import tqdm

# Custom module imports
try:
    from functions import train_loop, eval_loop, init_weights
    from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
    from model import LM_LSTM_With_Dropout # Model for Step A3
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure they are accessible.")
    exit()

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================
if __name__ == "__main__":
    # ==============================================================================
    # --- GLOBAL SETTINGS AND FIXED PARAMETERS ---
    # ==============================================================================
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    experiment_step = "Part1A_Step3_Best" # Identifier for this specific run
    model_save_dir = "bin"                 # Directory to save the model

    # Fixed parameters for the best A3 configuration
    lr = 0.001
    batch_size = 16
    hid = 300
    emb = 200
    clip_grad_norm_value = 5 # Renamed for clarity, original 'clip' was 5
    n_epochs = 100
    patience_setting = 3
    emb_dropout_rate = 0.1
    fc_dropout_rate = 0.1

    # ==============================================================================
    # --- DATA LOADING AND PREPARATION ---
    # ==============================================================================
    print("\nLoading data & vocabulary...")
    try:
        train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
        dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
        test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")
        lang = Lang(train_raw, ["<pad>", "<eos>"])
        vocab_len = len(lang.word2id)
        pad_token_id = lang.word2id["<pad>"] # Used for collate_fn and loss ignore_index
        print("Data loaded.")
    except FileNotFoundError as e:
        print(f"Error: Dataset not found: {e}")
        exit()

    print("Creating Datasets...")
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    print("Datasets created.")

    # ==============================================================================
    # --- SINGLE CONFIGURATION EXECUTION ---
    # ==============================================================================
    print(f"\n--- Running Configuration: {experiment_step} ---")
    print(f"   LR={lr}, BS={batch_size}, Optim=AdamW")

    # --- DataLoader Creation ---
    # Assuming 'utils.collate_fn' expects 'pad_token' and 'DEVICE' as original
    # If utils.py was updated to expect 'pad_token_id' and 'device', adjust here.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token=pad_token_id, DEVICE=DEVICE),
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size * 2,
        collate_fn=partial(collate_fn, pad_token=pad_token_id, DEVICE=DEVICE),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        collate_fn=partial(collate_fn, pad_token=pad_token_id, DEVICE=DEVICE),
    )

    # --- Model, Optimizer, Loss Initialization ---
    print("Initializing Model, Optimizer, Loss...")
    model = LM_LSTM_With_Dropout(
        emb_size=emb,
        hidden_size=hid,
        output_size=vocab_len,
        pad_index=pad_token_id,
        emb_dropout=emb_dropout_rate,
        fc_dropout=fc_dropout_rate,
    ).to(DEVICE)
    model.apply(init_weights) # Apply custom weight initialization

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    # Use reduction='sum' for eval criterion to correctly calculate perplexity later
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="sum")

    # ==============================================================================
    # --- TRAINING LOOP ---
    # ==============================================================================
    print("Starting Training...")
    patience = patience_setting
    best_dev_ppl = math.inf
    best_model_state_dict = None # Store the state_dict of the best model

    pbar = tqdm(range(1, n_epochs + 1), desc="Epoch", leave=True)
    for epoch in pbar:
        # Training step
        # Assuming 'functions.train_loop' expects 'clip' as original
        # If functions.py was updated to 'clip_value', adjust here.
        train_loss = train_loop(
            train_loader, optimizer, criterion_train, model, clip=clip_grad_norm_value
        )

        # Evaluation step
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        pbar.set_description(f"Epoch {epoch}/{n_epochs} -> Dev PPL: {ppl_dev:.3f}")

        # Early stopping and best model saving logic
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

    # ==============================================================================
    # --- FINAL EVALUATION AND MODEL SAVING ---
    # ==============================================================================
    final_test_ppl = math.inf
    if best_model_state_dict is not None:
        print("\nEvaluating best model on Test Set...")
        model.load_state_dict(best_model_state_dict) # Load best model state
        model.eval()
        with torch.no_grad():
            final_test_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print(f"   Best Dev PPL achieved: {best_dev_ppl:.4f}")
        print(f"   Final Test PPL (using best dev model): {final_test_ppl:.4f}")

        # Save the best model state_dict
        print("\nSaving best model state dict...")
        os.makedirs(model_save_dir, exist_ok=True) # Ensure save directory exists
        model_filename = f"{experiment_step}_lr{lr}_bs{batch_size}.pt"
        model_save_path = os.path.join(model_save_dir, model_filename)
        try:
            torch.save(best_model_state_dict, model_save_path)
            print(f"   Model saved to: {model_save_path}")
        except Exception as e:
            print(f"   Error saving model: {e}")
    else:
        print("\nNo best model state saved (training might have failed or not improved).")

    print(f"\n--- Script {experiment_step} Finished ---")
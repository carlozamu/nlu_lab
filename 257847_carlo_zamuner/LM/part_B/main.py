import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import copy
from functools import partial

# os removed as we are not saving files
from tqdm import tqdm

# Custom module imports
try:
    from functions import train_loop, eval_loop, init_weights
    from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
    from model import LM_LSTM_VarDropWT  # Specific model for this run
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure they are accessible.")
    exit()

if __name__ == "__main__":
    # --- Global Settings & Parameters ---
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    # Parameters for Part1B_FineTuned_Best configuration
    run_identifier = "Part1B_FineTuned_Best"
    learning_rate = 3.75
    batch_s = 16  # Shortened variable name
    patience_epochs = 7
    nt_avg_trigger_epochs = 3  # NT-AvSGD trigger
    hidden_dim = 300
    embedding_dim = 300  # Must match hidden_dim for LM_LSTM_VarDropWT
    grad_clip_norm = 5
    num_epochs = 100
    dropout_e_rate = 0.1  # Variational dropout for embedding output
    dropout_o_rate = 0.4  # Variational dropout for LSTM output

    print(f"\n--- Running: {run_identifier} ---")
    print(
        f"   LR={learning_rate}, BS={batch_s}, Patience={patience_epochs}, Trigger={nt_avg_trigger_epochs}"
    )
    print(f"   Model: {LM_LSTM_VarDropWT.__name__}, Optim: NT-AvSGD(SGD)")

    # --- Data Loading and Preparation ---
    print("\nLoading data & vocabulary...")
    train_raw_data = read_file("./dataset/PennTreeBank/ptb.train.txt")
    dev_raw_data = read_file("./dataset/PennTreeBank/ptb.valid.txt")
    test_raw_data = read_file("./dataset/PennTreeBank/ptb.test.txt")
    lang_processor = Lang(train_raw_data, ["<pad>", "<eos>"])  # Renamed for clarity
    vocabulary_size = len(lang_processor.word2id)
    pad_idx = lang_processor.word2id["<pad>"]
    print("Data loaded.")

    print("Creating Datasets...")
    train_dataset = PennTreeBank(train_raw_data, lang_processor)
    dev_dataset = PennTreeBank(dev_raw_data, lang_processor)
    test_dataset = PennTreeBank(test_raw_data, lang_processor)
    print("Datasets created.")

    # --- DataLoader Creation ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_s,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=pad_idx, device=DEVICE),
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_s * 2,
        collate_fn=partial(collate_fn, pad_token_id=pad_idx, device=DEVICE),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_s * 2,
        collate_fn=partial(collate_fn, pad_token_id=pad_idx, device=DEVICE),
    )

    # --- Model, Optimizer, Loss Initialization ---
    print("Initializing Model, Optimizer, Loss...")
    model = LM_LSTM_VarDropWT(
        emb_size=embedding_dim,
        hidden_size=hidden_dim,
        output_size=vocabulary_size,
        pad_index=pad_idx,
        n_layers=1,  # Assuming 1 layer
        dropoute=dropout_e_rate,
        dropouto=dropout_o_rate,
    ).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Base SGD for NT-AvSGD
    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")

    # --- Training Loop ---
    print("Starting Training...")
    current_patience = patience_epochs
    avg_trigger_countdown = nt_avg_trigger_epochs
    is_averaging_active = False  # Flag for NT-AvSGD state
    averaged_model_state = None  # Stores state_dict if averaging
    avg_update_steps = 0
    averaging_triggered_at_epoch = -1
    best_dev_perplexity = math.inf
    best_ppl_for_nt_trigger = math.inf  # Tracks PPL for NT trigger logic

    pbar = tqdm(range(1, num_epochs + 1), desc=f"Epoch ({run_identifier})", leave=False)
    for epoch_num in pbar:  # Renamed for clarity
        model.train()
        train_epoch_loss = train_loop(  # Renamed for clarity
            train_loader, optimizer, criterion_train, model, clip_value=grad_clip_norm
        )

        # NT-AvSGD: Averaging step if active
        if is_averaging_active and averaged_model_state is not None:
            avg_update_steps += 1
            # Temporarily create model instance to update averaged weights
            temp_avg_model_instance = LM_LSTM_VarDropWT(
                emb_size=embedding_dim,
                hidden_size=hidden_dim,
                output_size=vocabulary_size,
                pad_index=pad_idx,
                n_layers=1,
                dropoute=dropout_e_rate,
                dropouto=dropout_o_rate,
            ).to("cpu")  # Perform averaging on CPU to save GPU memory
            temp_avg_model_instance.load_state_dict(averaged_model_state)

            with torch.no_grad():
                for avg_p, current_p in zip(
                    temp_avg_model_instance.parameters(), model.parameters()
                ):
                    avg_p.data.add_(
                        current_p.data.to(avg_p.device) - avg_p.data,
                        alpha=1.0 / avg_update_steps,
                    )
            averaged_model_state = (
                temp_avg_model_instance.state_dict()
            )  # Store updated state

        # Evaluation
        model_for_eval = model  # Default to current model
        if is_averaging_active and averaged_model_state is not None:
            # If averaging, evaluate the averaged model
            eval_averaged_model = LM_LSTM_VarDropWT(
                emb_size=embedding_dim,
                hidden_size=hidden_dim,
                output_size=vocabulary_size,
                pad_index=pad_idx,
                n_layers=1,
                dropoute=dropout_e_rate,
                dropouto=dropout_o_rate,
            ).to(DEVICE)
            eval_averaged_model.load_state_dict(averaged_model_state)
            eval_averaged_model.eval()
            model_for_eval = eval_averaged_model
        else:
            model.eval()

        current_dev_ppl, _ = eval_loop(dev_loader, criterion_eval, model_for_eval)
        pbar.set_description(
            f"Epoch {epoch_num}/{num_epochs} -> Dev PPL: {current_dev_ppl:.3f}"
        )

        # NT-AvSGD: Trigger logic
        if not is_averaging_active:
            if current_dev_ppl < best_ppl_for_nt_trigger:
                best_ppl_for_nt_trigger = current_dev_ppl
                avg_trigger_countdown = nt_avg_trigger_epochs
            else:
                avg_trigger_countdown -= 1
                if avg_trigger_countdown <= 0:
                    print(
                        f"\n   NT-AvSGD Triggered at Epoch {epoch_num}! Starting averaging."
                    )
                    is_averaging_active = True
                    averaging_triggered_at_epoch = epoch_num
                    averaged_model_state = copy.deepcopy(
                        model.state_dict()
                    )  # Start averaging from current model state
                    avg_update_steps = 1

        # Early Stopping & Tracking Best Perplexity
        if current_dev_ppl < best_dev_perplexity:
            best_dev_perplexity = current_dev_ppl
            current_patience = patience_epochs  # Reset patience
        else:
            current_patience -= 1
            if current_patience <= 0:
                print(f"\n   Early stopping at epoch {epoch_num}.")
                break
    pbar.close()
    print("--- Training Finished ---")

    # --- Final Evaluation (print only) ---
    print("\nEvaluating final model state on Test Set...")
    final_model_for_test_run = (
        model  # Default to the last state of the primary trained model
    )
    if is_averaging_active and averaged_model_state is not None:
        # If averaging was active, use the final averaged model state for the test run
        final_model_for_test_run = LM_LSTM_VarDropWT(
            emb_size=embedding_dim,
            hidden_size=hidden_dim,
            output_size=vocabulary_size,
            pad_index=pad_idx,
            n_layers=1,
            dropoute=dropout_e_rate,
            dropouto=dropout_o_rate,
        ).to(DEVICE)
        final_model_for_test_run.load_state_dict(averaged_model_state)

    final_model_for_test_run.eval()
    with torch.no_grad():
        final_test_perplexity, _ = eval_loop(
            test_loader, criterion_eval, final_model_for_test_run
        )

    print(f"\n--- Final Results for {run_identifier} ---")
    print(f"   Best Development PPL during training: {best_dev_perplexity:.4f}")
    print(f"   Final Test PPL: {final_test_perplexity:.4f}")
    trigger_status_message = (
        f"Yes (Epoch {averaging_triggered_at_epoch})"
        if averaging_triggered_at_epoch != -1
        else "No"
    )
    print(f"   Averaging Triggered: {trigger_status_message}")

    print(f"\n--- Script {run_identifier} Finished (Model Not Saved) ---")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import copy
from functools import partial # Kept for DataLoader collate_fn
import os
from tqdm import tqdm

# Custom module imports
try:
    from functions import train_loop, eval_loop, init_weights
    from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
    # Import all necessary models for Part B
    from model import LM_LSTM_WeightTying, LM_LSTM_VarDropWT
except ImportError as e:
    print(f"Error importing modules: {e}. Ensure they are accessible.")
    exit()

# ==============================================================================
# --- HELPER FUNCTION TO RUN A SINGLE CONFIGURATION ---
# ==============================================================================
def run_and_save_config(config, global_data):
    """
    Trains a model with the given configuration and saves the best performing state.

    Args:
        config (dict): Dictionary containing all parameters for the run.
        global_data (dict): Dictionary containing pre-loaded data (datasets, lang, etc.).
    """
    DEVICE = config['device']
    print(f"\n{'='*20} Starting: {config['experiment_step']} {'='*20}")
    print(f"   Config: LR={config['lr']}, BS={config['batch_size']}, "
          f"Patience={config.get('patience', 'N/A')}, Trigger={config.get('trigger', 'N/A')}")
    print(f"   Model: {config['model_class'].__name__}, Optim: {config['optimizer_name']}")

    # Unpack global data
    train_dataset = global_data['train_dataset']
    dev_dataset = global_data['dev_dataset']
    test_dataset = global_data['test_dataset']
    vocab_len = global_data['vocab_len']
    pad_token_id = global_data['pad_token_id']

    # --- DataLoader Creation ---
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id, device=DEVICE),
        drop_last=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config['batch_size'] * 2,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id, device=DEVICE)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'] * 2,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id, device=DEVICE)
    )

    # --- Model Initialization ---
    print("Initializing Model, Optimizer, Loss...")
    model_params = {
        'emb_size': config['emb'],
        'hidden_size': config['hid'],
        'output_size': vocab_len,
        'pad_index': pad_token_id,
        'n_layers': 1 # Assuming 1 layer for all models in Part B
    }
    if config['model_class'] == LM_LSTM_VarDropWT:
        model_params['dropoute'] = config['dropoute']
        model_params['dropouto'] = config['dropouto']

    model = config['model_class'](**model_params).to(DEVICE)
    model.apply(init_weights)

    # --- Optimizer and Loss Function ---
    if config['optimizer_name'].startswith("NT-AvSGD") or config['optimizer_name'] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer_name']}")

    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="sum")

    # --- Training Loop ---
    print("Starting Training...")
    current_patience_epochs = config['patience'] # Use 'patience' from config
    trigger_setting = config.get('trigger')
    avg_trigger_countdown = trigger_setting if trigger_setting is not None else float('inf')
    is_averaging = False
    avg_model = None
    avg_step_count = 0
    trigger_epoch = -1
    best_dev_ppl = math.inf
    best_model_state_dict = None
    best_ppl_for_avg_trigger = math.inf
    # epochs_trained = 0 # Not strictly needed if only using epoch from pbar
    n_epochs = config['n_epochs']
    clip_value = config['clip'] # Use 'clip_value' consistent with functions.py

    pbar = tqdm(range(1, n_epochs + 1), desc=f"Epoch ({config['experiment_step']})", leave=False)
    for epoch in pbar:
        # epochs_trained = epoch # Update if needed for detailed logging
        model.train()
        train_loss = train_loop(train_loader, optimizer, criterion_train, model, clip_value=clip_value)

        # Averaging logic (only if NT-AvSGD is intended)
        if config['optimizer_name'].startswith("NT-AvSGD") and is_averaging:
            avg_step_count += 1
            with torch.no_grad():
                for avg_param, param in zip(avg_model.parameters(), model.parameters()):
                    avg_param.data.add_(param.data.to(avg_param.device) - avg_param.data, alpha=1.0 / avg_step_count)

        # Evaluation
        eval_model_ref = avg_model if (config['optimizer_name'].startswith("NT-AvSGD") and is_averaging) else model
        if is_averaging and config['optimizer_name'].startswith("NT-AvSGD"): eval_model_ref.to(DEVICE)
        eval_model_ref.eval()
        ppl_dev, _ = eval_loop(dev_loader, criterion_eval, eval_model_ref)
        if is_averaging and config['optimizer_name'].startswith("NT-AvSGD"): eval_model_ref.to("cpu")

        pbar.set_description(f"Epoch {epoch}/{n_epochs} -> Dev PPL: {ppl_dev:.3f}")

        # NT-AvSGD Trigger logic
        if trigger_setting is not None and not is_averaging and config['optimizer_name'].startswith("NT-AvSGD"):
            if ppl_dev < best_ppl_for_avg_trigger:
                best_ppl_for_avg_trigger = ppl_dev
                avg_trigger_countdown = trigger_setting
            else:
                avg_trigger_countdown -= 1
                if avg_trigger_countdown <= 0:
                    print(f"\n   NT-AvSGD Triggered at Epoch {epoch}! Starting averaging.")
                    is_averaging = True
                    trigger_epoch = epoch
                    avg_model = copy.deepcopy(model).to("cpu")
                    avg_step_count = 1

        # Early Stopping & Best Model Logic
        if ppl_dev < best_dev_ppl:
            best_dev_ppl = ppl_dev
            model_to_save = avg_model if (config['optimizer_name'].startswith("NT-AvSGD") and is_averaging) else model
            best_model_state_dict = copy.deepcopy(model_to_save.state_dict())
            current_patience_epochs = config['patience'] # Reset patience
        else:
            current_patience_epochs -= 1
            if current_patience_epochs <= 0:
                print(f"\n   Early stopping at epoch {epoch}.")
                break
    pbar.close()
    print("--- Training Finished ---")

    # --- Final Evaluation & Saving ---
    final_test_ppl = math.inf
    if best_model_state_dict is not None:
        print("\nEvaluating best model on Test Set...")
        # Re-initialize model for loading state_dict to ensure clean state
        eval_model_params = {
            'emb_size': config['emb'], 'hidden_size': config['hid'],
            'output_size': vocab_len, 'pad_index': pad_token_id, 'n_layers': 1
        }
        if config['model_class'] == LM_LSTM_VarDropWT:
            eval_model_params['dropoute'] = config['dropoute']
            eval_model_params['dropouto'] = config['dropouto']
        eval_model = config['model_class'](**eval_model_params).to(DEVICE)
        eval_model.load_state_dict(best_model_state_dict)
        eval_model.eval()
        with torch.no_grad():
             final_test_ppl, _ = eval_loop(test_loader, criterion_eval, eval_model)

        print(f"\n--- Final Results for {config['experiment_step']} ---")
        print(f"   Best Dev PPL: {best_dev_ppl:.4f}")
        print(f"   Final Test PPL: {final_test_ppl:.4f}")
        if config['optimizer_name'].startswith("NT-AvSGD"):
            trigger_info = f"Yes (Epoch {trigger_epoch})" if trigger_epoch != -1 else "No"
            print(f"   Averaging Triggered: {trigger_info}")

        # Saving the model
        print("\nSaving best model state dict...")
        os.makedirs(config['model_save_dir'], exist_ok=True)
        model_filename = f"{config['experiment_step']}_lr{config['lr']}_bs{config['batch_size']}"
        # Add trigger and patience to filename if they are relevant (not None or default)
        if config.get('trigger') is not None: model_filename += f"_t{config['trigger']}"
        if config.get('patience') !=3 and config.get('patience') is not None : model_filename += f"_p{config['patience']}" # Assuming default patience is 3
        model_filename += ".pt"
        model_save_path = os.path.join(config['model_save_dir'], model_filename)
        try:
            torch.save(best_model_state_dict, model_save_path)
            print(f"   Model saved to: {model_save_path}")
        except Exception as e:
            print(f"   Error saving model: {e}")
    else:
        print("\nNo best model state saved for this configuration.")
    print(f"--- Finished: {config['experiment_step']} ---")

# ==============================================================================
# --- MAIN SCRIPT EXECUTION ---
# ==============================================================================
if __name__ == "__main__":
    # Common Settings
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")
    MODEL_SAVE_DIR = "bin" # Use a constant for the save directory
    N_EPOCHS_DEFAULT = 100
    CLIP_DEFAULT = 5
    HID_DEFAULT = 300 # Common hidden size

    # Load Data (once)
    print("\nLoading data & vocabulary (once)...")
    try:
        train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
        dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
        test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")
        lang_obj = Lang(train_raw, ["<pad>", "<eos>"]) # Renamed for clarity
        vocab_len = len(lang_obj.word2id)
        pad_token_id = lang_obj.word2id["<pad>"]
        print("Data loaded.")
    except FileNotFoundError as e:
        print(f"Error: Dataset not found: {e}")
        exit()

    print("Creating Datasets (once)...")
    train_dataset = PennTreeBank(train_raw, lang_obj)
    dev_dataset = PennTreeBank(dev_raw, lang_obj)
    test_dataset = PennTreeBank(test_raw, lang_obj)
    print("Datasets created.")

    # Store pre-loaded data to pass to the helper function
    global_data_package = {
        'train_dataset': train_dataset,
        'dev_dataset': dev_dataset,
        'test_dataset': test_dataset,
        'vocab_len': vocab_len,
        'pad_token_id': pad_token_id,
        'lang_obj': lang_obj # Pass lang object if needed by any part (though not directly used by run_and_save_config)
    }

    # Configuration Definitions for Part B
    configurations_to_run = [
        # B1 Best: LSTM + Weight Tying + SGD
        {
            'experiment_step': "Part1B_Step1_Best",
            'model_class': LM_LSTM_WeightTying,
            'optimizer_name': "SGD",
            'lr': 0.1, 'batch_size': 16,
            'emb': 300, 'hid': HID_DEFAULT, # emb must match hid for LM_LSTM_WeightTying
            'patience': 3, # Default patience for this step
            'trigger': None, # No NT-AvSGD
            'n_epochs': N_EPOCHS_DEFAULT, 'clip': CLIP_DEFAULT, 'device': DEVICE, 'model_save_dir': MODEL_SAVE_DIR
        },
        # B3 pre-FT Best: LSTM + WT + VarDrop + NT-AvSGD
        {
            'experiment_step': "Part1B_Step3_preFT_Best",
            'model_class': LM_LSTM_VarDropWT,
            'optimizer_name': "NT-AvSGD(SGD)",
            'lr': 2.0, 'batch_size': 16,
            'emb': 300, 'hid': HID_DEFAULT, # emb must match hid
            'dropoute': 0.1, 'dropouto': 0.4,
            'patience': 3, # Default patience for this step
            'trigger': 5,
            'n_epochs': N_EPOCHS_DEFAULT, 'clip': CLIP_DEFAULT, 'device': DEVICE, 'model_save_dir': MODEL_SAVE_DIR
        },
        # B3 post-FT Best: LSTM + WT + VarDrop + NT-AvSGD (Fine-Tuned)
        {
            'experiment_step': "Part1B_FineTuned_Best",
            'model_class': LM_LSTM_VarDropWT,
            'optimizer_name': "NT-AvSGD(SGD)",
            'lr': 3.75, 'batch_size': 16,
            'emb': 300, 'hid': HID_DEFAULT, # emb must match hid
            'dropoute': 0.1, 'dropouto': 0.4,
            'patience': 7, # Specific patience for FT
            'trigger': 3,  # Specific trigger for FT
            'n_epochs': N_EPOCHS_DEFAULT, 'clip': CLIP_DEFAULT, 'device': DEVICE, 'model_save_dir': MODEL_SAVE_DIR
        },
    ]

    # --- Main Loop over Configurations ---
    for current_config in configurations_to_run:
        run_and_save_config(current_config, global_data_package)

    print("\n--- All Part B Configurations Finished ---")
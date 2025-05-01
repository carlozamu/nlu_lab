# Esegue le 3 configurazioni migliori della Parte 1.B e salva i modelli.
# Configs: B1(LSTM+WT+SGD), B3_preFT(LSTM+WT+VarDrop+NTAvSGD), B3_FT(FineTuned B3)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import copy
from functools import partial
import os
from tqdm import tqdm

# Import Moduli Custom (versioni corrette/pulite)
try:
    from functions import train_loop, eval_loop, init_weights
    from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
    # Importa TUTTI i modelli necessari per la Parte B
    from model import LM_LSTM_WeightTying, LM_LSTM_VarDropWT
except ImportError as e:
    print(f"Errore importazione moduli: {e}. Assicurati che siano accessibili.")
    exit()

# --- Funzione Helper per Eseguire una Configurazione ---
def run_and_save_config(config):
    """Esegue un addestramento con la configurazione data e salva il miglior modello."""
    DEVICE = config['device']
    print(f"\n{'='*10} Starting: {config['experiment_step']} {'='*10}")
    print(f"   Config: LR={config['lr']}, BS={config['batch_size']}, Patience={config.get('patience', 'N/A')}, Trigger={config.get('trigger', 'N/A')}")
    print(f"   Model: {config['model_class'].__name__}, Optim: {config['optimizer_name']}")

    # --- Dati (caricati una volta all'inizio) ---
    # Assumiamo che lang, vocab_len, pad_token_id, train_dataset, dev_dataset, test_dataset
    # siano disponibili nello scope globale o passati come argomenti.
    # Per semplicità, li assumiamo globali in questo esempio.

    # --- DataLoader ---
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

    # --- Modello ---
    print("Initializing Model, Optimizer, Loss...")
    # Istanzia il modello corretto in base alla configurazione
    model_params = {
        'emb_size': config['emb'],
        'hidden_size': config['hid'],
        'output_size': vocab_len,
        'pad_index': pad_token_id,
        'n_layers': 1 # Assumiamo 1 layer
    }
    # Aggiungi parametri specifici per VarDropWT se necessario
    if config['model_class'] == LM_LSTM_VarDropWT:
        model_params['dropoute'] = config['dropoute']
        model_params['dropouto'] = config['dropouto']

    model = config['model_class'](**model_params).to(DEVICE)
    model.apply(init_weights)

    # --- Optimizer e Loss ---
    if config['optimizer_name'].startswith("NT-AvSGD"):
        optimizer = optim.SGD(model.parameters(), lr=config['lr']) # Base SGD
    elif config['optimizer_name'] == "SGD":
         optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    else:
        raise ValueError(f"Optimizer sconosciuto: {config['optimizer_name']}")

    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="sum")

    # --- Training Loop (con logica NT-AvSGD se applicabile) ---
    print("Starting Training...")
    patience = config['patience']
    trigger_setting = config.get('trigger', None) # Ottieni trigger se presente
    avg_trigger_countdown = trigger_setting if trigger_setting is not None else float('inf') # Evita trigger se non definito
    is_averaging = False
    avg_model = None
    avg_step_count = 0
    trigger_epoch = -1
    best_dev_ppl = math.inf
    best_model_state_dict = None
    best_ppl_for_avg_trigger = math.inf
    epochs_trained = 0
    n_epochs = config['n_epochs']
    clip = config['clip']

    pbar = tqdm(range(1, n_epochs + 1), desc=f"Epoch ({config['experiment_step']})", leave=False)
    for epoch in pbar:
        epochs_trained = epoch
        model.train()
        train_loss = train_loop(train_loader, optimizer, criterion_train, model, clip_value=clip)

        # Averaging (solo se NT-AvSGD)
        if is_averaging:
            avg_step_count += 1
            with torch.no_grad():
                for avg_param, param in zip(avg_model.parameters(), model.parameters()):
                    avg_param.data.add_(param.data.to(avg_param.device) - avg_param.data, alpha=1.0 / avg_step_count)

        # Valutazione
        eval_model_ref = avg_model if is_averaging else model
        if is_averaging: eval_model_ref.to(DEVICE)
        eval_model_ref.eval()
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, eval_model_ref)
        if is_averaging: eval_model_ref.to("cpu")

        pbar.set_description(f"Epoch {epoch}/{n_epochs} -> Dev PPL: {ppl_dev:.3f}")

        # Trigger NT-AvSGD (solo se definito e non ancora attivo)
        if trigger_setting is not None and not is_averaging:
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

        # Early Stopping & Miglior Modello
        if ppl_dev < best_dev_ppl:
            best_dev_ppl = ppl_dev
            model_to_save = avg_model if is_averaging else model
            best_model_state_dict = copy.deepcopy(model_to_save.state_dict())
            patience = config['patience']
        else:
            patience -= 1
            if patience <= 0:
                print(f"\n   Early stopping at epoch {epoch}.")
                break
    pbar.close()
    print("--- Training Finished ---")

    # --- Valutazione Finale & Salvataggio ---
    final_test_ppl = math.inf
    trigger_info = f"Yes (Epoch {trigger_epoch})" if trigger_epoch != -1 else "No"
    if best_model_state_dict is not None:
        print("\nEvaluating best model on Test Set...")
        # Ricarica il modello corretto per la valutazione
        eval_model = config['model_class'](**model_params).to(DEVICE)
        eval_model.load_state_dict(best_model_state_dict)
        eval_model.eval()
        with torch.no_grad():
             final_test_ppl, _ = eval_loop(test_loader, criterion_eval, eval_model)

        print(f"\n--- Final Results for {config['experiment_step']} ---")
        print(f"   Best Dev PPL: {best_dev_ppl:.4f}")
        print(f"   Final Test PPL: {final_test_ppl:.4f}")
        if config['optimizer_name'].startswith("NT-AvSGD"):
            print(f"   Averaging Triggered: {trigger_info}")

        print("\nSaving best model state dict...")
        os.makedirs(config['model_save_dir'], exist_ok=True)
        model_filename = f"{config['experiment_step']}_lr{config['lr']}_bs{config['batch_size']}"
        if 'trigger' in config: model_filename += f"_t{config['trigger']}"
        if 'patience' in config and config['patience'] != 3 : model_filename += f"_p{config['patience']}"
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


# ================== SCRIPT PRINCIPALE ==================
if __name__ == "__main__":
    # --- Impostazioni Comuni ---
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")
    model_save_dir = "bin"
    n_epochs = 100
    clip = 5
    hid = 300 # Dimensione nascosta comune

    # --- Caricamento Dati (una sola volta) ---
    print("\nLoading data & vocabulary (once)...")
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

    print("Creating Datasets (once)...")
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    print("Datasets created.")

    # --- Definizioni delle Configurazioni da Eseguire ---
    configurations = [
        # --- B1 Best ---
        {
            'experiment_step': "Part1B_Step1_Best",
            'model_class': LM_LSTM_WeightTying,
            'optimizer_name': "SGD",
            'lr': 0.1,
            'batch_size': 16,
            'emb': 300, # Deve matchare hid per questo modello
            'hid': 300,
            'patience': 3, # Patience default
            'trigger': None, # No NT-AvSGD
            'n_epochs': n_epochs, 'clip': clip, 'device': DEVICE, 'model_save_dir': model_save_dir
        },
        # --- B3 pre-FT Best ---
        {
            'experiment_step': "Part1B_Step3_preFT_Best",
            'model_class': LM_LSTM_VarDropWT,
            'optimizer_name': "NT-AvSGD(SGD)",
            'lr': 2.0,
            'batch_size': 16,
            'emb': 300, # Deve matchare hid per questo modello
            'hid': 300,
            'dropoute': 0.1,
            'dropouto': 0.4,
            'patience': 3, # Patience default
            'trigger': 5,  # Trigger per B3 pre-FT
            'n_epochs': n_epochs, 'clip': clip, 'device': DEVICE, 'model_save_dir': model_save_dir
        },
        # --- B3 post-FT Best ---
        {
            'experiment_step': "Part1B_FineTuned_Best",
            'model_class': LM_LSTM_VarDropWT,
            'optimizer_name': "NT-AvSGD(SGD)",
            'lr': 3.75,
            'batch_size': 16,
            'emb': 300, # Deve matchare hid per questo modello
            'hid': 300,
            'dropoute': 0.1,
            'dropouto': 0.4,
            'patience': 7, # Patience per FT
            'trigger': 3,  # Trigger per FT
            'n_epochs': n_epochs, 'clip': clip, 'device': DEVICE, 'model_save_dir': model_save_dir
        },
    ]

    # --- Loop Principale sulle Configurazioni ---
    for config in configurations:
        # Passa le variabili globali necessarie (dati, vocab_len, pad_token_id) alla funzione
        # Questo evita di doverle ridefinire dentro la funzione o passarle esplicitamente
        # nel dizionario config, che sarebbe meno pulito.
        run_and_save_config(config)

    print("\n--- All Part B Configurations Finished ---")
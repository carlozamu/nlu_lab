# Questo script esegue l'esperimento corrispondente al Punto 1 della Part 1.A:
# - Utilizza LSTM invece di RNN.
# - Mantiene l'ottimizzatore SGD.
# - Non utilizza i dropout aggiuntivi.
# - Salva risultati (CSV, PNG, TXT) e genera grafici.

# --- Import Standard e PyTorch ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import copy
from functools import partial
import os # Necessario per creare cartelle
import pandas as pd # Necessario per il CSV
import matplotlib.pyplot as plt # Necessario per i grafici
from tqdm import tqdm
from tabulate import tabulate # Per output formattato

# --- Import Moduli Custom ---
from functions import train_loop, eval_loop, init_weights
from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
from model import LM_LSTM_Base # Modello LSTM base (Step 1)

if __name__ == "__main__":
    # --- 1. Impostazioni Iniziali ---
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # --- Nome Esperimento e Configurazione Iperparametri (per cartelle output) ---
    experiment_name = "puntoA1" # Nome base per questo set di modifiche (LSTM)
    # Parametri fissi per questa esecuzione specifica
    lr = 0.001
    batch_size = 16
    hid = 300
    emb = 200
    clip = 5
    n_epochs = 100
    patience_setting = 3 # Rinominato per chiarezza

    # Stringa per identificare la configurazione specifica di iperparametri
    hyperparam_str = f"lr_{lr}_bs_{batch_size}_emb_{emb}_hid_{hid}" # Aggiunti emb/hid per completezza

    # --- Creazione Cartelle di Output ---
    output_dir = os.path.join("results", experiment_name, hyperparam_str)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    # --- 2. Caricamento Dati e Vocabolario ---
    print("Loading data...")
    train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)

    # --- 3. Creazione Dataset e DataLoader ---
    print("Creating Datasets and DataLoaders...")
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=True, collate_fn=partial( collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE) )
    dev_loader = DataLoader( dev_dataset, batch_size=batch_size * 2, collate_fn=partial( collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE) )
    test_loader = DataLoader( test_dataset, batch_size=batch_size * 2, collate_fn=partial( collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE) )

    # --- 4. Stampa Configurazione Esperimento ---
    exp_params = [
        ["Experiment Step", f"{experiment_name} (Step 1: LSTM + SGD)"],
        ["Output Directory", output_dir],
        ["Device", DEVICE],
        ["Model", "LM_LSTM_Base"],
        ["Optimizer", "SGD"],
        ["Learning Rate", lr],
        ["Batch Size (train)", batch_size],
        ["Embedding Size", emb],
        ["Hidden Size", hid],
        ["Gradient Clip", clip],
        ["Max Epochs", n_epochs],
        ["Patience", patience_setting],
    ]
    print("\n--- Experiment Configuration ---")
    print(tabulate(exp_params, headers=["Parameter", "Value"], tablefmt="pretty"))

    # --- 5. Inizializzazione Modello, Ottimizzatore, Loss ---
    print("\nInitializing Model, Optimizer, and Loss...")
    model = LM_LSTM_Base(emb, hid, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction="sum")

    # --- 6. Training Loop ---
    print("\n--- Starting Training ---")
    patience = patience_setting
    sampled_epochs = []
    losses_train = []
    losses_dev = []
    ppls_dev = []
    best_dev_ppl = math.inf
    best_model = None

    pbar = tqdm(range(1, n_epochs + 1), desc="Training Progress")
    for epoch in pbar:
        train_loss = train_loop(train_loader, optimizer, criterion_train, model, clip=clip)
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

        pbar.set_description(f"Epoch {epoch}/{n_epochs} -> Dev PPL: {ppl_dev:.3f} | Train Loss: {train_loss:.3f}")

        # Salva metriche dell'epoca
        sampled_epochs.append(epoch)
        losses_train.append(train_loss)
        losses_dev.append(loss_dev)
        ppls_dev.append(ppl_dev)

        # Early stopping e salvataggio modello migliore
        if ppl_dev < best_dev_ppl:
            best_dev_ppl = ppl_dev
            best_model = copy.deepcopy(model).to("cpu") # Salva il miglior modello sulla CPU
            patience = patience_setting # Resetta la pazienza
            pbar.set_postfix_str("✨ New Best Model!")
        else:
            patience -= 1
            pbar.set_postfix_str(f"Patience: {patience}/{patience_setting}")
            if patience <= 0:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break # Interrompi il ciclo

    print("--- Training Finished ---")

    # --- 7. Salvataggio Dati CSV ---
    if sampled_epochs: # Solo se almeno un'epoca è stata completata
        print(f"\nSaving training metrics to CSV...")
        results_df = pd.DataFrame({
            'epoch': sampled_epochs,
            'train_loss': losses_train,
            'dev_loss': losses_dev,
            'dev_perplexity': ppls_dev
        })
        csv_path = os.path.join(output_dir, "training_metrics.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Metrics saved to: {csv_path}")
    else:
        print("\nNo epochs completed, skipping CSV saving.")

    # --- 8. Valutazione Finale e Salvataggio Risultati TXT ---
    final_test_ppl = math.inf # Valore di default se non c'è un best_model
    epochs_trained = 0
    if sampled_epochs:
        epochs_trained = sampled_epochs[-1]

    if best_model is not None:
        print("\nEvaluating best model on Test Set...")
        best_model.to(DEVICE) # Sposta il miglior modello sul device corretto per la valutazione
        final_test_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print(f"Final Test PPL: {final_test_ppl:.3f}")

        print("\nSaving final results to TXT...")
        results_summary = (
            f"Experiment: {experiment_name} (Step 1: LSTM + SGD)\n"
            f"Configuration: {hyperparam_str}\n"
            f"-------------------------------------\n"
            f"Best Development PPL: {best_dev_ppl:.4f}\n"
            f"Final Test PPL: {final_test_ppl:.4f}\n"
            f"Total Epochs Trained: {epochs_trained}\n"
            f"Achieved Best Dev PPL at Epoch: {sampled_epochs[ppls_dev.index(best_dev_ppl)] if best_dev_ppl != math.inf else 'N/A'}\n"
        )
        results_txt_path = os.path.join(output_dir, "final_results.txt")
        with open(results_txt_path, "w") as f:
            f.write(results_summary)
        print(f"Results summary saved to: {results_txt_path}")
    else:
        print("\nNo best model found (e.g., PPL never improved). Skipping final evaluation and TXT saving.")
        results_summary = "Training failed to produce a best model." # Messaggio per il terminale


    # --- 9. Generazione e Salvataggio Grafici ---
    if sampled_epochs: # Solo se ci sono dati da plottare
        print("\nGenerating and saving plots...")
        plt.style.use('seaborn-v0_8-grid') # Stile gradevole per i grafici

        # Grafico 1: Loss (Training vs Development)
        plt.figure(figsize=(10, 6))
        plt.plot(sampled_epochs, losses_train, label='Training Loss', marker='o', linestyle='-')
        plt.plot(sampled_epochs, losses_dev, label='Development Loss', marker='x', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Cross-Entropy Loss')
        plt.title(f'Training vs Development Loss\n({experiment_name} - {hyperparam_str})')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(output_dir, "loss_vs_epochs.png")
        plt.savefig(loss_plot_path)
        plt.close() # Chiudi la figura per liberare memoria
        print(f"Loss plot saved to: {loss_plot_path}")

        # Grafico 2: Perplexity (Development)
        plt.figure(figsize=(10, 6))
        plt.plot(sampled_epochs, ppls_dev, label='Development Perplexity', marker='o', linestyle='-', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Perplexity (PPL)')
        plt.title(f'Development Perplexity over Epochs\n({experiment_name} - {hyperparam_str})')
        # Evidenzia il punto di minima PPL
        if best_dev_ppl != math.inf:
            best_epoch_idx = ppls_dev.index(best_dev_ppl)
            best_epoch = sampled_epochs[best_epoch_idx]
            plt.scatter(best_epoch, best_dev_ppl, color='blue', s=100, zorder=5, label=f'Best PPL ({best_dev_ppl:.2f}) at Epoch {best_epoch}')
        plt.legend()
        plt.grid(True)
        ppl_plot_path = os.path.join(output_dir, "perplexity_vs_epochs.png")
        plt.savefig(ppl_plot_path)
        plt.close() # Chiudi la figura
        print(f"Perplexity plot saved to: {ppl_plot_path}")
    else:
        print("\nNo epochs completed, skipping plot generation.")


    # --- 10. Stampa Finale Risultati su Terminale ---
    print("\n--- Final Summary ---")
    final_results_data = [
        ["Best Development PPL", f"{best_dev_ppl:.4f}" if best_dev_ppl != math.inf else "N/A"],
        ["Final Test PPL", f"{final_test_ppl:.4f}" if final_test_ppl != math.inf else "N/A"],
        ["Total Epochs Trained", epochs_trained if epochs_trained > 0 else "N/A"],
    ]
    print(tabulate(final_results_data, headers=["Metric", "Value"], tablefmt="pretty"))
    print("--- Experiment Run Finished ---")
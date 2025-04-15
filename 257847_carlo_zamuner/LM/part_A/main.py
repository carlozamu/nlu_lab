# Questo script esegue l'esperimento corrispondente al Punto 1 della Part 1.A:
# - Utilizza LSTM invece di RNN.
# - Mantiene l'ottimizzatore SGD.
# - Non utilizza i dropout aggiuntivi.
# - ITERA su diverse combinazioni di Learning Rate e Batch Size.
# - Salva risultati (CSV, PNG, TXT) per OGNI combinazione.
# - ALLA FINE, aggrega i risultati e crea grafici comparativi.

# --- Import Standard e PyTorch ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import copy
from functools import partial
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Per heatmap
from tqdm import tqdm
from tabulate import tabulate

# --- Import Moduli Custom ---
from functions import train_loop, eval_loop, init_weights
from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
# !! Importa il modello LSTM Base !!
from model import LM_LSTM_Base

if __name__ == "__main__":
    # --- 1. Impostazioni Globali e Liste Iperparametri ---
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Global settings: Using device: {DEVICE}")

    # !! Nome Esperimento per Step 1 !!
    experiment_name = "puntoA1" # Nome base per questo set di modifiche (LSTM + SGD)
    main_output_folder = os.path.join("results", experiment_name) # Cartella principale per A1

    # Liste di iperparametri da testare
    learning_rates = [0.1, 1.0, 1.5]
    batch_sizes = [16, 32, 64, 128]

    # Parametri fissi
    hid = 300
    emb = 200
    clip = 5
    n_epochs = 100
    patience_setting = 3

    # --- 2. Caricamento Dati e Vocabolario (una sola volta) ---
    print("\nLoading data and building vocabulary (once)...")
    train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)

    # --- 3. Creazione Dataset (una sola volta) ---
    print("Creating Datasets (once)...")
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # --- 4. Loop sugli Iperparametri ---
    print("\n--- Starting Hyperparameter Tuning Loop ---")
    total_runs = len(learning_rates) * len(batch_sizes)
    current_run = 0
    all_run_results = [] # Lista per raccogliere i risultati di ogni run

    for lr in learning_rates:
        for batch_size in batch_sizes:
            current_run += 1
            print(f"\n===== RUN {current_run}/{total_runs} =====")
            print(f"Configuration: lr={lr}, batch_size={batch_size}")

            # --- 4.1. Configurazione Specifica della Run ---
            hyperparam_str = f"lr_{lr}_bs_{batch_size}_emb_{emb}_hid_{hid}"
            output_dir = os.path.join(main_output_folder, hyperparam_str) # Usa cartella principale A1
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory for this run: {output_dir}")

            # --- 4.2. Creazione DataLoader ---
            # (Codice identico a prima)
            print("Creating DataLoaders for current batch size...")
            train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=partial( collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),
                                      drop_last=True)
            dev_loader = DataLoader( dev_dataset, batch_size=batch_size * 2,
                                     collate_fn=partial( collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE) )
            test_loader = DataLoader( test_dataset, batch_size=batch_size * 2,
                                      collate_fn=partial( collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE) )


            # --- 4.3. Stampa Configurazione della Run ---
            # (Codice identico a prima)
            exp_params = [
                ["Experiment Step", f"{experiment_name} (Step 1: LSTM + SGD)"],
                ["Run", f"{current_run}/{total_runs}"],
                ["Output Directory", output_dir],
                ["Device", DEVICE],
                ["Model", "LM_LSTM_Base"],
                ["Optimizer", "SGD"],
                ["Current Learning Rate", lr],
                ["Current Batch Size", batch_size],
                ["Embedding Size", emb],
                ["Hidden Size", hid],
                ["Gradient Clip", clip],
                ["Max Epochs", n_epochs],
                ["Patience", patience_setting],
            ]
            print("\n--- Current Run Configuration ---")
            print(tabulate(exp_params, headers=["Parameter", "Value"], tablefmt="pretty"))

            # --- 4.4. Inizializzazione Modello, Ottimizzatore, Loss ---
            # (Codice identico a prima)
            print("\nInitializing Model, Optimizer, and Loss for this run...")
            model = LM_LSTM_Base(
                emb_size=emb, hidden_size=hid, output_size=vocab_len,
                pad_index=lang.word2id["<pad>"]
            ).to(DEVICE)
            model.apply(init_weights)
            optimizer = optim.SGD(model.parameters(), lr=lr)
            criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
            criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction="sum")


            # --- 4.5. Training Loop Specifico della Run ---
            # (Codice identico a prima)
            print("\n--- Starting Training for this configuration ---")
            patience = patience_setting
            sampled_epochs = []
            losses_train = []
            losses_dev = []
            ppls_dev = []
            best_dev_ppl = math.inf
            best_model = None
            pbar = tqdm(range(1, n_epochs + 1), desc=f"Run {current_run} (lr={lr}, bs={batch_size})", leave=False)
            # ... (loop epoche identico a prima) ...
            for epoch in pbar:
                train_loss = train_loop(train_loader, optimizer, criterion_train, model, clip=clip)
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                pbar.set_description(f"Epoch {epoch}/{n_epochs} -> Dev PPL: {ppl_dev:.3f} | Train Loss: {train_loss:.3f}")
                sampled_epochs.append(epoch)
                losses_train.append(train_loss)
                losses_dev.append(loss_dev)
                ppls_dev.append(ppl_dev)
                if ppl_dev < best_dev_ppl:
                    best_dev_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to("cpu")
                    patience = patience_setting
                    pbar.set_postfix_str("✨ New Best Model!")
                else:
                    patience -= 1
                    pbar.set_postfix_str(f"Patience: {patience}/{patience_setting}")
                    if patience <= 0:
                        print(f"\nEarly stopping triggered at epoch {epoch} for this run.")
                        break
            pbar.close()
            print("--- Training Finished for this configuration ---")


            # --- 4.6. Salvataggio Dati CSV (per questa run) ---
            # (Codice identico a prima)
            if sampled_epochs:
                print(f"\nSaving training metrics to CSV for this run...")
                results_df = pd.DataFrame({ 'epoch': sampled_epochs, 'train_loss': losses_train, 'dev_loss': losses_dev, 'dev_perplexity': ppls_dev })
                csv_path = os.path.join(output_dir, "training_metrics.csv")
                results_df.to_csv(csv_path, index=False)
                print(f"Metrics saved to: {csv_path}")
            else:
                print("\nNo epochs completed for this run, skipping CSV saving.")


            # --- 4.7. Valutazione Finale e Salvataggio Risultati TXT (per questa run) ---
            final_test_ppl = math.inf # Inizializza a inf
            epochs_trained = 0
            best_epoch_num = None
            if sampled_epochs:
                epochs_trained = sampled_epochs[-1]
                if best_dev_ppl != math.inf:
                   best_epoch_num = sampled_epochs[ppls_dev.index(best_dev_ppl)]


            if best_model is not None:
                print("\nEvaluating best model on Test Set for this run...")
                best_model.to(DEVICE)
                final_test_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
                print(f"Final Test PPL for this run: {final_test_ppl:.3f}")

                print("\nSaving final results to TXT for this run...")
                results_summary = (
                    f"Experiment: {experiment_name} (Step 1: LSTM + SGD)\n"
                    f"Configuration: {hyperparam_str}\n"
                    f"-------------------------------------\n"
                    f"Best Development PPL: {best_dev_ppl:.4f}\n"
                    f"Final Test PPL: {final_test_ppl:.4f}\n"
                    f"Total Epochs Trained: {epochs_trained}\n"
                    f"Achieved Best Dev PPL at Epoch: {best_epoch_num if best_epoch_num is not None else 'N/A'}\n"
                )
                results_txt_path = os.path.join(output_dir, "final_results.txt")
                with open(results_txt_path, "w") as f:
                    f.write(results_summary)
                print(f"Results summary saved to: {results_txt_path}")

                # --- NUOVO: Aggiungi risultati alla lista aggregata ---
                run_summary_dict = {
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'best_dev_ppl': best_dev_ppl,
                    'final_test_ppl': final_test_ppl,
                    'epochs_trained': epochs_trained,
                    'best_epoch': best_epoch_num
                }
                all_run_results.append(run_summary_dict)
                # ----------------------------------------------------

            else:
                print("\nNo best model found for this run. Skipping final evaluation and TXT saving.")
                results_summary = "Training failed to produce a best model for this run."
                 # --- NUOVO: Aggiungi placeholder se la run fallisce ---
                run_summary_dict = {
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'best_dev_ppl': math.inf, # O NaN
                    'final_test_ppl': math.inf, # O NaN
                    'epochs_trained': epochs_trained,
                    'best_epoch': None
                }
                all_run_results.append(run_summary_dict)
                # ----------------------------------------------------


            # --- 4.8. Generazione e Salvataggio Grafici (per questa run) ---
            # (Codice identico a prima, usa 'seaborn-v0_8-darkgrid')
            if sampled_epochs:
                print("\nGenerating and saving plots for this run...")
                try:
                    plt.style.use('seaborn-v0_8-darkgrid')
                    print("Using 'seaborn-v0_8-darkgrid' style for plots.")
                except OSError:
                    print("Warning: Style 'seaborn-v0_8-darkgrid' not found. Using default style.")
                    pass
                # ... (codice plot loss e ppl identico a prima) ...
                # Grafico 1: Loss
                plt.figure(figsize=(10, 6))
                plt.plot(sampled_epochs, losses_train, label='Training Loss', marker='o', linestyle='-')
                plt.plot(sampled_epochs, losses_dev, label='Development Loss', marker='x', linestyle='--')
                plt.xlabel('Epochs')
                plt.ylabel('Cross-Entropy Loss')
                plt.title(f'Training vs Development Loss\n({experiment_name} - Run {current_run}: lr={lr}, bs={batch_size})')
                plt.legend()
                loss_plot_path = os.path.join(output_dir, "loss_vs_epochs.png")
                plt.savefig(loss_plot_path)
                plt.close()
                print(f"Loss plot saved to: {loss_plot_path}")

                # Grafico 2: Perplexity
                plt.figure(figsize=(10, 6))
                plt.plot(sampled_epochs, ppls_dev, label='Development Perplexity', marker='o', linestyle='-', color='red')
                plt.xlabel('Epochs')
                plt.ylabel('Perplexity (PPL)')
                plt.title(f'Development Perplexity over Epochs\n({experiment_name} - Run {current_run}: lr={lr}, bs={batch_size})')
                if best_dev_ppl != math.inf:
                    # best_epoch_idx = ppls_dev.index(best_dev_ppl) # Già calcolato sopra come best_epoch_num
                    plt.scatter(best_epoch_num, best_dev_ppl, color='blue', s=100, zorder=5, label=f'Best PPL ({best_dev_ppl:.2f}) at Epoch {best_epoch_num}')
                plt.legend()
                ppl_plot_path = os.path.join(output_dir, "perplexity_vs_epochs.png")
                plt.savefig(ppl_plot_path)
                plt.close()
                print(f"Perplexity plot saved to: {ppl_plot_path}")

            else:
                print("\nNo epochs completed for this run, skipping plot generation.")


            # --- 4.9. Stampa Finale Risultati della Run su Terminale ---
            # (Codice identico a prima)
            print(f"\n--- Summary for Run {current_run} (lr={lr}, bs={batch_size}) ---")
            final_results_data = [
                ["Best Development PPL", f"{best_dev_ppl:.4f}" if best_dev_ppl != math.inf else "N/A"],
                ["Final Test PPL", f"{final_test_ppl:.4f}" if final_test_ppl != math.inf else "N/A"],
                ["Total Epochs Trained", epochs_trained if epochs_trained > 0 else "N/A"],
            ]
            print(tabulate(final_results_data, headers=["Metric", "Value"], tablefmt="pretty"))
            print(f"===== END OF RUN {current_run}/{total_runs} =====")


    print("\n--- All Hyperparameter Tuning Runs Finished ---")


    # --- 5. Analisi Aggregata e Grafici Comparativi ---
    print(f"\n--- Starting Aggregate Analysis for {experiment_name} ---")

    if not all_run_results:
        print("No run results collected. Skipping aggregate analysis.")
    else:
        # Crea DataFrame riassuntivo
        summary_df = pd.DataFrame(all_run_results)
        # Converte PPL infinite in NaN per plotting/analisi
        summary_df.replace([math.inf, -math.inf], pd.NA, inplace=True)

        # Salva il DataFrame riassuntivo
        summary_csv_path = os.path.join(main_output_folder, f"summary_results_{experiment_name}.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSummary results saved to: {summary_csv_path}")

        # Stampa il DataFrame riassuntivo
        print("\n--- Aggregate Results Summary ---")
        # Rimuovi NaN per la stampa, se preferisci mostra "Failed" o lascia vuoto
        print(tabulate(summary_df.round(4).fillna('Failed'), headers='keys', tablefmt='pretty', showindex=False))


        # Genera Heatmap (Best Dev PPL)
        try:
            # Pivot per heatmap: indice=lr, colonne=bs, valori=best_dev_ppl
            # Assicurati che i tipi siano corretti (float per PPL)
            summary_df['best_dev_ppl'] = pd.to_numeric(summary_df['best_dev_ppl'], errors='coerce')
            heatmap_data_dev = summary_df.pivot(index='learning_rate', columns='batch_size', values='best_dev_ppl')

            plt.figure(figsize=(10, 7))
            sns.heatmap(heatmap_data_dev, annot=True, fmt=".2f", cmap="viridis_r", # _r inverte la mappa (valori bassi = colori "migliori")
                        linewidths=.5, cbar_kws={'label': 'Best Development PPL'})
            plt.title(f'Best Development PPL Heatmap ({experiment_name})')
            plt.xlabel("Batch Size")
            plt.ylabel("Learning Rate")
            heatmap_dev_path = os.path.join(main_output_folder, f"summary_heatmap_dev_ppl_{experiment_name}.png")
            plt.savefig(heatmap_dev_path)
            plt.close()
            print(f"Development PPL heatmap saved to: {heatmap_dev_path}")

        except Exception as e:
            print(f"Could not generate Dev PPL heatmap: {e}")


        # Genera Heatmap (Final Test PPL) - Opzionale, simile a sopra
        try:
            summary_df['final_test_ppl'] = pd.to_numeric(summary_df['final_test_ppl'], errors='coerce')
            heatmap_data_test = summary_df.pivot(index='learning_rate', columns='batch_size', values='final_test_ppl')

            plt.figure(figsize=(10, 7))
            sns.heatmap(heatmap_data_test, annot=True, fmt=".2f", cmap="viridis_r",
                        linewidths=.5, cbar_kws={'label': 'Final Test PPL'})
            plt.title(f'Final Test PPL Heatmap ({experiment_name})')
            plt.xlabel("Batch Size")
            plt.ylabel("Learning Rate")
            heatmap_test_path = os.path.join(main_output_folder, f"summary_heatmap_test_ppl_{experiment_name}.png")
            plt.savefig(heatmap_test_path)
            plt.close()
            print(f"Test PPL heatmap saved to: {heatmap_test_path}")

        except Exception as e:
            print(f"Could not generate Test PPL heatmap: {e}")


    print("\n--- Aggregate Analysis Finished ---")
    print(f"--- Experiment {experiment_name} Run Finished ---")
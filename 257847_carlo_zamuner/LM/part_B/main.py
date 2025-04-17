# Questo script esegue l'esperimento corrispondente al Punto B3:
# - Parte dalla configurazione B2 (LSTM + SGD + WT + VarDrop).
# - AGGIUNGE Non-monotonically Triggered Averaged SGD (NT-AvSGD).
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
import seaborn as sns  # Per heatmap
from tqdm import tqdm
from tabulate import tabulate

# --- Import Moduli Custom ---
from functions import train_loop, eval_loop, init_weights
from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn

# !! Importa il modello con Variational Dropout e WT (lo stesso di B2) !!
from model import LM_LSTM_VarDropWT

if __name__ == "__main__":
    # --- 1. Impostazioni Globali e Liste Iperparametri ---
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Global settings: Using device: {DEVICE}")

    # !! Nome Esperimento per Step B3 - NT-AvSGD !!
    experiment_name = "puntoB3_NTAvSGD"
    main_output_folder = os.path.join(
        "results", experiment_name
    )  # Cartella principale per B3

    # Liste di iperparametri da testare (usiamo quelle di A1/B1/B2)
    learning_rates = [0.05, 0.1, 0.5, 1, 1.5, 2]
    batch_sizes = [16, 32, 64, 128]

    # Parametri fissi
    hid = 300
    emb = 300  # Mantiene emb == hid per WT
    clip = 5
    n_epochs = 100
    patience_setting = 3  # Pazienza per early stopping standard
    # Tassi di Variational Dropout (come B2)
    dropoute_rate = 0.1
    dropouto_rate = 0.4
    # !! Parametri per NT-AvSGD !!
    AVG_TRIGGER_THRESHOLD = 5  # Epoche di non miglioramento per attivare AvSGD

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
    all_run_results = []  # Lista per raccogliere i risultati di ogni run

    for lr in learning_rates:
        for batch_size in batch_sizes:
            current_run += 1
            print(f"\n===== RUN {current_run}/{total_runs} =====")
            print(
                f"Configuration: lr={lr}, batch_size={batch_size}, Optimizer=NT-AvSGD, Model=VarDrop+WT, drop_e={dropoute_rate}, drop_o={dropouto_rate}"
            )

            # --- 4.1. Configurazione Specifica della Run ---
            hyperparam_str = f"lr_{lr}_bs_{batch_size}_emb_{emb}_hid_{hid}_VarDropWT_e{dropoute_rate}_o{dropouto_rate}_NTAvSGD{AVG_TRIGGER_THRESHOLD}"
            output_dir = os.path.join(main_output_folder, hyperparam_str)
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory for this run: {output_dir}")

            # --- 4.2. Creazione DataLoader ---
            print("Creating DataLoaders for current batch size...")
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=partial(
                    collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE
                ),
                drop_last=True,
            )
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=batch_size * 2,
                collate_fn=partial(
                    collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE
                ),
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size * 2,
                collate_fn=partial(
                    collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE
                ),
            )

            # --- 4.3. Stampa Configurazione della Run ---
            exp_params = [
                # !! Aggiorna nome step e ottimizzatore !!
                [
                    "Experiment Step",
                    f"{experiment_name} (Step B3: LSTM + NT-AvSGD + WT + VarDrop)",
                ],
                ["Run", f"{current_run}/{total_runs}"],
                ["Output Directory", output_dir],
                ["Device", DEVICE],
                ["Model", "LM_LSTM_VarDropWT"],
                [
                    "Optimizer",
                    f"NT-AvSGD (Trigger={AVG_TRIGGER_THRESHOLD})",
                ],  # !! Aggiornato !!
                ["Base Optimizer", "SGD"],  # Specifichiamo l'ottimizzatore sottostante
                ["Current Learning Rate", lr],
                ["Current Batch Size", batch_size],
                ["Embedding Size", emb],
                ["Hidden Size", hid],
                ["Weight Tying", True],
                ["Dropout Embedding (Var)", dropoute_rate],
                ["Dropout Output (Var)", dropouto_rate],
                ["Gradient Clip", clip],
                ["Max Epochs", n_epochs],
                ["Patience (Early Stop)", patience_setting],
            ]
            print("\n--- Current Run Configuration ---")
            print(
                tabulate(exp_params, headers=["Parameter", "Value"], tablefmt="pretty")
            )

            # --- 4.4. Inizializzazione Modello, Ottimizzatore, Loss ---
            print("\nInitializing Model, Optimizer, and Loss for this run...")
            model = LM_LSTM_VarDropWT(
                emb_size=emb,
                hidden_size=hid,
                output_size=vocab_len,
                pad_index=lang.word2id["<pad>"],
                dropoute=dropoute_rate,
                dropouto=dropouto_rate,
            ).to(DEVICE)
            model.apply(init_weights)

            # Ottimizzatore base SGD
            optimizer = optim.SGD(model.parameters(), lr=lr)
            criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
            criterion_eval = nn.CrossEntropyLoss(
                ignore_index=lang.word2id["<pad>"], reduction="sum"
            )

            # --- 4.5. Training Loop Specifico della Run con NT-AvSGD ---
            print(
                "\n--- Starting Training for this configuration (with NT-AvSGD logic) ---"
            )
            patience = patience_setting  # Per early stopping standard
            avg_trigger_countdown = AVG_TRIGGER_THRESHOLD  # Per attivare AvSGD
            is_averaging = False  # Flag per stato AvSGD
            avg_model = None  # Placeholder per modello mediato
            avg_step_count = 0  # Contatore step mediati
            best_ppl_for_avg_trigger = math.inf  # Miglior PPL per triggerare
            trigger_epoch = -1  # Epoca in cui si attiva l'averaging

            sampled_epochs = []
            losses_train = []
            losses_dev = []
            ppls_dev = []
            best_dev_ppl = math.inf  # Miglior PPL (potrebbe essere da avg_model)
            best_model_state_dict = (
                None  # Salva lo state_dict del modello migliore (avg o non avg)
            )

            pbar = tqdm(
                range(1, n_epochs + 1),
                desc=f"Run {current_run} (lr={lr}, bs={batch_size}, NT-AvSGD)",
                leave=False,
            )
            for epoch in pbar:
                # --- Training Step (usa sempre 'model' e 'optimizer' base) ---
                train_loss = train_loop(
                    train_loader, optimizer, criterion_train, model, clip=clip
                )
                losses_train.append(train_loss)

                # --- Averaging Update (se attivo) ---
                if is_averaging:
                    avg_step_count += 1
                    # Aggiorna i parametri del modello mediato
                    # Usiamo to('cpu') per sicurezza memoria GPU, ma potrebbe rallentare
                    with torch.no_grad():
                        for avg_param, param in zip(
                            avg_model.parameters(), model.parameters()
                        ):
                            avg_param.data.add_(
                                param.data.to(avg_param.device) - avg_param.data,
                                alpha=1.0 / avg_step_count,
                            )

                # --- Evaluation Step ---
                # Se stiamo facendo averaging, valutiamo con avg_model, altrimenti con model
                eval_model_ref = avg_model if is_averaging else model
                eval_model_ref.to(DEVICE)  # Assicura che sia sul device giusto
                ppl_dev_current_epoch, loss_dev_current_epoch = eval_loop(
                    dev_loader, criterion_eval, eval_model_ref
                )
                if is_averaging:
                    eval_model_ref.to("cpu")  # Sposta indietro avg_model se era su CPU

                losses_dev.append(loss_dev_current_epoch)
                ppls_dev.append(ppl_dev_current_epoch)
                sampled_epochs.append(epoch)

                pbar.set_description(
                    f"Epoch {epoch}/{n_epochs} -> Dev PPL: {ppl_dev_current_epoch:.3f} | Train Loss: {train_loss:.3f}"
                )

                # --- Logica NT-AvSGD Trigger & Early Stopping ---
                avg_triggered_this_epoch = False
                if ppl_dev_current_epoch < best_ppl_for_avg_trigger:
                    # Miglioramento PPL (resetta countdown per trigger)
                    best_ppl_for_avg_trigger = ppl_dev_current_epoch
                    avg_trigger_countdown = AVG_TRIGGER_THRESHOLD
                    pbar.set_postfix_str(f"PPL Imp.->AvgTrig Reset")
                elif not is_averaging:
                    # Non miglioramento E non stiamo ancora facendo averaging
                    avg_trigger_countdown -= 1
                    pbar.set_postfix_str(
                        f"AvgTrig: {avg_trigger_countdown}/{AVG_TRIGGER_THRESHOLD}"
                    )
                    if avg_trigger_countdown <= 0:
                        print(
                            f"\n--- NT-AvSGD Triggered at epoch {epoch}! Starting weight averaging. ---"
                        )
                        is_averaging = True
                        trigger_epoch = epoch
                        # Inizializza avg_model come copia del modello *corrente*
                        avg_model = copy.deepcopy(model).to("cpu")  # Inizia su CPU
                        avg_step_count = (
                            1  # Primo step di averaging sarà il prossimo aggiornamento
                        )

                # Logica per Early Stopping standard e salvataggio miglior modello
                # Confrontiamo sempre con best_dev_ppl (che tiene traccia del migliore finora, avg o non avg)
                if ppl_dev_current_epoch < best_dev_ppl:
                    best_dev_ppl = ppl_dev_current_epoch
                    # Salva lo *state dict* del modello che ha dato la best PPL
                    # Se is_averaging, salva avg_model, altrimenti model
                    model_to_save = avg_model if is_averaging else model
                    best_model_state_dict = copy.deepcopy(model_to_save.state_dict())
                    patience = patience_setting  # Resetta pazienza early stopping
                    pbar.set_postfix_str(
                        f"✨ New Best PPL! AvgTrig:{avg_trigger_countdown}"
                    )
                else:
                    patience -= 1
                    pbar.set_postfix_str(
                        f"Patience: {patience}/{patience_setting} AvgTrig:{avg_trigger_countdown}"
                    )
                    if patience <= 0:
                        print(
                            f"\nStandard Early stopping triggered at epoch {epoch} for this run."
                        )
                        break
            pbar.close()
            print("--- Training Finished for this configuration ---")

            # --- 4.6. Salvataggio Dati CSV (per questa run) ---
            if sampled_epochs:
                print(f"\nSaving training metrics to CSV for this run...")
                results_df = pd.DataFrame(
                    {
                        "epoch": sampled_epochs,
                        "train_loss": losses_train,
                        "dev_loss": losses_dev,
                        "dev_perplexity": ppls_dev,
                    }
                )
                csv_path = os.path.join(output_dir, "training_metrics.csv")
                results_df.to_csv(csv_path, index=False)
                print(f"Metrics saved to: {csv_path}")
            else:
                print("\nNo epochs completed for this run, skipping CSV saving.")

            # --- 4.7. Valutazione Finale e Salvataggio Risultati TXT (per questa run) ---
            final_test_ppl = math.inf
            epochs_trained = 0
            best_epoch_num = None
            final_model_to_eval = None  # Modello da usare per il test set

            if sampled_epochs:
                epochs_trained = sampled_epochs[-1]
                if best_dev_ppl != math.inf:
                    # Trova l'epoca corrispondente alla best_dev_ppl salvata
                    best_epoch_num = sampled_epochs[ppls_dev.index(best_dev_ppl)]

            # Determina quale modello usare per la valutazione finale
            if best_model_state_dict is not None:
                print("\nLoading best model state for final evaluation...")
                # Ricrea l'architettura e carica lo state_dict salvato
                final_model_to_eval = LM_LSTM_VarDropWT(
                    emb_size=emb,
                    hidden_size=hid,
                    output_size=vocab_len,
                    pad_index=lang.word2id["<pad>"],
                    dropoute=dropoute_rate,
                    dropouto=dropouto_rate,
                ).to(DEVICE)
                final_model_to_eval.load_state_dict(best_model_state_dict)
                final_model_to_eval.eval()  # Metti in modalità valutazione

                print("Evaluating best model on Test Set for this run...")
                final_test_ppl, _ = eval_loop(
                    test_loader, criterion_eval, final_model_to_eval
                )
                print(f"Final Test PPL for this run: {final_test_ppl:.3f}")

                print("\nSaving final results to TXT for this run...")
                results_summary = (
                    f"Experiment: {experiment_name} (Step B3: LSTM + NT-AvSGD + WT + VarDrop)\n"
                    f"Configuration: {hyperparam_str}\n"
                    f"Optimizer: NT-AvSGD (Base: SGD, Trigger={AVG_TRIGGER_THRESHOLD})\n"
                    f"Model Features: Weight Tying, Variational Dropout\n"
                    f"Dropout Rates (Emb/Out): {dropoute_rate}/{dropouto_rate}\n"
                    f"Averaging Triggered: {'Yes (at epoch ' + str(trigger_epoch) + ')' if trigger_epoch != -1 else 'No'}\n"  # Indica se triggerato
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

                # Aggiungi risultati alla lista aggregata
                run_summary_dict = {
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "optimizer": f"NT-AvSGD(T={AVG_TRIGGER_THRESHOLD})",  # Aggiornato
                    "weight_tying": True,
                    "var_dropout_e": dropoute_rate,
                    "var_dropout_o": dropouto_rate,
                    "avg_triggered": trigger_epoch != -1,  # Boolean se triggerato
                    "trigger_epoch": trigger_epoch if trigger_epoch != -1 else None,
                    "best_dev_ppl": best_dev_ppl,
                    "final_test_ppl": final_test_ppl,
                    "epochs_trained": epochs_trained,
                    "best_epoch": best_epoch_num,
                }
                all_run_results.append(run_summary_dict)

            else:  # Caso in cui non si è salvato neanche un modello
                print(
                    "\nNo best model state found for this run. Skipping final evaluation and TXT saving."
                )
                # Aggiungi placeholder
                run_summary_dict = {
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "optimizer": f"NT-AvSGD(T={AVG_TRIGGER_THRESHOLD})",
                    "weight_tying": True,
                    "var_dropout_e": dropoute_rate,
                    "var_dropout_o": dropouto_rate,
                    "avg_triggered": False,
                    "trigger_epoch": None,
                    "best_dev_ppl": math.inf,
                    "final_test_ppl": math.inf,
                    "epochs_trained": epochs_trained,
                    "best_epoch": None,
                }
                all_run_results.append(run_summary_dict)

            # --- 4.8. Generazione e Salvataggio Grafici (per questa run) ---
            if sampled_epochs:
                print("\nGenerating and saving plots for this run...")
                try:
                    plt.style.use("seaborn-v0_8-darkgrid")
                    print("Using 'seaborn-v0_8-darkgrid' style for plots.")
                except OSError:
                    print(
                        "Warning: Style 'seaborn-v0_8-darkgrid' not found. Using default style."
                    )
                    pass

                # Grafico 1: Loss
                plt.figure(figsize=(10, 6))
                plt.plot(
                    sampled_epochs,
                    losses_train,
                    label="Training Loss",
                    marker="o",
                    linestyle="-",
                )
                plt.plot(
                    sampled_epochs,
                    losses_dev,
                    label="Development Loss",
                    marker="x",
                    linestyle="--",
                )
                # Aggiungi linea verticale per trigger AvSGD
                if trigger_epoch != -1:
                    plt.axvline(
                        x=trigger_epoch,
                        color="r",
                        linestyle="--",
                        label=f"AvSGD Trigger (Epoch {trigger_epoch})",
                    )
                plt.xlabel("Epochs")
                plt.ylabel("Cross-Entropy Loss")
                plt.title(
                    f"Training vs Development Loss\n({experiment_name} - Run {current_run}: lr={lr}, bs={batch_size}, NT-AvSGD)"
                )
                plt.legend()
                loss_plot_path = os.path.join(output_dir, "loss_vs_epochs.png")
                plt.savefig(loss_plot_path)
                plt.close()
                print(f"Loss plot saved to: {loss_plot_path}")

                # Grafico 2: Perplexity
                plt.figure(figsize=(10, 6))
                plt.plot(
                    sampled_epochs,
                    ppls_dev,
                    label="Development Perplexity",
                    marker="o",
                    linestyle="-",
                    color="red",
                )
                # Aggiungi linea verticale per trigger AvSGD
                if trigger_epoch != -1:
                    plt.axvline(
                        x=trigger_epoch,
                        color="r",
                        linestyle="--",
                        label=f"AvSGD Trigger (Epoch {trigger_epoch})",
                    )
                plt.xlabel("Epochs")
                plt.ylabel("Perplexity (PPL)")
                plt.title(
                    f"Development Perplexity over Epochs\n({experiment_name} - Run {current_run}: lr={lr}, bs={batch_size}, NT-AvSGD)"
                )
                if best_dev_ppl != math.inf:
                    plt.scatter(
                        best_epoch_num,
                        best_dev_ppl,
                        color="blue",
                        s=100,
                        zorder=5,
                        label=f"Best PPL ({best_dev_ppl:.2f}) at Epoch {best_epoch_num}",
                    )
                plt.legend()
                ppl_plot_path = os.path.join(output_dir, "perplexity_vs_epochs.png")
                plt.savefig(ppl_plot_path)
                plt.close()
                print(f"Perplexity plot saved to: {ppl_plot_path}")
            else:
                print("\nNo epochs completed for this run, skipping plot generation.")

            # --- 4.9. Stampa Finale Risultati della Run su Terminale ---
            print(f"\n--- Summary for Run {current_run} (lr={lr}, bs={batch_size}) ---")
            final_results_data = [
                [
                    "Averaging Triggered",
                    f"{'Yes (Epoch ' + str(trigger_epoch) + ')' if trigger_epoch != -1 else 'No'}",
                ],
                [
                    "Best Development PPL",
                    f"{best_dev_ppl:.4f}" if best_dev_ppl != math.inf else "N/A",
                ],
                [
                    "Final Test PPL",
                    f"{final_test_ppl:.4f}" if final_test_ppl != math.inf else "N/A",
                ],
                [
                    "Total Epochs Trained",
                    epochs_trained if epochs_trained > 0 else "N/A",
                ],
            ]
            print(
                tabulate(
                    final_results_data, headers=["Metric", "Value"], tablefmt="pretty"
                )
            )
            print(f"===== END OF RUN {current_run}/{total_runs} =====")

    print("\n--- All Hyperparameter Tuning Runs Finished ---")

    # --- 5. Analisi Aggregata e Grafici Comparativi ---
    print(f"\n--- Starting Aggregate Analysis for {experiment_name} ---")

    if not all_run_results:
        print("No run results collected. Skipping aggregate analysis.")
    else:
        # Crea DataFrame riassuntivo
        summary_df = pd.DataFrame(all_run_results)
        summary_df.replace([math.inf, -math.inf], pd.NA, inplace=True)

        # Salva il DataFrame riassuntivo
        summary_csv_path = os.path.join(
            main_output_folder, f"summary_results_{experiment_name}.csv"
        )
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nSummary results saved to: {summary_csv_path}")

        # Stampa il DataFrame riassuntivo
        print("\n--- Aggregate Results Summary ---")
        print(
            tabulate(
                summary_df.round(4).fillna("Failed"),
                headers="keys",
                tablefmt="pretty",
                showindex=False,
            )
        )

        # Genera Heatmap (Best Dev PPL)
        try:
            summary_df["best_dev_ppl"] = pd.to_numeric(
                summary_df["best_dev_ppl"], errors="coerce"
            )
            heatmap_data_dev = summary_df.pivot(
                index="learning_rate", columns="batch_size", values="best_dev_ppl"
            )
            plt.figure(figsize=(10, 7))
            sns.heatmap(
                heatmap_data_dev,
                annot=True,
                fmt=".2f",
                cmap="viridis_r",
                linewidths=0.5,
                cbar_kws={"label": "Best Development PPL"},
            )
            # !! Aggiorna titolo !!
            plt.title(
                f"Best Development PPL Heatmap ({experiment_name}: LSTM+NTAvSGD+WT+VarDrop)"
            )
            plt.xlabel("Batch Size")
            plt.ylabel("Learning Rate")
            heatmap_dev_path = os.path.join(
                main_output_folder, f"summary_heatmap_dev_ppl_{experiment_name}.png"
            )
            plt.savefig(heatmap_dev_path)
            plt.close()
            print(f"Development PPL heatmap saved to: {heatmap_dev_path}")
        except Exception as e:
            print(f"Could not generate Dev PPL heatmap: {e}")

        # Genera Heatmap (Final Test PPL)
        try:
            summary_df["final_test_ppl"] = pd.to_numeric(
                summary_df["final_test_ppl"], errors="coerce"
            )
            heatmap_data_test = summary_df.pivot(
                index="learning_rate", columns="batch_size", values="final_test_ppl"
            )
            plt.figure(figsize=(10, 7))
            sns.heatmap(
                heatmap_data_test,
                annot=True,
                fmt=".2f",
                cmap="viridis_r",
                linewidths=0.5,
                cbar_kws={"label": "Final Test PPL"},
            )
            # !! Aggiorna titolo !!
            plt.title(
                f"Final Test PPL Heatmap ({experiment_name}: LSTM+NTAvSGD+WT+VarDrop)"
            )
            plt.xlabel("Batch Size")
            plt.ylabel("Learning Rate")
            heatmap_test_path = os.path.join(
                main_output_folder, f"summary_heatmap_test_ppl_{experiment_name}.png"
            )
            plt.savefig(heatmap_test_path)
            plt.close()
            print(f"Test PPL heatmap saved to: {heatmap_test_path}")
        except Exception as e:
            print(f"Could not generate Test PPL heatmap: {e}")

    print("\n--- Aggregate Analysis Finished ---")
    print(f"--- Experiment {experiment_name} Run Finished ---")

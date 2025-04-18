# Questo script esegue un FINE-TUNING dell'esperimento B3:
# - Usa la configurazione base: LSTM + WT + VarDrop + NT-AvSGD.
# - ITERA su uno specifico range di LR (con BS, Patience, Trigger fissi).
# - Salva risultati (CSV, PNG, TXT) per OGNI combinazione di LR.
# - ALLA FINE, aggrega i risultati e crea un grafico comparativo (Bar Plot LR vs PPL).
# - AGGIUNTA: Monitoraggio tempo trascorso e stimato rimanente.

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
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
import warnings
import time  # <-- AGGIUNTO

# Ignora warning specifici
warnings.filterwarnings("ignore", message="seaborn gridspec")
warnings.filterwarnings("ignore", message="use_inf_as_na") # Aggiunto per pandas/seaborn

# --- Import Moduli Custom ---
# Assicurati che functions.py, utils.py, model.py siano nella stessa directory o nel python path
try:
    from functions import train_loop, eval_loop, init_weights
    from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
    from model import LM_LSTM_VarDropWT
except ImportError as e:
    print(f"Errore nell'importazione dei moduli: {e}")
    print("Assicurati che functions.py, utils.py, model.py siano accessibili.")
    exit()

# --- Helper Function per formattare il tempo ---
def format_time(seconds):
    """Converte secondi in formato HH:MM:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    # --- 1. Impostazioni Globali e Liste Iperparametri ---
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Global settings: Using device: {DEVICE}")

    # MODIFICA: Nome esperimento più specifico
    experiment_name = "puntoB3_LR_Scan_FixedParams"
    main_output_folder = os.path.join("results", experiment_name)
    os.makedirs(main_output_folder, exist_ok=True) # Crea la cartella principale se non esiste

    # MODIFICA: Nuovi iperparametri come richiesto
    learning_rates = [3.0, 3.25, 3.5, 3.75, 4.0]
    batch_sizes = [16]
    patience_values = [7]
    trigger_values = [3]

    # Parametri fissi del modello/training
    hid = 300
    emb = 300 # Deve essere == hid per Weight Tying in questo modello
    clip = 5
    n_epochs = 100
    #0.1 e 0.4 ref dal paper
    dropoute_rate = 0.1 # Dropout su embedding output
    dropouto_rate = 0.4 # Dropout su LSTM output

    # --- 2. Caricamento Dati e Vocabolario ---
    print("\nLoading data and building vocabulary (once)...")
    try:
        train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
        dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
        test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")
        vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
        lang = Lang(train_raw, ["<pad>", "<eos>"])
        vocab_len = len(lang.word2id)
        print("Data and vocabulary loaded successfully.")
    except FileNotFoundError as e:
        print(f"Errore: File dataset non trovato: {e}")
        print("Assicurati che il percorso './dataset/PennTreeBank/' sia corretto.")
        exit()

    # --- 3. Creazione Dataset ---
    print("Creating Datasets (once)...")
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    print("Datasets created.")

    # --- 4. Loop sugli Iperparametri ---
    print(f"\n--- Starting Experiment: {experiment_name} ---")
    print(f"Iterating over Learning Rates: {learning_rates}")
    print(f"Fixed Batch Size: {batch_sizes[0]}, Patience: {patience_values[0]}, Trigger: {trigger_values[0]}")

    total_runs = (
        len(learning_rates)
        * len(batch_sizes)
        * len(patience_values)
        * len(trigger_values)
    )
    current_run = 0
    all_run_results = []

    # !! Registra ora di inizio del loop !!
    start_time_total = time.time()

    # Il loop esterno ora è solo su learning_rates
    for lr in learning_rates:
        # I valori di bs, patience, trigger sono fissi
        batch_size = batch_sizes[0]
        current_patience = patience_values[0]
        current_trigger = trigger_values[0]

        current_run += 1
        print(f"\n===== RUN {current_run}/{total_runs} =====")
        print(
            f"Configuration: lr={lr}, bs={batch_size}, patience={current_patience}, trigger={current_trigger}"
        )

        # --- 4.1. Configurazione Specifica della Run ---
        hyperparam_str = f"lr_{lr}_bs_{batch_size}_P{current_patience}_T{current_trigger}_emb_{emb}_hid_{hid}_VarDropWT_e{dropoute_rate}_o{dropouto_rate}"
        output_dir = os.path.join(main_output_folder, f"run_lr_{lr}") # Cartella specifica per LR
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory for this run: {output_dir}")

        # --- 4.2. Creazione DataLoader ---
        # print("Creating DataLoaders...") # Rimosso per brevità
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=partial(
                collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE
            ),
            drop_last=True,
        )
        # Usa batch size * 2 per validazione/test per velocizzare
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

        # --- 4.3. Stampa Configurazione della Run (Opzionale ma utile) ---
        exp_params = [
            ["Learning Rate", lr],
            ["Batch Size", batch_size],
            ["Patience (Early Stop)", current_patience],
            ["Trigger (NT-AvSGD)", current_trigger],
            ["Embedding Size", emb],
            ["Hidden Size", hid],
            ["Max Epochs", n_epochs],
            ["Dropout Embedding", dropoute_rate],
            ["Dropout LSTM Out", dropouto_rate],
            ["Gradient Clipping", clip],
            ["Model Type", "LSTM + WeightTying + VarDrop"],
            ["Optimizer", "NT-AvSGD (base SGD)"]
        ]
        print("\n--- Current Run Configuration ---")
        print(tabulate(exp_params, headers=["Parameter", "Value"], tablefmt="pretty"))


        # --- 4.4. Inizializzazione Modello, Ottimizzatore, Loss ---
        print("\nInitializing Model, Optimizer, and Loss...")
        model = LM_LSTM_VarDropWT(
            emb_size=emb,
            hidden_size=hid,
            output_size=vocab_len,
            pad_index=lang.word2id["<pad>"],
            n_layers=1, # Assumendo 1 layer LSTM, modifica se necessario
            dropoute=dropoute_rate,
            dropouto=dropouto_rate,
        ).to(DEVICE)
        model.apply(init_weights) # Applica inizializzazione custom
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion_train = nn.CrossEntropyLoss(
            ignore_index=lang.word2id["<pad>"]
        )
        criterion_eval = nn.CrossEntropyLoss(
            ignore_index=lang.word2id["<pad>"], reduction="sum" # Per calcolo PPL corretto
        )
        print("Model, Optimizer, Loss Initialized.")

        # --- 4.5. Training Loop Specifico della Run con NT-AvSGD ---
        print("\n--- Starting Training ---")
        patience = current_patience
        avg_trigger_countdown = current_trigger
        is_averaging = False
        avg_model = None
        avg_step_count = 0
        # Rimuovi best_ppl_for_avg_trigger qui, verrà inizializzato dentro il loop
        trigger_epoch = -1
        sampled_epochs = []
        losses_train = []
        losses_dev = []
        ppls_dev = []
        best_dev_ppl = math.inf
        best_model_state_dict = None

        # Inizializza il best PPL per il trigger *prima* dell'inizio del loop epoche
        # Idealmente dovresti fare una valutazione iniziale, ma per semplicità lo settiamo a inf
        # Questo significa che il trigger si attiverà solo dopo current_trigger epoche *senza miglioramento*
        best_ppl_for_avg_trigger = math.inf

        pbar = tqdm(
            range(1, n_epochs + 1),
            desc=f"Run {current_run}/{total_runs} (lr={lr})", # Rimosso BS/P/T dalla desc.
            leave=False,
        )
        for epoch in pbar:
            # --- Training Step ---
            model.train() # Assicura che il modello sia in modalità training
            train_loss = train_loop(
                train_loader, optimizer, criterion_train, model, clip=clip
            )
            losses_train.append(train_loss)

            # --- Averaging Step (se attivo) ---
            if is_averaging:
                avg_step_count += 1
                with torch.no_grad():
                    for avg_param, param in zip(avg_model.parameters(), model.parameters()):
                        # Aggiorna la media: avg_t = avg_{t-1} + (param_t - avg_{t-1}) / step_count
                        avg_param.data.add_(param.data.to(avg_param.device) - avg_param.data, alpha=1.0 / avg_step_count)

            # --- Evaluation Step ---
            # Seleziona il modello corretto per la valutazione
            eval_model_ref = avg_model if is_averaging else model
            if is_averaging: eval_model_ref.to(DEVICE) # Sposta avg_model su DEVICE solo per eval

            eval_model_ref.eval() # Metti il modello selezionato in modalità valutazione
            ppl_dev_current_epoch, loss_dev_current_epoch = eval_loop(
                dev_loader, criterion_eval, eval_model_ref
            )

            if is_averaging: eval_model_ref.to("cpu") # Riporta avg_model su CPU dopo eval

            losses_dev.append(loss_dev_current_epoch)
            ppls_dev.append(ppl_dev_current_epoch)
            sampled_epochs.append(epoch)
            pbar.set_description(
                f"E{epoch} -> DevPPL:{ppl_dev_current_epoch:.2f} | TrL:{train_loss:.2f}"
            )

            # --- Logica NT-AvSGD Trigger ---
            if not is_averaging: # Controlla trigger solo se non si sta già mediando
                if ppl_dev_current_epoch < best_ppl_for_avg_trigger:
                    # Miglioramento PPL -> Resetta il contatore del trigger non-monotono (n)
                    best_ppl_for_avg_trigger = ppl_dev_current_epoch
                    avg_trigger_countdown = current_trigger
                    # print(f"Epoch {epoch}: Dev PPL improved to {ppl_dev_current_epoch:.2f}. Trigger reset to {current_trigger}.") # Debug print
                else:
                    # Nessun miglioramento PPL rispetto al migliore nella finestra -> Decrementa contatore
                    avg_trigger_countdown -= 1
                    # print(f"Epoch {epoch}: Dev PPL did not improve ({ppl_dev_current_epoch:.2f} vs best {best_ppl_for_avg_trigger:.2f}). Trigger countdown: {avg_trigger_countdown}.") # Debug print
                    if avg_trigger_countdown <= 0:
                        # Il contatore è a zero -> Attiva l'averaging (irreversibile)
                        print(f"\n *** NT-AvSGD Triggered at Epoch {epoch}! Starting averaging. ***")
                        is_averaging = True
                        trigger_epoch = epoch
                        # Inizializza il modello mediato come copia del modello corrente
                        avg_model = copy.deepcopy(model).to("cpu") # Sposta su CPU per risparmiare GPU VRAM
                        avg_step_count = 1 # Contatore passi per la media


            # --- Logica Early Stopping & Miglior Modello ---
            if ppl_dev_current_epoch < best_dev_ppl:
                best_dev_ppl = ppl_dev_current_epoch
                # Salva lo stato del modello corretto (quello mediato se attivo)
                model_to_save = avg_model if is_averaging else model
                best_model_state_dict = copy.deepcopy(model_to_save.state_dict())
                patience = current_patience # Resetta pazienza
                pbar.set_postfix_str(f"✨Best PPL: {best_dev_ppl:.2f}")
            else:
                patience -= 1
                pbar.set_postfix_str(f"P:{patience}/{current_patience}")
                if patience <= 0:
                    print(f"\nEarly stopping triggered at epoch {epoch}.")
                    break # Interrompi il loop delle epoche
        pbar.close()
        print("--- Training Finished ---")

        # --- 4.6. Salvataggio Dati CSV ---
        if sampled_epochs:
            print(f"Saving training metrics to CSV...")
            results_df = pd.DataFrame(
                {
                    "epoch": sampled_epochs,
                    "train_loss": losses_train,
                    "dev_loss": losses_dev,
                    "dev_perplexity": ppls_dev,
                }
            )
            csv_path = os.path.join(output_dir, "training_metrics.csv")
            results_df.to_csv(csv_path, index=False, float_format="%.4f")
            print(f"Metrics saved to: {csv_path}")

        # --- 4.7. Valutazione Finale e Salvataggio Risultati TXT ---
        final_test_ppl = math.inf
        epochs_trained = 0
        best_epoch_num = "N/A" # Default più informativo
        if sampled_epochs:
            epochs_trained = epoch # Ultima epoca completata o interrotta
            if best_dev_ppl != math.inf:
                try:
                    # Trova l'indice del miglior PPL e poi l'epoca corrispondente
                    best_epoch_index = ppls_dev.index(best_dev_ppl)
                    best_epoch_num = sampled_epochs[best_epoch_index]
                except (ValueError, IndexError):
                    best_epoch_num = "N/A (Error finding index)"

        if best_model_state_dict is not None:
            print("Loading best model state for final evaluation...")
            # Ricarica il modello con la stessa architettura
            final_model_to_eval = LM_LSTM_VarDropWT(
                emb_size=emb,
                hidden_size=hid,
                output_size=vocab_len,
                pad_index=lang.word2id["<pad>"],
                n_layers=1, # Assicurati sia coerente
                dropoute=dropoute_rate,
                dropouto=dropouto_rate,
            ).to(DEVICE)
            final_model_to_eval.load_state_dict(best_model_state_dict)
            final_model_to_eval.eval() # Modalità valutazione

            print("Evaluating on Test Set...")
            with torch.no_grad(): # Disabilita calcolo gradienti per valutazione
                final_test_ppl, _ = eval_loop(
                    test_loader, criterion_eval, final_model_to_eval
                )
            print(f"Final Test PPL: {final_test_ppl:.4f}")

            print("Saving final results to TXT...")
            results_summary = (
                f"Experiment: {experiment_name}\n"
                f"Configuration String: {hyperparam_str}\n\n"
                f"--- Hyperparameters ---\n"
                f"Learning Rate: {lr}\n"
                f"Batch Size: {batch_size}\n"
                f"Patience (Early Stop): {current_patience}\n"
                f"Trigger (NT-AvSGD): {current_trigger}\n"
                f"Embedding Size: {emb}\n"
                f"Hidden Size: {hid}\n"
                f"Dropout Embedding: {dropoute_rate}\n"
                f"Dropout LSTM Out: {dropouto_rate}\n"
                f"Gradient Clipping: {clip}\n\n"
                f"--- Training Summary ---\n"
                f"Optimizer: NT-AvSGD (Base: SGD)\n"
                f"Averaging Triggered: {'Yes (at epoch ' + str(trigger_epoch) + ')' if trigger_epoch != -1 else 'No'}\n"
                f"Total Epochs Run: {epochs_trained}\n"
                f"Best Dev PPL achieved at Epoch: {best_epoch_num}\n\n"
                f"--- Final Performance ---\n"
                f"Best Development PPL: {best_dev_ppl:.4f}\n"
                f"Final Test PPL (using best dev model): {final_test_ppl:.4f}\n"
            )
            results_txt_path = os.path.join(output_dir, "final_results.txt")
            with open(results_txt_path, "w") as f:
                f.write(results_summary)
            print(f"Results saved to: {results_txt_path}")

            # Aggiungi risultati alla lista aggregata
            run_summary_dict = {
                "learning_rate": lr,
                "batch_size": batch_size, # Anche se fisso, utile per coerenza
                "patience_setting": current_patience,
                "trigger_setting": current_trigger,
                "optimizer": f"NT-AvSGD(T={current_trigger})",
                "weight_tying": True,
                "var_dropout_e": dropoute_rate,
                "var_dropout_o": dropouto_rate,
                "avg_triggered": trigger_epoch != -1,
                "trigger_epoch": trigger_epoch if trigger_epoch != -1 else pd.NA, # Usa NA per coerenza DataFrame
                "best_dev_ppl": best_dev_ppl if best_dev_ppl != math.inf else pd.NA,
                "final_test_ppl": final_test_ppl if final_test_ppl != math.inf else pd.NA,
                "epochs_trained": epochs_trained,
                "best_epoch": best_epoch_num if best_epoch_num != "N/A" else pd.NA,
            }
            all_run_results.append(run_summary_dict)
        else:
            print("No best model state saved. Skipping final evaluation and result saving.")
            # Aggiungi placeholder se nessun modello è stato salvato
            run_summary_dict = {
                "learning_rate": lr,
                "batch_size": batch_size,
                "patience_setting": current_patience,
                "trigger_setting": current_trigger,
                "optimizer": f"NT-AvSGD(T={current_trigger})",
                "weight_tying": True,
                "var_dropout_e": dropoute_rate,
                "var_dropout_o": dropouto_rate,
                "avg_triggered": False,
                "trigger_epoch": pd.NA,
                "best_dev_ppl": pd.NA,
                "final_test_ppl": pd.NA,
                "epochs_trained": epochs_trained,
                "best_epoch": pd.NA,
            }
            all_run_results.append(run_summary_dict)


        # --- 4.8. Generazione e Salvataggio Grafici (per singola run) ---
        if sampled_epochs:
            print("Generating and saving plots for this run...")
            try:
                plt.style.use("seaborn-v0_8-darkgrid")
            except OSError:
                print("Seaborn style 'seaborn-v0_8-darkgrid' not found, using default.")

            plot_title_suffix = f"(lr={lr}, bs={batch_size}, P={current_patience}, T={current_trigger})"
            # Loss Plot
            plt.figure(figsize=(10, 6))
            plt.plot(sampled_epochs, losses_train, label="Training Loss", marker=".", linestyle="-", markersize=4)
            plt.plot(sampled_epochs, losses_dev, label="Development Loss", marker=".", linestyle="--", markersize=4)
            if trigger_epoch != -1:
                plt.axvline(x=trigger_epoch, color='r', linestyle='--', linewidth=1, label=f"AvSGD Trigger (Ep {trigger_epoch})")
            plt.xlabel("Epochs")
            plt.ylabel("Loss (Cross Entropy)")
            plt.title(f"Loss vs. Epochs\n{experiment_name} {plot_title_suffix}")
            plt.legend()
            plt.grid(True)
            loss_plot_path = os.path.join(output_dir, "loss_vs_epochs.png")
            plt.savefig(loss_plot_path)
            plt.close()

            # PPL Plot
            plt.figure(figsize=(10, 6))
            plt.plot(sampled_epochs, ppls_dev, label="Dev Perplexity", marker=".", linestyle="-", color="red", markersize=4)
            if trigger_epoch != -1:
                plt.axvline(x=trigger_epoch, color='r', linestyle='--', linewidth=1, label=f"AvSGD Trigger (Ep {trigger_epoch})")
            plt.xlabel("Epochs")
            plt.ylabel("Perplexity (PPL)")
            plt.title(f"Development Perplexity vs. Epochs\n{experiment_name} {plot_title_suffix}")
            # Evidenzia il punto migliore
            if best_dev_ppl != math.inf and best_epoch_num != "N/A" and isinstance(best_epoch_num, int):
                plt.scatter(best_epoch_num, best_dev_ppl, color='blue', s=100, zorder=5, label=f"Best Dev PPL ({best_dev_ppl:.2f} @Ep {best_epoch_num})")
            plt.legend()
            plt.grid(True)
            ppl_plot_path = os.path.join(output_dir, "perplexity_vs_epochs.png")
            plt.savefig(ppl_plot_path)
            plt.close()
            print(f"Plots saved in: {output_dir}")

        # --- 4.9. Stampa Finale Risultati Run e Tempo ---
        print(f"\n--- Summary for Run {current_run}/{total_runs} (lr={lr}) ---")
        final_results_data = [
            ["Averaging Triggered", f"{'Yes (Epoch ' + str(trigger_epoch) + ')' if trigger_epoch != -1 else 'No'}"],
            ["Best Development PPL", f"{best_dev_ppl:.4f}" if best_dev_ppl != math.inf and pd.notna(best_dev_ppl) else "N/A"],
            ["Final Test PPL", f"{final_test_ppl:.4f}" if final_test_ppl != math.inf and pd.notna(final_test_ppl) else "N/A"],
            ["Total Epochs Trained", epochs_trained if epochs_trained > 0 else "N/A"],
            ["Best Dev Epoch", best_epoch_num]
        ]
        print(tabulate(final_results_data, headers=["Metric", "Value"], tablefmt="pretty"))

        # !! Calcolo e stampa tempo !!
        current_time = time.time()
        elapsed_seconds = current_time - start_time_total
        avg_time_per_run = elapsed_seconds / current_run
        runs_remaining = total_runs - current_run
        estimated_remaining_seconds = avg_time_per_run * runs_remaining

        print(f"--- Time Info ---")
        print(f"Elapsed Time: {format_time(elapsed_seconds)}")
        print(f"Avg Time/Run: {format_time(avg_time_per_run)}")
        print(f"Estimated Remaining Time: {format_time(estimated_remaining_seconds)}")
        print(f"===== END OF RUN {current_run}/{total_runs} =====")

    print("\n--- All Runs Finished ---")
    # !! Stampa tempo totale finale !!
    end_time_total = time.time()
    total_elapsed_seconds = end_time_total - start_time_total
    print(f"Total Execution Time: {format_time(total_elapsed_seconds)}")

    # --- 5. Analisi Aggregata e Grafici Comparativi ---
    print(f"\n--- Starting Aggregate Analysis for {experiment_name} ---")
    if not all_run_results:
        print("No results were collected. Skipping aggregate analysis.")
    else:
        summary_df = pd.DataFrame(all_run_results)
        # Imposta opzioni pandas per migliore visualizzazione
        pd.set_option("display.float_format", "{:.4f}".format)
        # Converti inf in NA per sorting e calcoli
        summary_df.replace([math.inf, -math.inf], pd.NA, inplace=True)

        # Salva il DataFrame aggregato
        summary_csv_path = os.path.join(
            main_output_folder, f"summary_results_{experiment_name}.csv"
        )
        # Ordina per PPL Test finale (migliore prima)
        # Converti la colonna in numerico forzando errori a NaN, poi ordina
        summary_df['final_test_ppl'] = pd.to_numeric(summary_df['final_test_ppl'], errors='coerce')
        summary_df.sort_values(by="final_test_ppl", inplace=True, na_position='last')
        summary_df.to_csv(summary_csv_path, index=False, float_format="%.4f")
        print(f"\nSorted summary results saved to: {summary_csv_path}")

        # Stampa i migliori N risultati in tabella
        print("\n--- Aggregate Results Summary (Sorted by Test PPL) ---")
        # Mostra tutte le run dato che sono poche
        print(
            tabulate(
                summary_df.fillna("N/A"), # Riempi NA per stampa pulita
                headers="keys",
                tablefmt="pretty",
                showindex=False,
                floatfmt=".4f",
            )
        )

        # --- MODIFICA: Generazione Grafico Aggregato (Bar Plot LR vs Test PPL) ---
        print("\nGenerating Aggregate Plot (Learning Rate vs. Test PPL)...")
        plot_dir = os.path.join(main_output_folder, "aggregate_plots")
        os.makedirs(plot_dir, exist_ok=True)

        # Filtra eventuali run fallite (NaN in PPL) prima di plottare
        plot_df = summary_df.dropna(subset=['final_test_ppl', 'learning_rate']).copy()
        plot_df.sort_values(by='learning_rate', inplace=True) # Ordina per LR per il plot

        if not plot_df.empty:
            try:
                plt.style.use("seaborn-v0_8-darkgrid")
            except OSError:
                pass # Usa default se stile non trovato

            plt.figure(figsize=(10, 6))
            # Crea le barre
            bars = plt.bar(plot_df['learning_rate'].astype(str), # Usa LR come stringa per categorie distinte
                           plot_df['final_test_ppl'],
                           width=0.6, # Larghezza barre
                           color=sns.color_palette("viridis", len(plot_df))) # Colori diversi

            plt.xlabel("Learning Rate")
            plt.ylabel("Final Test Perplexity (PPL)")
            plt.title(f"Final Test PPL vs. Learning Rate\n(BS={batch_sizes[0]}, P={patience_values[0]}, T={trigger_values[0]}) - {experiment_name}")
            # Imposta limiti Y per migliore visualizzazione
            min_ppl = plot_df['final_test_ppl'].min()
            max_ppl = plot_df['final_test_ppl'].max()
            plt.ylim(bottom=max(0, min_ppl - 10), top=max_ppl + 10) # Aggiusta margini

            # Aggiungi etichette con valore PPL sopra ogni barra
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}", va='bottom', ha='center') # Posiziona testo

            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plot_path = os.path.join(plot_dir, "bar_chart_LR_vs_TestPPL.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved aggregate bar chart to: {plot_path}")
        else:
            print("No valid data to generate aggregate plot (all runs might have failed or produced NaN PPL).")

        print("--- Aggregate Plot Generation Finished ---")
        # --- Fine MODIFICA ---

    print("\n--- Aggregate Analysis Finished ---")
    print(f"--- Experiment {experiment_name} Finished ---")
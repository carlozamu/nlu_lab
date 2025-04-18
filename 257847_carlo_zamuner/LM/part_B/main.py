# Questo script esegue un FINE-TUNING dell'esperimento B3:
# - Usa la configurazione base: LSTM + WT + VarDrop + NT-AvSGD.
# - ITERA su specifici range di LR, BS, Patience e Trigger.
# - Salva risultati (CSV, PNG, TXT) per OGNI combinazione.
# - ALLA FINE, aggrega i risultati e crea grafici comparativi (heatmap multiple).
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
import time # <-- AGGIUNTO

# Ignora warning specifici
warnings.filterwarnings("ignore", message="seaborn gridspec")
warnings.filterwarnings("ignore", message="use_inf_as_na")

# --- Import Moduli Custom ---
from functions import train_loop, eval_loop, init_weights
from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
from model import LM_LSTM_VarDropWT

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

    experiment_name = "puntoB3_FineTuning"
    main_output_folder = os.path.join("results", experiment_name)

    learning_rates = [1.5, 2.0, 3.0, 5.0]
    batch_sizes = [16, 32]
    patience_values = [3, 7]
    trigger_values = [3, 7]

    hid = 300; emb = 300; clip = 5; n_epochs = 100
    dropoute_rate = 0.1; dropouto_rate = 0.4

    # --- 2. Caricamento Dati e Vocabolario ---
    print("\nLoading data and building vocabulary (once)...")
    train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)

    # --- 3. Creazione Dataset ---
    print("Creating Datasets (once)...")
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # --- 4. Loop sugli Iperparametri ---
    print("\n--- Starting Fine-Tuning Loop (LR, BS, Patience, Trigger) ---")
    total_runs = len(learning_rates) * len(batch_sizes) * len(patience_values) * len(trigger_values)
    current_run = 0
    all_run_results = []

    # !! Registra ora di inizio del loop !!
    start_time_total = time.time()

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for current_patience in patience_values:
                for current_trigger in trigger_values:
                    current_run += 1
                    print(f"\n===== RUN {current_run}/{total_runs} =====")
                    print(f"Configuration: lr={lr}, bs={batch_size}, patience={current_patience}, trigger={current_trigger}")

                    # --- 4.1. Configurazione Specifica della Run ---
                    hyperparam_str = f"lr_{lr}_bs_{batch_size}_P{current_patience}_T{current_trigger}_emb_{emb}_hid_{hid}_VarDropWT_e{dropoute_rate}_o{dropouto_rate}"
                    output_dir = os.path.join(main_output_folder, hyperparam_str)
                    os.makedirs(output_dir, exist_ok=True)
                    # print(f"Output directory for this run: {output_dir}") # Rimosso per brevità

                    # --- 4.2. Creazione DataLoader ---
                    # print("Creating DataLoaders...") # Rimosso per brevità
                    train_loader = DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE), drop_last=True)
                    dev_loader = DataLoader(
                        dev_dataset, batch_size=batch_size * 2,
                        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))
                    test_loader = DataLoader(
                        test_dataset, batch_size=batch_size * 2,
                        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE))

                    # --- 4.3. Stampa Configurazione della Run ---
                    # (Codice tabulate omesso per brevità, ma la logica è invariata)
                    # print("\n--- Current Run Configuration ---")
                    # print(tabulate(exp_params, headers=["Parameter", "Value"], tablefmt="pretty"))

                    # --- 4.4. Inizializzazione Modello, Ottimizzatore, Loss ---
                    # print("\nInitializing Model, Optimizer, and Loss...") # Rimosso per brevità
                    model = LM_LSTM_VarDropWT(
                        emb_size=emb, hidden_size=hid, output_size=vocab_len,
                        pad_index=lang.word2id["<pad>"], dropoute=dropoute_rate, dropouto=dropouto_rate
                    ).to(DEVICE)
                    model.apply(init_weights)
                    optimizer = optim.SGD(model.parameters(), lr=lr)
                    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
                    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction="sum")

                    # --- 4.5. Training Loop Specifico della Run con NT-AvSGD ---
                    # print("\n--- Starting Training for this configuration ---") # Rimosso per brevità
                    patience = current_patience
                    avg_trigger_countdown = current_trigger
                    # ... (altre inizializzazioni invariate) ...
                    is_averaging = False; avg_model = None; avg_step_count = 0
                    best_ppl_for_avg_trigger = math.inf; trigger_epoch = -1
                    sampled_epochs = []; losses_train = []; losses_dev = []; ppls_dev = []
                    best_dev_ppl = math.inf; best_model_state_dict = None

                    pbar = tqdm(range(1, n_epochs + 1), desc=f"Run {current_run}/{total_runs} (lr={lr}, bs={batch_size}, P={current_patience}, T={current_trigger})", leave=False)
                    for epoch in pbar:
                        # ... (Logica train_loop, averaging, eval_loop, trigger, early stopping invariata) ...
                        train_loss = train_loop(train_loader, optimizer, criterion_train, model, clip=clip)
                        losses_train.append(train_loss)

                        if is_averaging:
                            avg_step_count += 1
                            with torch.no_grad():
                                for avg_param, param in zip(avg_model.parameters(), model.parameters()):
                                    avg_param.data.add_(param.data.to(avg_param.device) - avg_param.data, alpha=1.0 / avg_step_count)

                        eval_model_ref = avg_model if is_averaging else model
                        eval_model_ref.to(DEVICE)
                        ppl_dev_current_epoch, loss_dev_current_epoch = eval_loop(dev_loader, criterion_eval, eval_model_ref)
                        if is_averaging: eval_model_ref.to("cpu")

                        losses_dev.append(loss_dev_current_epoch); ppls_dev.append(ppl_dev_current_epoch); sampled_epochs.append(epoch)
                        pbar.set_description(f"E{epoch} -> DevPPL:{ppl_dev_current_epoch:.2f}|TrL:{train_loss:.2f}")

                        if ppl_dev_current_epoch < best_ppl_for_avg_trigger:
                            best_ppl_for_avg_trigger = ppl_dev_current_epoch
                            avg_trigger_countdown = current_trigger
                        elif not is_averaging:
                            avg_trigger_countdown -= 1
                            if avg_trigger_countdown <= 0:
                                is_averaging = True; trigger_epoch = epoch; avg_model = copy.deepcopy(model).to("cpu"); avg_step_count = 1

                        if ppl_dev_current_epoch < best_dev_ppl:
                            best_dev_ppl = ppl_dev_current_epoch
                            model_to_save = avg_model if is_averaging else model
                            best_model_state_dict = copy.deepcopy(model_to_save.state_dict())
                            patience = current_patience
                            pbar.set_postfix_str("✨Best")
                        else:
                            patience -= 1
                            pbar.set_postfix_str(f"P:{patience}/{current_patience}")
                            if patience <= 0: break
                    pbar.close()
                    # print("--- Training Finished ---") # Rimosso per brevità

                    # --- 4.6. Salvataggio Dati CSV ---
                    # ... (codice invariato) ...
                    if sampled_epochs:
                        # print(f"Saving training metrics to CSV...") # Rimosso per brevità
                        results_df = pd.DataFrame({'epoch': sampled_epochs, 'train_loss': losses_train, 'dev_loss': losses_dev, 'dev_perplexity': ppls_dev})
                        csv_path = os.path.join(output_dir, "training_metrics.csv")
                        results_df.to_csv(csv_path, index=False)
                        # print(f"Metrics saved to: {csv_path}") # Rimosso per brevità

                    # --- 4.7. Valutazione Finale e Salvataggio Risultati TXT ---
                    # ... (codice invariato per caricare best model e valutare) ...
                    final_test_ppl = math.inf; epochs_trained = 0; best_epoch_num = None; final_model_to_eval = None
                    if sampled_epochs:
                        epochs_trained = sampled_epochs[-1]
                        if best_dev_ppl != math.inf:
                             try: best_epoch_num = sampled_epochs[ppls_dev.index(best_dev_ppl)]
                             except ValueError: best_epoch_num = 'N/A (Error)'

                    if best_model_state_dict is not None:
                        # print("Loading best model state...") # Rimosso per brevità
                        final_model_to_eval = LM_LSTM_VarDropWT(
                            emb_size=emb, hidden_size=hid, output_size=vocab_len, pad_index=lang.word2id["<pad>"],
                            dropoute=dropoute_rate, dropouto=dropouto_rate).to(DEVICE)
                        final_model_to_eval.load_state_dict(best_model_state_dict)
                        final_model_to_eval.eval()
                        # print("Evaluating on Test Set...") # Rimosso per brevità
                        final_test_ppl, _ = eval_loop(test_loader, criterion_eval, final_model_to_eval)
                        # print(f"Final Test PPL: {final_test_ppl:.3f}") # Rimosso per brevità

                        # print("Saving final results to TXT...") # Rimosso per brevità
                        results_summary = ( f"Experiment: {experiment_name}\nConfiguration: {hyperparam_str}\n"
                                            f"Optimizer: NT-AvSGD (Base: SGD, Trigger={current_trigger})\nPatience (Early Stop): {current_patience}\n"
                                            f"Model Features: WT, VarDrop\nDropout Rates (Emb/Out): {dropoute_rate}/{dropouto_rate}\n"
                                            f"Averaging Triggered: {'Yes (epoch ' + str(trigger_epoch) + ')' if trigger_epoch != -1 else 'No'}\n"
                                            f"-------------------------------------\nBest Dev PPL: {best_dev_ppl:.4f}\nFinal Test PPL: {final_test_ppl:.4f}\n"
                                            f"Total Epochs Trained: {epochs_trained}\nBest Dev PPL at Epoch: {best_epoch_num}\n")
                        results_txt_path = os.path.join(output_dir, "final_results.txt")
                        with open(results_txt_path, "w") as f: f.write(results_summary)
                        # print(f"Results saved to: {results_txt_path}") # Rimosso per brevità

                        # Aggiungi risultati alla lista aggregata
                        run_summary_dict = { 'learning_rate': lr, 'batch_size': batch_size, 'patience_setting': current_patience, 'trigger_setting': current_trigger,
                                             'optimizer': f"NT-AvSGD(T={current_trigger})", 'weight_tying': True, 'var_dropout_e': dropoute_rate, 'var_dropout_o': dropouto_rate,
                                             'avg_triggered': trigger_epoch != -1, 'trigger_epoch': trigger_epoch if trigger_epoch != -1 else None,
                                             'best_dev_ppl': best_dev_ppl, 'final_test_ppl': final_test_ppl, 'epochs_trained': epochs_trained, 'best_epoch': best_epoch_num, }
                        all_run_results.append(run_summary_dict)
                    else: # Caso no best model
                         run_summary_dict = { 'learning_rate': lr, 'batch_size': batch_size, 'patience_setting': current_patience, 'trigger_setting': current_trigger,
                                              'optimizer': f"NT-AvSGD(T={current_trigger})", 'weight_tying': True, 'var_dropout_e': dropoute_rate, 'var_dropout_o': dropouto_rate,
                                              'avg_triggered': False, 'trigger_epoch': None, 'best_dev_ppl': math.inf, 'final_test_ppl': math.inf,
                                              'epochs_trained': epochs_trained, 'best_epoch': None, }
                         all_run_results.append(run_summary_dict)


                    # --- 4.8. Generazione e Salvataggio Grafici ---
                    # ... (codice invariato, ma eseguito per ogni run) ...
                    if sampled_epochs:
                        # print("Generating and saving plots...") # Rimosso per brevità
                        try: plt.style.use('seaborn-v0_8-darkgrid')
                        except OSError: pass # Usa default se non trovato

                        plot_title_suffix = f"(Run {current_run}: lr={lr}, bs={batch_size}, P={current_patience}, T={current_trigger})"
                        # Loss Plot
                        plt.figure(figsize=(10, 6))
                        plt.plot(sampled_epochs, losses_train, label='Training Loss', marker='o', linestyle='-'); plt.plot(sampled_epochs, losses_dev, label='Development Loss', marker='x', linestyle='--')
                        if trigger_epoch != -1: plt.axvline(x=trigger_epoch, color='r', linestyle='--', label=f'AvSGD Trigger (Ep {trigger_epoch})')
                        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title(f'Loss\n{experiment_name} {plot_title_suffix}'); plt.legend()
                        loss_plot_path = os.path.join(output_dir, "loss_vs_epochs.png"); plt.savefig(loss_plot_path); plt.close()

                        # PPL Plot
                        plt.figure(figsize=(10, 6))
                        plt.plot(sampled_epochs, ppls_dev, label='Dev PPL', marker='o', linestyle='-', color='red')
                        if trigger_epoch != -1: plt.axvline(x=trigger_epoch, color='r', linestyle='--', label=f'AvSGD Trigger (Ep {trigger_epoch})')
                        plt.xlabel('Epochs'); plt.ylabel('Perplexity (PPL)'); plt.title(f'Perplexity\n{experiment_name} {plot_title_suffix}')
                        if best_dev_ppl != math.inf and best_epoch_num != 'N/A (Error)': plt.scatter(best_epoch_num, best_dev_ppl, color='blue', s=100, zorder=5, label=f'Best ({best_dev_ppl:.2f} @Ep {best_epoch_num})')
                        plt.legend(); ppl_plot_path = os.path.join(output_dir, "perplexity_vs_epochs.png"); plt.savefig(ppl_plot_path); plt.close()

                    # --- 4.9. Stampa Finale Risultati Run e Tempo ---
                    print(f"\n--- Summary for Run {current_run}/{total_runs} ---")
                    # ... (stampa tabulate invariata) ...
                    final_results_data = [ ["Averaging Triggered", f"{'Yes (Epoch ' + str(trigger_epoch) + ')' if trigger_epoch != -1 else 'No'}"],
                                           ["Best Development PPL", f"{best_dev_ppl:.4f}" if best_dev_ppl != math.inf else "N/A"],
                                           ["Final Test PPL", f"{final_test_ppl:.4f}" if final_test_ppl != math.inf else "N/A"],
                                           ["Total Epochs Trained", epochs_trained if epochs_trained > 0 else "N/A"], ]
                    print(tabulate(final_results_data, headers=["Metric", "Value"], tablefmt="pretty"))

                    # !! Calcolo e stampa tempo !!
                    current_time = time.time()
                    elapsed_seconds = current_time - start_time_total
                    avg_time_per_run = elapsed_seconds / current_run
                    runs_remaining = total_runs - current_run
                    estimated_remaining_seconds = avg_time_per_run * runs_remaining

                    print(f"--- Time Info ---")
                    print(f"Elapsed Time: {format_time(elapsed_seconds)}")
                    print(f"Avg Time/Run: {avg_time_per_run:.2f} s")
                    print(f"Estimated Remaining Time: {format_time(estimated_remaining_seconds)}")
                    print(f"===== END OF RUN {current_run}/{total_runs} =====")


    print("\n--- All Fine-Tuning Runs Finished ---")
    # !! Stampa tempo totale finale !!
    end_time_total = time.time()
    total_elapsed_seconds = end_time_total - start_time_total
    print(f"Total Execution Time: {format_time(total_elapsed_seconds)}")


    # --- 5. Analisi Aggregata e Grafici Comparativi ---
    print(f"\n--- Starting Aggregate Analysis for {experiment_name} ---")
    # ... (Codice per DataFrame, salvataggio CSV, stampa top N invariato) ...
    if not all_run_results:
        print("No results. Skipping aggregate analysis.")
    else:
        summary_df = pd.DataFrame(all_run_results)
        pd.set_option('display.float_format', '{:.4f}'.format)
        summary_df.replace([math.inf, -math.inf], pd.NA, inplace=True)
        summary_csv_path = os.path.join(main_output_folder, f"summary_results_{experiment_name}.csv")
        summary_df.sort_values(by='final_test_ppl', inplace=True)
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
        print(f"\nSorted summary results saved to: {summary_csv_path}")
        print("\n--- Aggregate Results Summary (Top 15) ---")
        print(tabulate(summary_df.head(15).fillna("Failed"), headers="keys", tablefmt="pretty", showindex=False, floatfmt=".4f"))

        # --- Generazione Heatmap Multiple (LR vs BS per ogni P/T) ---
        print("\nGenerating Heatmaps (LR vs BS for each Patience/Trigger combo)...")
        heatmap_dir = os.path.join(main_output_folder, "heatmaps_LR_vs_BS")
        os.makedirs(heatmap_dir, exist_ok=True)
        # !! Usa i valori testati per iterare !!
        tested_patiences = summary_df['patience_setting'].unique()
        tested_triggers = summary_df['trigger_setting'].unique()

        for p_val in tested_patiences:
            for t_val in tested_triggers:
                df_subset = summary_df[(summary_df['patience_setting'] == p_val) & (summary_df['trigger_setting'] == t_val)]
                if df_subset.empty: continue

                try: # Heatmap Test PPL
                    heatmap_data_test = df_subset.pivot(index="learning_rate", columns="batch_size", values="final_test_ppl")
                    heatmap_data_test = heatmap_data_test.astype(float)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(heatmap_data_test, annot=True, fmt=".2f", cmap="viridis_r", linewidths=0.5, cbar_kws={'label': 'Final Test PPL'})
                    plt.title(f"Final Test PPL Heatmap\n(Patience={p_val}, Trigger={t_val}) - {experiment_name}")
                    plt.xlabel("Batch Size"); plt.ylabel("Learning Rate")
                    heatmap_test_path = os.path.join(heatmap_dir, f"heatmap_TestPPL_P{p_val}_T{t_val}.png")
                    plt.tight_layout(); plt.savefig(heatmap_test_path); plt.close()
                    # print(f"Saved heatmap: {heatmap_test_path}") # Rimosso per brevità
                except Exception as e: print(f"Error heatmap Test PPL P={p_val}, T={t_val}: {e}")

        print("--- Heatmap Generation Finished ---")

    print("\n--- Aggregate Analysis Finished ---")
    print(f"--- Experiment {experiment_name} Run Finished ---")
# --- START OF FILE main.py ---

# Questo file è usato per eseguire le tue funzioni e stampare i risultati
# Modificato per eseguire una GRID SEARCH su Dropout Rate, Learning Rate, Batch Size
# per il modello BERT fine-tunato (Parte 2.B - Grid Search)
# e generare HEATMAPS

import os
import csv
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm  # Importato per la barra di progresso

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from collections import Counter

from tabulate import tabulate  # Importato per la tabella finale

# Import per Heatmap
import pandas as pd
import seaborn as sns

# Importa tutto da utils.py
from utils import *

# Importa il nuovo modello BERT
from model import JointBERTModel

# Importa lo script di valutazione conll
try:
    from conll import evaluate
except ImportError:
    print("Errore: assicurati che il file 'conll.py' sia presente nella directory.")
    exit()

# Importa il tokenizer di BERT Fast
from transformers import BertTokenizerFast


# --- Funzioni di Training e Valutazione ADATTATE per BERT (invariate) ---
# train_loop e eval_loop rimangono le stesse (senza tqdm interno)


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=1.0):
    """Esegue un'epoca di addestramento per il modello BERT."""
    model.train()
    total_loss = 0.0
    # Rimuoviamo tqdm dai batch per output più pulito nella grid search
    for batch in data:
        optimizer.zero_grad()
        intent_logits, slot_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )
        loss_intent = criterion_intents(intent_logits, batch["intent_labels"])
        loss_slot = criterion_slots(slot_logits.permute(0, 2, 1), batch["slot_labels"])
        loss = loss_intent + loss_slot
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return total_loss / len(data) if len(data) > 0 else float("inf")


def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    """Esegue la valutazione per il modello BERT, gestendo allineamento slot."""
    model.eval()
    total_loss = 0.0
    ref_intents, hyp_intents = [], []
    ref_slots_list, hyp_slots_list = [], []
    ignore_index = IGNORE_INDEX
    with torch.no_grad():
        # Rimuoviamo tqdm dai batch
        for batch in data:
            intent_logits, slot_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )
            loss_intent = criterion_intents(intent_logits, batch["intent_labels"])
            loss_slot = criterion_slots(
                slot_logits.permute(0, 2, 1), batch["slot_labels"]
            )
            loss = loss_intent + loss_slot
            total_loss += loss.item()
            # Intent Inference
            out_intent_ids = torch.argmax(intent_logits, dim=1).cpu().tolist()
            hyp_intents.extend(
                [lang.id2intent.get(idx, "UNK_INTENT") for idx in out_intent_ids]
            )
            ref_intents.extend(
                [
                    lang.id2intent.get(idx, "UNK_INTENT")
                    for idx in batch["intent_labels"].cpu().tolist()
                ]
            )

            # Slot Inference e Allineamento Sub-token per CoNLL
            predicted_slot_ids = torch.argmax(slot_logits, dim=-1).cpu().tolist()
            for i in range(batch["input_ids"].size(0)):
                word_id_sequence = batch["word_ids"][i]
                attention_mask_sequence = batch["attention_mask"][i].cpu().tolist()
                original_slot_labels_list = batch["original_slot_labels"][i]
                original_words_list = batch["original_words"][i]
                predicted_slot_id_sequence = predicted_slot_ids[i]
                current_pred_slots_labels_aligned = []
                last_word_id = None
                for token_idx, word_id in enumerate(word_id_sequence):
                    if word_id is not None and attention_mask_sequence[token_idx] == 1:
                        is_first_subtoken = (
                            token_idx == 0 or word_id_sequence[token_idx - 1] != word_id
                        )
                        if is_first_subtoken:
                            if word_id < len(original_slot_labels_list):
                                predicted_id = predicted_slot_id_sequence[token_idx]
                                predicted_slot_label = lang.id2slot.get(
                                    predicted_id, "UNK_SLOT_PRED"
                                )
                                current_pred_slots_labels_aligned.append(
                                    predicted_slot_label
                                )
                            else:
                                pass
                if len(original_words_list) == len(original_slot_labels_list) and len(
                    original_words_list
                ) == len(current_pred_slots_labels_aligned):
                    ref_slots_list.append(
                        list(zip(original_words_list, original_slot_labels_list))
                    )
                    hyp_slots_list.append(
                        list(
                            zip(original_words_list, current_pred_slots_labels_aligned)
                        )
                    )
                else:
                    # Stampa warning meno frequentemente se necessario
                    # tqdm.write(f"\nWarning: Mismatch in lengths for conll eval in batch {i}.")
                    min_len = min(
                        len(original_words_list),
                        len(original_slot_labels_list),
                        len(current_pred_slots_labels_aligned),
                    )
                    ref_slots_list.append(
                        list(
                            zip(
                                original_words_list[:min_len],
                                original_slot_labels_list[:min_len],
                            )
                        )
                    )
                    hyp_slots_list.append(
                        list(
                            zip(
                                original_words_list[:min_len],
                                current_pred_slots_labels_aligned[:min_len],
                            )
                        )
                    )

    # --- Calcolo Metriche ---
    try:
        results = evaluate(ref_slots_list, hyp_slots_list)
        slot_f1 = results.get("total", {}).get("f", 0.0)
    except Exception as ex:
        tqdm.write(f"\nAttenzione: Errore durante la valutazione conll: {ex}")
        slot_f1 = 0.0
    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=0, output_dict=True
    )
    intent_acc = report_intent.get("accuracy", 0.0)
    mean_loss = total_loss / len(data) if len(data) > 0 else float("inf")
    return slot_f1, intent_acc, mean_loss


# --- Funzioni per la gestione dei risultati e delle cartelle (invariate) ---
def create_run_output_dir(base_results_dir, step_name, run_params_str):
    run_dir = os.path.join(base_results_dir, step_name, run_params_str)
    os.makedirs(run_dir, exist_ok=True)
    # Non stampiamo qui, facciamo nel loop principale
    return run_dir


def save_epoch_history(run_output_dir, history_data):
    csv_filepath = os.path.join(run_output_dir, "epoch_history.csv")
    if not history_data:
        return
    fieldnames = history_data[0].keys()
    with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_data)


def save_test_results(run_output_dir, test_f1, test_acc, test_loss):
    csv_filepath = os.path.join(run_output_dir, "test_results.csv")
    fieldnames = ["Test_Slot_F1", "Test_Intent_Acc", "Test_Loss"]
    with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "Test_Slot_F1": test_f1,
                "Test_Intent_Acc": test_acc,
                "Test_Loss": test_loss,
            }
        )


# --- Funzione per generare e salvare Heatmap (invariata) ---
def save_heatmap(
    df,
    index_col,
    columns_col,
    value_col,
    title,
    filename,
    run_output_dir_base,
    cmap="viridis",
    fmt=".4f",
):
    try:
        pivot_table = pd.pivot_table(
            df, values=value_col, index=index_col, columns=columns_col, aggfunc=np.mean
        )
        fig_width = max(6, len(pivot_table.columns) * 1.5 + 3)
        fig_height = max(4, len(pivot_table.index) * 0.8 + 2)
        plt.figure(figsize=(fig_width, fig_height))
        cmap_val = "viridis" if "F1" in metric or "Acc" in metric else "YlGnBu_r"
        sns.heatmap(pivot_table, annot=True, fmt=fmt, cmap=cmap_val, linewidths=0.5)
        plt.title(
            f"{metric} vs {idx_param} & {col_param}\n(Aggregato su {agg_param})",
            fontsize=12,
        )
        plt.xlabel(col_param, fontsize=10)
        plt.ylabel(idx_param, fontsize=10)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        filepath = os.path.join(run_output_dir_base, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"  Heatmap salvata: {filepath}")
    except Exception as e:
        print(
            f"  Errore durante la generazione della heatmap per {metric} ({idx_param} vs {col_param}): {e}"
        )


# --- Blocco Principale di Esecuzione ---
if __name__ == "__main__":
    # --- 1. Caricamento e Preparazione Dati ---
    # ... (Codice di caricamento, split, Lang, Tokenizer come prima) ...
    print("Caricamento dati ATIS...")
    data_dir = os.path.join("dataset", "ATIS")
    train_raw_all = load_data(os.path.join(data_dir, "train.json"))
    test_raw = load_data(os.path.join(data_dir, "test.json"))
    print(f"Campioni Train caricati (prima dello split): {len(train_raw_all)}")
    print(f"Campioni Test caricati: {len(test_raw)}")
    print("Creazione Dev set...")
    portion = 0.10
    intents_list_all = [x["intent"] for x in train_raw_all]
    inputs_split, labels_split, mini_train_split = [], [], []
    count_y_split = Counter(intents_list_all)
    for i, intent in enumerate(intents_list_all):
        if count_y_split[intent] > 1:
            inputs_split.append(train_raw_all[i])
            labels_split.append(intent)
        else:
            mini_train_split.append(train_raw_all[i])
    X_train_split, X_dev_split, _, _ = train_test_split(
        inputs_split,
        labels_split,
        test_size=portion,
        random_state=42,
        shuffle=True,
        stratify=labels_split,
    )
    train_raw = X_train_split
    train_raw.extend(mini_train_split)
    dev_raw = X_dev_split
    print(
        f"Dimensioni finali (dopo split): Train={len(train_raw)}, Dev={len(dev_raw)}, Test={len(test_raw)}"
    )
    corpus_all = train_raw + dev_raw + test_raw
    all_slots_corpus = set(sum([line["slots"].split() for line in corpus_all], []))
    all_intents_corpus = set([line["intent"] for line in corpus_all])
    lang = Lang(all_intents_corpus, all_slots_corpus)
    print(f"Slot: {len(lang.slot2id)} etichette")
    print(f"Intenti: {len(lang.intent2id)} etichette")
    print("\nInizializzazione BertTokenizer...")
    tokenizer_name = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    print(f"BertTokenizer type: {type(tokenizer)}")
    print(f"BertTokenizer vocab size: {len(tokenizer)}")
    print(f"BertTokenizer [PAD] token ID: {tokenizer.pad_token_id}")

    # ===========================================================
    # --- SEZIONE PARAMETRI DI FINE-TUNING PER GRID SEARCH ---
    # ===========================================================

    # --- Parametri Fissi Generali ---
    bert_model_name = tokenizer_name  # O 'bert-large-uncased'
    hid_size = (
        768 if "base" in bert_model_name else 1024
    )  # Non usato direttamente nel modello, ma per info
    emb_size = 768 if "base" in bert_model_name else 1024  # Come sopra
    clip = 1.0
    n_epochs = 50  # Numero massimo di epoche per *ogni* run della grid search
    patience = 5  # Pazienza più bassa per grid search più veloce, o più alta (10) per più stabilità
    max_seq_length = 50
    batch_size_eval_fixed = 128  # Batch size fisso per valutazione

    # --- Liste di Parametri da Testare nella Grid Search ---
    # ATTENZIONE: Il numero totale di run sarà len(dropout)*len(lr)*len(batch)
    dropout_rates_to_test = [0.5]  # Esempi, puoi modificare
    learning_rates_to_test = [2e-5, 3e-5, 5e-5]  # Valori comuni per BERT fine-tuning
    batch_sizes_to_test = [32, 64, 128]  # Batch size per il training

    # --- Parametri per Nomi Cartelle e File ---
    base_results_directory = "results"
    step_directory_name = "step_bert_grid_search"  # Nome per questo step di grid search

    # ===========================================================
    # --- FINE SEZIONE PARAMETRI GRID SEARCH ---
    # ===========================================================

    # --- Creazione Dataset (fuori dai loop, non dipende dai parametri di training) ---
    print("Creazione Dataset (BERT compatible)...")
    train_dataset = IntentsAndSlotsBERT(
        train_raw, lang, tokenizer, max_len=max_seq_length
    )
    dev_dataset = IntentsAndSlotsBERT(dev_raw, lang, tokenizer, max_len=max_seq_length)
    test_dataset = IntentsAndSlotsBERT(
        test_raw, lang, tokenizer, max_len=max_seq_length
    )
    print("Dataset pronti.")

    # --- Calcolo dimensioni output e definizione loss ---
    num_intent_labels = len(lang.intent2id)
    num_slot_labels = len(lang.slot2id)
    criterion_intents = nn.CrossEntropyLoss()
    criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # --- 6. Inizio Grid Search ---
    all_runs_test_results = []  # Lista per raccogliere i risultati di TUTTE le combinazioni
    total_runs = (
        len(dropout_rates_to_test)
        * len(learning_rates_to_test)
        * len(batch_sizes_to_test)
    )
    print(f"\nInizio Grid Search per {bert_model_name}.")
    print(f"Parametri da testare:")
    print(f"  - Dropout Rates: {dropout_rates_to_test}")
    print(f"  - Learning Rates: {learning_rates_to_test}")
    print(f"  - Batch Sizes: {batch_sizes_to_test}")
    print(f"Numero totale di combinazioni (run): {total_runs}")
    print("-" * 30)

    # Loop esterno principale per la Grid Search
    run_counter = 0

    for dropout_rate in dropout_rates_to_test:
        for lr in learning_rates_to_test:
            for batch_size_train in batch_sizes_to_test:
                run_counter += 1
                print(
                    f"\n--- RUN {run_counter}/{total_runs}: DR={dropout_rate}, LR={lr}, BS={batch_size_train} ---"
                )

                # --- Configurazione Specifica per questa Run ---
                config_name = f"BERT_DR{dropout_rate}_LR{lr}_BS{batch_size_train}"
                # Nome cartella più conciso
                params_string = f"dr{str(dropout_rate).replace('.', '')}_lr{str(lr)}_bs{batch_size_train}"

                run_output_directory = create_run_output_dir(
                    base_results_directory, step_directory_name, params_string
                )
                tqdm.write(f"Output directory: {run_output_directory}")

                # --- Creazione DataLoader DINAMICA ---
                current_train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size_train,
                    collate_fn=collate_fn_bert,
                    shuffle=True,
                )
                dev_loader = DataLoader(
                    dev_dataset,
                    batch_size=batch_size_eval_fixed,
                    collate_fn=collate_fn_bert,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size_eval_fixed,
                    collate_fn=collate_fn_bert,
                )
                tqdm.write("DataLoader pronti per questa run.")

                # --- Addestramento e Valutazione ---
                tqdm.write(f"Esecuzione Configurazione: {config_name}")

                model = JointBERTModel(
                    num_intent_labels,
                    num_slot_labels,
                    bert_model_name=bert_model_name,
                    dropout_rate=dropout_rate,
                )
                model.to(device)
                # Non reinizializziamo BERT, solo i layer lineari se necessario (ma l'init di default è ok)
                optimizer = optim.Adam(model.parameters(), lr=lr)

                epoch_history_data = []
                best_f1_dev, current_patience_count, best_epoch_num = 0.0, patience, 0
                best_model_state_dict = None

                tqdm.write(f"Inizio addestramento per max {n_epochs} epoche...")
                epoch_iterator = tqdm(
                    range(1, n_epochs + 1),
                    desc=f"Run {run_counter}/{total_runs}",
                    leave=False,
                )

                try:
                    for epoch in epoch_iterator:
                        loss_train_epoch = train_loop(
                            current_train_loader,
                            optimizer,
                            criterion_slots,
                            criterion_intents,
                            model,
                            clip,
                        )
                        f1_dev_epoch, acc_dev_epoch, loss_dev_epoch = eval_loop(
                            dev_loader,
                            criterion_slots,
                            criterion_intents,
                            model,
                            lang,
                            tokenizer,
                        )
                        epoch_iterator.set_postfix(
                            {
                                "TrainLoss": f"{loss_train_epoch:.4f}",
                                "DevLoss": f"{loss_dev_epoch:.4f}",
                                "DevF1": f"{f1_dev_epoch:.4f}",
                                "DevAcc": f"{acc_dev_epoch:.4f}",
                            }
                        )
                        epoch_history_data.append(
                            {
                                "Epoch": epoch,
                                "Train_Loss": loss_train_epoch,
                                "Dev_Loss": loss_dev_epoch,
                                "Dev_Slot_F1": f1_dev_epoch,
                                "Dev_Intent_Acc": acc_dev_epoch,
                            }
                        )

                        if f1_dev_epoch > best_f1_dev:
                            best_f1_dev = f1_dev_epoch
                            best_epoch_num = epoch
                            current_patience_count = patience
                            tqdm.write(
                                f"  (Run {run_counter}) Epoca {epoch}: Nuovo miglior F1 Dev = {best_f1_dev:.4f}. Salvo stato."
                            )
                            best_model_state_dict = model.state_dict().copy()
                            torch.save(
                                best_model_state_dict,
                                os.path.join(run_output_directory, "best_model.pth"),
                            )
                        else:
                            current_patience_count -= 1
                        if current_patience_count <= 0:
                            tqdm.write(
                                f"  (Run {run_counter}) Early stopping attivato all'epoca {epoch}."
                            )
                            break
                except Exception as e:
                    tqdm.write(
                        f"\nERRORE CRITICO durante addestramento run {run_counter} ({config_name}): {e}"
                    )
                    try:
                        if model:
                            torch.save(
                                model.state_dict(),
                                os.path.join(
                                    run_output_directory, "last_state_on_error.pth"
                                ),
                            )
                    except Exception as save_e:
                        tqdm.write(f"Errore salvataggio stato: {save_e}")
                    f1_test, acc_test, loss_test = 0.0, 0.0, float("inf")
                    best_model_state_dict = None
                    continue  # Passa al prossimo run
                finally:
                    epoch_iterator.close()

                tqdm.write(f"Addestramento {config_name} completato.")
                if best_model_state_dict is None and epoch_history_data:
                    tqdm.write(
                        "  Attenzione: Nessun miglioramento F1 Dev trovato. Valuto con l'ultimo stato completato."
                    )
                    best_model_state_dict = model.state_dict().copy()
                    torch.save(
                        best_model_state_dict,
                        os.path.join(run_output_directory, "last_state_valid.pth"),
                    )
                elif best_model_state_dict is None and not epoch_history_data:
                    tqdm.write(
                        "  Attenzione: Addestramento fallito o interrotto subito. Impossibile eseguire valutazione finale."
                    )
                    # Salta valutazione e salvataggio, aggiungi risultati vuoti
                    all_runs_test_results.append(
                        {
                            "Config": config_name,
                            "Dropout Rate": dropout_rate,
                            "Learning Rate": lr,
                            "Batch Size Train": batch_size_train,
                            "Test Slot F1": 0.0,
                            "Test Intent Acc": 0.0,
                            "Test Loss": float("inf"),
                        }
                    )
                    del model, optimizer
                    torch.cuda.empty_cache() if device.type == "cuda" else None
                    torch.mps.empty_cache() if device.type == "mps" else None
                    continue

                # Valutazione Finale sul Test Set
                tqdm.write(f"Valutazione {config_name} sul Test set finale...")
                try:
                    # Ricarica modello per valutazione
                    model_final_eval = JointBERTModel(
                        num_intent_labels,
                        num_slot_labels,
                        bert_model_name=bert_model_name,
                        dropout_rate=dropout_rate,
                    )
                    model_final_eval.load_state_dict(best_model_state_dict)
                    model_final_eval.to(device)
                    f1_test, acc_test, loss_test = eval_loop(
                        test_loader,
                        criterion_slots,
                        criterion_intents,
                        model_final_eval,
                        lang,
                        tokenizer,
                    )
                    del model_final_eval  # Libera memoria subito
                except Exception as e:
                    tqdm.write(f"\nERRORE durante la valutazione finale: {e}")
                    f1_test, acc_test, loss_test = 0.0, 0.0, float("inf")

                tqdm.write(
                    f"  Risultati Test - {config_name}: F1={f1_test:.4f}, Acc={acc_test:.4f}, Loss={loss_test:.4f}"
                )

                # Salvataggio Risultati e Plot
                tqdm.write("Salvataggio storia, risultati finali e grafici...")
                save_epoch_history(run_output_directory, epoch_history_data)
                save_test_results(run_output_directory, f1_test, acc_test, loss_test)
                # ... (Codice plot invariato - verrà eseguito solo se epoch_history_data non è vuoto) ...
                if epoch_history_data:
                    # Plot Loss
                    plt.figure(figsize=(10, 6))
                    epochs_plot = [d["Epoch"] for d in epoch_history_data]
                    train_losses_plot = [d["Train_Loss"] for d in epoch_history_data]
                    dev_losses_plot = [d["Dev_Loss"] for d in epoch_history_data]
                    plt.plot(epochs_plot, train_losses_plot, label="Train Loss")
                    plt.plot(epochs_plot, dev_losses_plot, label="Dev Loss")
                    plt.title(f"Andamento Loss - {config_name}")
                    plt.xlabel("Epoche")
                    plt.ylabel("Loss Media")
                    if best_epoch_num > 0:
                        plt.axvline(
                            x=best_epoch_num,
                            color="r",
                            linestyle="--",
                            label=f"Best Dev F1 Epoch ({best_epoch_num})",
                        )
                    plt.legend()
                    plt.grid(True)
                    loss_plot_filepath = os.path.join(
                        run_output_directory, f"loss_plot.png"
                    )
                    plt.savefig(loss_plot_filepath)
                    plt.close()
                    tqdm.write(f"  Grafico loss salvato: {loss_plot_filepath}")
                    # Plot Metriche
                    plt.figure(figsize=(10, 6))
                    dev_f1s_plot = [d["Dev_Slot_F1"] for d in epoch_history_data]
                    dev_accs_plot = [d["Dev_Intent_Acc"] for d in epoch_history_data]
                    plt.plot(epochs_plot, dev_f1s_plot, label="Dev Slot F1")
                    plt.plot(epochs_plot, dev_accs_plot, label="Dev Intent Accuracy")
                    plt.title(f"Andamento Metriche Dev - {config_name}")
                    plt.xlabel("Epoche")
                    plt.ylabel("Score")
                    if best_epoch_num > 0:
                        plt.axvline(
                            x=best_epoch_num,
                            color="r",
                            linestyle="--",
                            label=f"Best Dev F1 Epoch ({best_epoch_num})",
                        )
                    plt.legend()
                    plt.grid(True)
                    metrics_plot_filepath = os.path.join(
                        run_output_directory, f"metrics_plot.png"
                    )
                    plt.savefig(metrics_plot_filepath)
                    plt.close()
                    tqdm.write(f"  Grafico metriche salvato: {metrics_plot_filepath}")

                # Raccogli i risultati finali di questa run
                all_runs_test_results.append(
                    {
                        "Config": config_name,
                        "Dropout Rate": dropout_rate,
                        "Learning Rate": lr,
                        "Batch Size Train": batch_size_train,
                        "Test Slot F1": f1_test,
                        "Test Intent Acc": acc_test,
                        "Test Loss": loss_test,
                    }
                )

                # Pulisce memoria
                del (
                    model,
                    optimizer,
                    best_model_state_dict,
                    epoch_history_data,
                    current_train_loader,
                    dev_loader,
                    test_loader,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()

    # --- 11. Riepilogo Comparativo Finale e HEATMAPS ---
    print(
        "\n--- Riepilogo Comparativo Risultati Test (Grid Search BERT Fine-tuning) ---"
    )

    df_results = pd.DataFrame(all_runs_test_results)
    print(tabulate(df_results, headers="keys", tablefmt="grid", showindex=False))

    summary_csv_path = os.path.join(
        base_results_directory, step_directory_name, "grid_search_summary.csv"
    )
    df_results.to_csv(summary_csv_path, index=False)
    print(f"\nRiepilogo completo salvato in: {summary_csv_path}")

    # --- Generazione Heatmaps ---
    print("\nGenerazione Heatmaps (valori medi dove applicabile)...")
    metrics_to_plot = ["Test Slot F1", "Test Intent Acc", "Test Loss"]
    param_pairs = [
        ("Dropout Rate", "Learning Rate", "Batch Size Train"),
        ("Dropout Rate", "Batch Size Train", "Learning Rate"),
        ("Learning Rate", "Batch Size Train", "Dropout Rate"),
    ]
    heatmap_dir = os.path.join(base_results_directory, step_directory_name, "_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    for metric in metrics_to_plot:
        metric_safe_name = metric.replace(" ", "_")
        for idx_param, col_param, agg_param in param_pairs:
            if (
                idx_param in df_results.columns
                and col_param in df_results.columns
                and agg_param in df_results.columns
            ):
                save_heatmap(
                    df_results,
                    idx_param,
                    col_param,
                    metric,
                    f"{metric} vs {idx_param} & {col_param}\n(Aggregato su {agg_param})",
                    f"heatmap_{metric_safe_name}_vs_{idx_param.replace(' ', '')}_vs_{col_param.replace(' ', '')}.png",
                    heatmap_dir,
                )
            else:
                print(
                    f"  Skipping heatmap for {metric} ({idx_param} vs {col_param}): Colonne non trovate nel DataFrame."
                )

    print("\nGrid Search per BERT Fine-tuning completata.")

# --- END OF FILE main.py ---

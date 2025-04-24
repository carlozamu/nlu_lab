# --- START OF FILE main.py ---

# Questo file è usato per eseguire le tue funzioni e stampare i risultati
# Implementazione per la Parte 2.B: Fine-tuning BERT-large per Joint Intent & Slot
# Esegue un singolo run con parametri fissi

import os
import csv
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm # Importato per la barra di progresso

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from collections import Counter

from tabulate import tabulate # Importato per la tabella finale

# Non necessari per un singolo run BERT
# import pandas as pd
# import seaborn as sns

# Importa tutto da utils.py (data loading, Lang, IntentsAndSlotsBERT, collate_fn_bert, device, IGNORE_INDEX)
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
# train_loop e eval_loop rimangono le stesse


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=1.0):
    """Esegue un'epoca di addestramento per il modello BERT."""
    model.train()
    total_loss = 0.0
    for batch in tqdm(data, desc="Training Batches", leave=False):
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
        for batch in tqdm(data, desc="Evaluation Batches", leave=False):
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
                    tqdm.write(
                        f"\nWarning: Mismatch in lengths for conll eval in batch {i}."
                    )
                    tqdm.write(
                        f"  Original words: {len(original_words_list)}, Ref labels: {len(original_slot_labels_list)}, Pred labels (aligned): {len(current_pred_slots_labels_aligned)}"
                    )
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


# --- Funzioni per la gestione dei risultati e delle cartelle (simili a prima) ---
def create_run_output_dir(base_results_dir, step_name, run_params_str):
    run_dir = os.path.join(base_results_dir, step_name, run_params_str)
    os.makedirs(run_dir, exist_ok=True)
    tqdm.write(f"Risultati per questa run verranno salvati in: {run_dir}")
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


# --- Blocco Principale di Esecuzione ---
if __name__ == "__main__":
    # --- 1. Caricamento e Preparazione Dati ---
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

    # --- 2. Inizializzazione Tokenizer BERT ---
    print("\nInizializzazione BertTokenizer...")
    tokenizer_name = "bert-base-uncased" # Modificheremo questo per BERT-large
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    print(f"BertTokenizer type: {type(tokenizer)}")
    print(f"BertTokenizer vocab size: {len(tokenizer)}")
    print(f"BertTokenizer [PAD] token ID: {tokenizer.pad_token_id}")


    # ===========================================================
    # --- SEZIONE PARAMETRI DI FINE-TUNING MODIFICABILI ---
    # --- Configurazione per BERT-large (Singolo Run) ---
    # ===========================================================

    # --- Parametri Generali ---
    n_epochs = 100           # Numero massimo di epoche
    patience = 10           # Pazienza per l'early stopping (basato su Dev F1)
    clip = 1.0              # Valore per gradient clipping (tipico 1.0 per BERT)
    max_seq_length = 50     # Lunghezza massima sequenza tokenizzata BERT (come nel paper)

    # --- Parametri Specifici del Modello (BERT-large) e Ottimizzatore ---
    bert_model_name = 'bert-large-uncased' # <-- MODIFICA QUI per BERT-large
    # Learning Rate tipico per BERT-large può essere leggermente più basso del base
    lr_fixed = 5e-5  # Puoi provare 2e-5, 3e-5, 5e-5 (5e-5 era nel paper per base)
    bert_dropout_rate_fixed = 0.1 # Tasso di Dropout (come nel paper per base)
    # Batch size per BERT-large può essere più piccolo se la memoria è limitata
    batch_size_train_fixed = 64 # Iniziamo con 64 per BERT-large per gestire memoria
    batch_size_eval_fixed = 64  # Usa lo stesso per valutazione

    # --- Parametri per Nomi Cartelle e File ---
    base_results_directory = "results"
    # Nome specifico per il run di BERT-large con parametri fissi
    step_directory_name = "step_bert_large_finetuning_run1"

    # ===========================================================
    # --- FINE SEZIONE PARAMETRI MODIFICABILI ---
    # ===========================================================


    # --- 3. Creazione Dataset e DataLoader (ADATTATI per BERT) ---
    # Usa i parametri definiti sopra
    print("Creazione Dataset e DataLoader (BERT compatible)...")
    # Il tokenizer deve corrispondere al modello usato (bert-large-uncased usa lo stesso tokenizer)
    # Quindi ri-inizializziamo il tokenizer Fast qui per sicurezza, usando il nome del modello LARGE
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name) # <-- Re-inizializza tokenizer con nome modello LARGE
    print(f"BertTokenizer type (after model name change): {type(tokenizer)}")
    print(f"BertTokenizer vocab size (after model name change): {len(tokenizer)}")


    train_dataset = IntentsAndSlotsBERT(train_raw, lang, tokenizer, max_len=max_seq_length)
    dev_dataset = IntentsAndSlotsBERT(dev_raw, lang, tokenizer, max_len=max_seq_length)
    test_dataset = IntentsAndSlotsBERT(test_raw, lang, tokenizer, max_len=max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train_fixed, collate_fn=collate_fn_bert, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size_eval_fixed, collate_fn=collate_fn_bert)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval_fixed, collate_fn=collate_fn_bert)

    print("DataLoader BERT-compatible pronti.")
    print(f"Lunghezza massima sequenza: {max_seq_length}")
    print(f"Batch size (Train/Eval): {batch_size_train_fixed}/{batch_size_eval_fixed}")


    # --- 4. Definizione Modello, Ottimizzatore, Loss (ADATTATI per BERT) ---
    # Usa i parametri definiti sopra
    print("\nInizializzazione Modello BERT...")

    num_intent_labels = len(lang.intent2id)
    num_slot_labels = len(lang.slot2id)

    # Istanzia il modello BERT-large
    model = JointBERTModel(num_intent_labels, num_slot_labels, bert_model_name=bert_model_name, dropout_rate=bert_dropout_rate_fixed)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr_fixed)

    criterion_intents = nn.CrossEntropyLoss()
    criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)


    print("Modello BERT inizializzato.")
    print(f"Modello BERT Name: {bert_model_name}")
    print(f"Hyperparameters: LR={lr_fixed}, Dropout={bert_dropout_rate_fixed}, BatchSize={batch_size_train_fixed}, MaxLen={max_seq_length}, Patience={patience}, Clip={clip}")


    # --- Crea la cartella di output per questo run ---
    params_string = f"model_{bert_model_name.split('/')[-1]}_lr{str(lr_fixed).replace('.', '')}_dr{str(bert_dropout_rate_fixed).replace('.', '')}_bs{batch_size_train_fixed}_maxlen{max_seq_length}"
    run_output_directory = create_run_output_dir(base_results_directory, step_directory_name, params_string)


    # --- 6. Addestramento e Valutazione (BERT) ---
    print(f"\n--- Inizio Fine-tuning BERT-large ---")

    epoch_history_data = []

    best_f1_dev, current_patience_count, best_epoch_num = 0.0, patience, 0
    best_model_state_dict = None


    tqdm.write(f"Inizio fine-tuning per max {n_epochs} epoche...")
    epoch_iterator = tqdm(range(1, n_epochs + 1), desc=f"BERT-large Fine-tuning", leave=True)

    try:
        for epoch in epoch_iterator:
            loss_train_epoch = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip)

            f1_dev_epoch, acc_dev_epoch, loss_dev_epoch = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang, tokenizer)

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

            # Early Stopping Logic (basata su Dev F1)
            if f1_dev_epoch > best_f1_dev:
                best_f1_dev = f1_dev_epoch
                best_epoch_num = epoch
                current_patience_count = patience
                tqdm.write(
                    f"\n*** Epoca {epoch}: Nuovo miglior F1 Dev = {best_f1_dev:.4f}. Salvato stato modello. ***"
                )
                best_model_state_dict = model.state_dict().copy()
                torch.save(best_model_state_dict, os.path.join(run_output_directory, "best_model.pth"))
            else:
                current_patience_count -= 1
                if current_patience_count > 0 and current_patience_count < patience:
                     if (patience - current_patience_count) % max(1, patience // 3) == 0 or current_patience_count == 1:
                         tqdm.write(
                            f"Epoca {epoch}: Nessun miglioramento F1 Dev. Pazienza rimasta: {current_patience_count}"
                        )

            if current_patience_count <= 0:
                tqdm.write(f"--- Early stopping attivato all'epoca {epoch}. ---")
                break

    except Exception as e:
        tqdm.write(f"\nERRORE CRITICO durante fine-tuning BERT: {e}")
        try:
             if model:
                  torch.save(model.state_dict(), os.path.join(run_output_directory, "last_state_on_error.pth"))
        except Exception as save_e:
             tqdm.write(f"Errore durante salvataggio stato in caso di errore: {save_e}")
        f1_test, acc_test, loss_test = 0.0, 0.0, float('inf')
        best_model_state_dict = None

    finally:
         epoch_iterator.close()

    if best_model_state_dict is None and epoch_history_data:
         tqdm.write("Attenzione: Nessun miglioramento F1 Dev trovato. Valuto con l'ultimo stato completato.")
         best_model_state_dict = model.state_dict().copy()
         torch.save(best_model_state_dict, os.path.join(run_output_directory, "last_state_valid.pth"))
    elif best_model_state_dict is None and not epoch_history_data:
         tqdm.write("Attenzione: Addestramento fallito o interrotto subito. Impossibile eseguire valutazione finale.")


    # --- 7. Valutazione Finale sul Test Set ---
    if best_model_state_dict:
        print(f"\nValutazione finale sul Test set...")
        try:
            # Ricarica il modello pulito con gli stessi parametri usati per il training
            model_final_eval = JointBERTModel(num_intent_labels, num_slot_labels, bert_model_name=bert_model_name, dropout_rate=bert_dropout_rate_fixed)
            model_final_eval.load_state_dict(best_model_state_dict)
            model_final_eval.to(device)
            f1_test, acc_test, loss_test = eval_loop(test_loader, criterion_slots, criterion_intents, model_final_eval, lang, tokenizer)
            del model_final_eval

        except Exception as e:
            print(f"\nERRORE durante la valutazione finale sul Test set: {e}")
            f1_test, acc_test, loss_test = 0.0, 0.0, float('inf')


    else:
         print("\nSkipping final test set evaluation due to failed training or no improvement.")
         f1_test, acc_test, loss_test = 0.0, 0.0, float('inf')


    print("\n--- Risultati Finali sul Test Set ---")
    config_name_final = f"BERT_{bert_model_name.split('/')[-1]}_DR{bert_dropout_rate_fixed}_LR{lr_fixed}_BS{batch_size_train_fixed}_MaxLen{max_seq_length}"
    print(f"Configurazione: {config_name_final}")
    test_results_data = [
        ["Slot Filling F1", f1_test],
        ["Intent Classification Accuracy", acc_test],
        ["Loss Media Test", loss_test]
    ]
    print(tabulate(test_results_data, headers=["Metrica", "Valore"], tablefmt="grid"))

    # --- 8. Salvataggio Risultati e Plot ---
    print("\nSalvataggio storia, risultati finali e grafici...")

    if best_model_state_dict or epoch_history_data:
         save_epoch_history(run_output_directory, epoch_history_data)
         save_test_results(run_output_directory, f1_test, acc_test, loss_test)
    else:
         print("Nessun dato valido da salvare.")

    if epoch_history_data:
        plt.figure(figsize=(10, 6))
        epochs_plot = [d["Epoch"] for d in epoch_history_data]
        train_losses_plot = [d["Train_Loss"] for d in epoch_history_data]
        dev_losses_plot = [d["Dev_Loss"] for d in epoch_history_data]
        plt.plot(epochs_plot, train_losses_plot, label="Train Loss")
        plt.plot(epochs_plot, dev_losses_plot, label="Dev Loss")
        plt.title(f"Andamento Loss - {config_name_final}")
        plt.xlabel("Epoche")
        plt.ylabel("Loss Media (Cross Entropy)")
        if best_epoch_num > 0:
            plt.axvline(x=best_epoch_num,color="r",linestyle="--",label=f"Best Dev F1 Epoch ({best_epoch_num})")
        plt.legend(); plt.grid(True)
        loss_plot_filepath = os.path.join(run_output_directory, f"loss_plot.png")
        plt.savefig(loss_plot_filepath); plt.close()
        print(f"Grafico loss salvato in: {loss_plot_filepath}")

        plt.figure(figsize=(10, 6))
        dev_f1s_plot = [d["Dev_Slot_F1"] for d in epoch_history_data]
        dev_accs_plot = [d["Dev_Intent_Acc"] for d in epoch_history_data]
        plt.plot(epochs_plot, dev_f1s_plot, label="Dev Slot F1")
        plt.plot(epochs_plot, dev_accs_plot, label="Dev Intent Accuracy")
        plt.title(f"Andamento Metriche Dev - {config_name_final}")
        plt.xlabel("Epoche"); plt.ylabel("Score")
        if best_epoch_num > 0:
            plt.axvline(x=best_epoch_num,color="r",linestyle="--",label=f"Best Dev F1 Epoch ({best_epoch_num})")
        plt.legend(); plt.grid(True)
        metrics_plot_filepath = os.path.join(run_output_directory, f"metrics_plot.png")
        plt.savefig(metrics_plot_filepath); plt.close()
        print(f"Grafico metriche salvato in: {metrics_plot_filepath}")
    else:
        print("Nessun dato storico per generare i grafici.")

    # Pulizia finale della memoria del modello istanziato per la valutazione finale
    # (del model_final_eval è già stato fatto)

    print("\nEsecuzione Parte 2.B completata.")

# --- END OF FILE main.py ---
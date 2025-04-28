# --- START OF FILE main.py ---

# Questo file è usato per eseguire le tue funzioni e stampare i risultati
# Implementazione minimalista per la Parte 2.A: Addestramento, Valutazione Test e Salvataggio dei 3 modelli LSTM (.pth)
# Tutti i modelli .pth vengono salvati in una singola cartella results/bin/

import os
# Rimosso import csv
# Rimosso import pprint
# Rimosso import matplotlib.pyplot as plt
from tqdm import tqdm # Importato per la barra di progresso

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from collections import Counter

from tabulate import tabulate # Importato per la tabella finale

# Rimosso import pandas e seaborn

# Importa tutto da utils.py (data loading, Lang, IntentsAndSlots, collate_fn, device, PAD_TOKEN)
from utils import *

# Importa il modello LSTM configurabile
from model import ModelIAS

# Importa lo script di valutazione conll
try:
    from conll import evaluate
except ImportError:
    print("Errore: assicurati che il file 'conll.py' sia presente nella directory.")
    exit()


# --- Funzioni di Training e Valutazione (senza tqdm interno - invariate) ---
# train_loop e eval_loop rimangono le stesse


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    """Esegue un'epoca di addestramento."""
    model.train()
    total_loss = 0.0
    for sample in data:
        optimizer.zero_grad()
        slots, intent = model(sample["utterances"], sample["slots_len"])
        loss_intent = criterion_intents(intent, sample["intents"])
        loss_slot = criterion_slots(slots, sample["y_slots"])
        loss = loss_intent + loss_slot
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return total_loss / len(data) if len(data) > 0 else float("inf")

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """Esegue la valutazione."""
    model.eval()
    total_loss = 0.0
    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], [] # Liste per accumulare seq di slot (formato conll)

    with torch.no_grad():
        for sample in data:
            slots, intents = model(sample["utterances"], sample["slots_len"])
            loss_intent = criterion_intents(intents, sample["intents"])
            loss_slot = criterion_slots(slots, sample["y_slots"])
            loss = loss_intent + loss_slot
            total_loss += loss.item()
            # Intent Inference
            out_intents_ids = torch.argmax(intents, dim=1).cpu().tolist()
            hyp_intents.extend(
                [lang.id2intent.get(idx, "UNK_INTENT") for idx in out_intents_ids]
            )
            ref_intents.extend(
                [
                    lang.id2intent.get(idx, "UNK_INTENT")
                    for idx in sample["intents"].cpu().tolist()
                ]
            )
            # Slot Inference
            output_slots_ids = torch.argmax(slots, dim=1).cpu().tolist()
            for id_seq in range(len(output_slots_ids)):
                length = sample["slots_len"][id_seq].item()
                utt_ids = sample['utterances'][id_seq][:length].cpu().tolist()
                utterance_words = [lang.id2word.get(idx, lang.word2id.get('unk', 'UNK')) for idx in utt_ids]
                gt_ids = sample["y_slots"][id_seq][:length].cpu().tolist()
                gt_slots_labels = [lang.id2slot.get(idx, lang.slot2id.get("pad", "UNK_SLOT")) for idx in gt_ids]
                pred_ids = output_slots_ids[id_seq][:length]
                pred_slots_labels = [lang.id2slot.get(idx, lang.slot2id.get("pad", "UNK_SLOT")) for idx in pred_ids]
                ref_slots.append(list(zip(utterance_words, gt_slots_labels)))
                hyp_slots.append(list(zip(utterance_words, pred_slots_labels)))

    # --- Calcolo Metriche ---
    try:
        results = evaluate(ref_slots, hyp_slots)
        slot_f1 = results.get("total", {}).get("f", 0.0)
    except Exception as ex:
        tqdm.write(f"\nAttenzione: Errore durante la valutazione conll: {ex}")
        slot_f1 = 0.0 # Fallback

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=0, output_dict=True
    )
    intent_acc = report_intent.get("accuracy", 0.0)
    mean_loss = total_loss / len(data) if len(data) > 0 else float('inf')
    return slot_f1, intent_acc, mean_loss


# --- Funzione per creare la directory bin (unica) ---
def create_bin_dir(base_results_dir):
    bin_dir = os.path.join(base_results_dir, 'bin')
    os.makedirs(bin_dir, exist_ok=True)
    return bin_dir


# Rimosse funzioni create_run_output_dir, save_epoch_history, save_test_results, save_heatmap


# --- Blocco Principale di Esecuzione ---
if __name__ == "__main__":

    # --- 1. Caricamento e Preparazione Dati (LSTM compatible) ---
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

    print(f"Dimensioni finali (dopo split e gestione rari): Train={len(train_raw)}, Dev={len(dev_raw)}, Test={len(test_raw)}")

    # Creazione vocabolario PER LSTM
    words_train = sum([x['utterance'].split() for x in train_raw], [])
    corpus_all = train_raw + dev_raw + test_raw
    all_slots_corpus = set(sum([line["slots"].split() for line in corpus_all], []))
    all_intents_corpus = set([line["intent"] for line in corpus_all])
    lang = Lang(words_train, all_intents_corpus, all_slots_corpus, cutoff=0)
    print(f"Vocabolario (LSTM): {len(lang.word2id)} parole")
    print(f"Slot: {len(lang.slot2id)} etichette")
    print(f"Intenti: {len(lang.intent2id)} etichette")


    # --- 2. Creazione Dataset e DataLoader (LSTM Compatible) ---
    print("Creazione Dataset e DataLoader (LSTM compatible)...")
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Batch size eval fisso come nel notebook originale
    batch_size_eval_fixed = 64


    # --- 3. Definizione Iperparametri e Configurazioni per Parte 2.A ---
    # Parametri standard del notebook 05
    hid_size = 200
    emb_size = 300
    clip = 5
    # Nostri parametri di addestramento più robusti con early stopping
    n_epochs = 50 # Epoche max per early stopping
    patience = 10 # Pazienza early stopping

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()


    # Definisci le 3 configurazioni LSTM da eseguire con i loro parametri "migliori"
    lstm_configs = [
        {
            "name": "A1_Baseline_LSTM",
            "lr": 0.005,
            "batch_size_train": 128,
            "bidirectional": False,
            "dropout_rate": 0.0,
        },
         {
            "name": "A2_BiLSTM",
            "lr": 0.001,
            "batch_size_train": 64,
            "bidirectional": True,
            "dropout_rate": 0.0,
        },
        {
            "name": "A3_BiLSTM_Dropout",
            "lr": 0.005,
            "batch_size_train": 128,
            "bidirectional": True,
            "dropout_rate": 0.5,
        }
    ]

    base_results_directory = "results"
    # Crea la sottocartella bin all'inizio, fuori dal loop
    bin_output_directory = create_bin_dir(base_results_directory)
    print(f"Modelli .pth verranno salvati in: {bin_output_directory}")


    all_runs_test_results = [] # Per riepilogo finale

    # --- 4. Ciclo sulle Configurazioni LSTM della Parte 2.A ---
    for config in lstm_configs:
        config_name = config["name"]
        lr = config["lr"]
        batch_size_train = config["batch_size_train"]
        bidirectional = config["bidirectional"]
        dropout_rate = config["dropout_rate"]

        print(f"\n--- Esecuzione Configurazione: {config_name} ---")
        print(f"Hyperparameters: LR={lr}, Dropout={dropout_rate}, BatchSize={batch_size_train}, Bidirectional={bidirectional}, Patience={patience}, Clip={clip}")


        # --- Creazione DataLoader per la configurazione corrente ---
        train_loader = DataLoader(train_dataset, batch_size=batch_size_train, collate_fn=collate_fn, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size_eval_fixed, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size_eval_fixed, collate_fn=collate_fn)
        print("DataLoader pronti per questa run.")


        # --- 5. Definizione Modello, Ottimizzatore ---
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN,
                         bidirectional=bidirectional, dropout_rate=dropout_rate)
        model.to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # --- 6. Addestramento e Valutazione ---
        epoch_history_data = [] # Manteniamo questo solo per l'early stopping logic
        best_f1_dev, current_patience_count, best_epoch_num = 0.0, patience, 0
        best_model_state_dict = None # Resetta per ogni run


        tqdm.write(f"Inizio addestramento per max {n_epochs} epoche...")
        epoch_iterator = tqdm(range(1, n_epochs + 1), desc=f"{config_name} Training", leave=True)

        try:
            for epoch in epoch_iterator:
                loss_train_epoch = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip)
                f1_dev_epoch, acc_dev_epoch, loss_dev_epoch = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)

                epoch_iterator.set_postfix(
                    {
                        "TrainLoss": f"{loss_train_epoch:.4f}",
                        "DevLoss": f"{loss_dev_epoch:.4f}",
                        "DevF1": f"{f1_dev_epoch:.4f}",
                        "DevAcc": f"{acc_dev_epoch:.4f}",
                    }
                )

                # Rimuoviamo l'aggiunta a epoch_history_data se non ci serve più per grafici/csv

                # Early Stopping Logic (basata su Dev F1)
                if f1_dev_epoch > best_f1_dev:
                    best_f1_dev = f1_dev_epoch
                    best_epoch_num = epoch
                    current_patience_count = patience
                    # tqdm.write(f"\n*** Epoca {epoch}: Nuovo miglior F1 Dev = {best_f1_dev:.4f}. Salvato stato modello. ***") # Messaggio opzionale
                    best_model_state_dict = model.state_dict().copy()
                    # SALVA il modello migliore NELLA CARTELLA BIN COMUNE
                    # Il nome del file include il nome della configurazione e i parametri chiave
                    model_filename = f"{config_name}_lr{str(lr).replace('.', '')}_dr{str(dropout_rate).replace('.', '')}_bs{batch_size_train}_best.pth"
                    torch.save(best_model_state_dict, os.path.join(bin_output_directory, model_filename))

                else:
                    current_patience_count -= 1
                    if current_patience_count <= 3 and current_patience_count > 0: # Stampa meno frequentemente gli stati di pazienza bassa
                         if (patience - current_patience_count) % max(1, patience // 3) == 0 or current_patience_count == 1:
                              tqdm.write(f"Epoca {epoch}: Nessun miglioramento F1 Dev. Pazienza rimasta: {current_patience_count}")


                if current_patience_count <= 0:
                    tqdm.write(f"--- Early stopping attivato all'epoca {epoch}. ---")
                    break

        except Exception as e:
            tqdm.write(f"\nERRORE CRITICO durante addestramento {config_name}: {e}")
            # Salva lo stato attuale del modello in caso di errore, se esiste
            try:
                 if model:
                      error_model_filename = f"{config_name}_lr{str(lr).replace('.', '')}_dr{str(dropout_rate).replace('.', '')}_bs{batch_size_train}_error.pth"
                      torch.save(model.state_dict(), os.path.join(bin_output_directory, error_model_filename))
                      tqdm.write(f"Stato modello salvato in caso di errore: {error_model_filename}")
            except Exception as save_e:
                 tqdm.write(f"Errore durante salvataggio stato in caso di errore: {save_e}")
            # Imposta risultati a 0/inf in caso di errore critico
            f1_test, acc_test, loss_test = 0.0, 0.0, float('inf')
            best_model_state_dict = None # Non usare uno stato potenzialmente corrotto

        finally:
             epoch_iterator.close()


        tqdm.write(f"\nAddestramento {config_name} completato.")
        # Verifica se è stato trovato un modello migliore valido durante l'addestramento riuscito
        # Se best_model_state_dict è ancora None, significa che non c'è stato nessun miglioramento
        # E l'early stopping non ha salvato. Salviamo l'ultimo stato valido.
        if best_model_state_dict is None:
             tqdm.write("Attenzione: Nessun miglioramento F1 Dev trovato. Salvando l'ultimo stato valido.")
             best_model_state_dict = model.state_dict().copy()
             last_valid_model_filename = f"{config_name}_lr{str(lr).replace('.', '')}_dr{str(dropout_rate).replace('.', '')}_bs{batch_size_train}_lastvalid.pth"
             torch.save(best_model_state_dict, os.path.join(bin_output_directory, last_valid_model_filename))


        # --- 7. Valutazione Finale sul Test Set ---
        # Valutazione finale con il modello migliore (o l'ultimo valido salvato sopra)
        print(f"\nValutazione finale sul Test set per {config_name}...")
        try:
            # Ricarica il modello pulito con gli stessi parametri usati per il training
            model_final_eval = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN,
                                         bidirectional=bidirectional, dropout_rate=dropout_rate)
            model_final_eval.load_state_dict(best_model_state_dict)
            model_final_eval.to(device)
            f1_test, acc_test, loss_test = eval_loop(test_loader, criterion_slots, criterion_intents, model_final_eval, lang)
            del model_final_eval # Pulisci dopo l'uso

        except Exception as e:
            print(f"\nERRORE durante la valutazione finale: {e}")
            f1_test, acc_test, loss_test = 0.0, 0.0, float('inf') # Imposta risultati a 0 in caso di errore


        # --- 8. Stampa Risultati Finali ---
        print(f"\n--- Risultati Finali sul Test Set per {config_name} ---")
        test_results_data = [
            ["Slot Filling F1", f1_test],
            ["Intent Classification Accuracy", acc_test],
            ["Loss Media Test", loss_test]
        ]
        print(tabulate(test_results_data, headers=["Metrica", "Valore"], tablefmt="grid"))


        # Raccogli i risultati finali per il riepilogo totale
        all_runs_test_results.append({
            "Config": config_name,
            "Learning Rate": lr,
            "Dropout Rate": dropout_rate,
            "Batch Size": batch_size_train,
            "Bidirectional": bidirectional,
            "Test Slot F1": f1_test,
            "Test Intent Acc": acc_test,
            "Test Loss": loss_test
        })

        # Pulisce memoria dei dataloader (dopo che sono stati usati per la run)
        del train_loader, dev_loader, test_loader
        # torch.cuda.empty_cache() if device.type == "cuda" else None # Pulizia esplicita se usi GPU
        # torch.mps.empty_cache() if device.type == "mps" else None # Pulizia esplicita se usi MPS


    # --- 9. Riepilogo Comparativo Finale con Tabulate ---
    print("\n--- Riepilogo Comparativo Risultati Test (Parte 2.A LSTM) ---")

    table_data = []
    for run_results in all_runs_test_results:
        table_data.append([
            run_results["Config"],
            run_results["Learning Rate"],
            run_results["Dropout Rate"],
            run_results["Batch Size"],
            run_results["Bidirectional"],
            f"{run_results['Test Slot F1']:.4f}",
            f"{run_results['Test Intent Acc']:.4f}",
            f"{run_results['Test Loss']:.4f}"
        ])

    headers = [
        "Config", "LR", "DR", "BS", "Bi", "Test Slot F1", "Test Intent Acc", "Test Loss"
    ]

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print("\nEsecuzione di tutti i run LSTM della Parte 2.A completata.")

# --- END OF FILE main.py ---
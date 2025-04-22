# --- START OF FILE main.py ---

# Questo file è usato per eseguire le tue funzioni e stampare i risultati

import os
import csv
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from collections import Counter

# Importa tabulate
from tabulate import tabulate

# Importa tutto da utils.py
from utils import *

# Importa il modello BASELINE da model.py
from model import ModelIAS

# Importa lo script di valutazione conll
try:
    from conll import evaluate
except ImportError:
    print("Errore: assicurati che il file 'conll.py' sia presente nella directory.")
    exit()


# --- Funzioni di Training e Valutazione (invariate) ---

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    """Esegue un'epoca di addestramento."""
    model.train()
    loss_array = []
    for sample in tqdm(data, desc="Training Progress"):
        optimizer.zero_grad()
        slots, intent = model(sample["utterances"], sample["slots_len"])
        loss_intent = criterion_intents(intent, sample["intents"])
        loss_slot = criterion_slots(slots, sample["y_slots"])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return np.mean(loss_array)

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """Esegue la valutazione sul dataset fornito (dev o test)."""
    model.eval()
    loss_array = []
    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []
    with torch.no_grad():
        for sample in tqdm(data, desc="Evaluation Progress"):
            slots, intents = model(sample["utterances"], sample["slots_len"])
            loss_intent = criterion_intents(intents, sample["intents"])
            loss_slot = criterion_slots(slots, sample["y_slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
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
            output_slots_ids = torch.argmax(slots, dim=1)
            for id_seq in range(output_slots_ids.size(0)):
                length = sample["slots_len"][id_seq].item()
                utt_ids = sample["utterances"][id_seq][:length].cpu().tolist()
                utterance = [
                    lang.id2word.get(idx, lang.word2id.get("unk", "UNK"))
                    for idx in utt_ids
                ]
                gt_ids = sample["y_slots"][id_seq][:length].cpu().tolist()
                gt_slots_labels = [
                    lang.id2slot.get(idx, lang.slot2id.get("pad", "UNK_SLOT"))
                    for idx in gt_ids
                ]
                pred_ids = output_slots_ids[id_seq][:length].cpu().tolist()
                pred_slots_labels = [
                    lang.id2slot.get(idx, lang.slot2id.get("pad", "UNK_SLOT"))
                    for idx in pred_ids
                ]
                ref_slots.append(list(zip(utterance, gt_slots_labels)))
                hyp_slots.append(list(zip(utterance, pred_slots_labels)))
    # Calcolo Metriche Slot Filling (conll)
    try:
        results = evaluate(ref_slots, hyp_slots)
        slot_f1 = results.get("total", {}).get("f", 0.0)
    except Exception as ex:
        print(f"\nAttenzione: Errore durante la valutazione conll: {ex}")
        slot_f1 = 0.0
    # Calcolo Metriche Intent Classification (sklearn)
    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=0, output_dict=True
    )
    intent_acc = report_intent.get("accuracy", 0.0)
    return slot_f1, intent_acc, np.mean(loss_array)


# --- Funzione per creare le cartelle di output ---
def create_run_output_dir(base_results_dir, step_name, run_params_str):
    """Crea la struttura di directory per salvare i risultati di una run."""
    run_dir = os.path.join(base_results_dir, step_name, run_params_str)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Risultati per questa run verranno salvati in: {run_dir}")
    return run_dir

# --- Funzione per salvare la storia delle metriche per epoca ---
def save_epoch_history(run_output_dir, history_data):
    """Salva la lista di dizionari history_data in un file CSV."""
    csv_filepath = os.path.join(run_output_dir, 'epoch_history.csv')
    fieldnames = ['Epoch', 'Train_Loss', 'Dev_Loss', 'Dev_Slot_F1', 'Dev_Intent_Acc']
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data_row in history_data:
            writer.writerow(data_row)

# --- Funzione per salvare i risultati finali del test set ---
def save_test_results(run_output_dir, test_f1, test_acc, test_loss):
    """Salva i risultati finali del test set in un file CSV."""
    csv_filepath = os.path.join(run_output_dir, 'test_results.csv')
    fieldnames = ['Test_Slot_F1', 'Test_Intent_Acc', 'Test_Loss']
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Test_Slot_F1': test_f1, 'Test_Intent_Acc': test_acc, 'Test_Loss': test_loss})

# --- Blocco Principale di Esecuzione ---
if __name__ == "__main__":

    # --- 1. Caricamento e Preparazione Dati (invariato) ---
    print("Caricamento dati ATIS...")
    data_dir = os.path.join("dataset", "ATIS")
    train_raw = load_data(os.path.join(data_dir, "train.json"))
    test_raw = load_data(os.path.join(data_dir, "test.json"))
    print(f"Campioni Train caricati: {len(train_raw)}")
    print(f"Campioni Test caricati: {len(test_raw)}")
    print("Creazione Dev set...")
    portion = 0.10
    intents_list = [x["intent"] for x in train_raw]
    count_y = Counter(intents_list)
    labels, inputs, mini_train = [], [], []
    for i, intent in enumerate(intents_list):
        if count_y[intent] > 1:
            inputs.append(train_raw[i])
            labels.append(intent)
        else:
            mini_train.append(train_raw[i])
    X_train, X_dev, _, _ = train_test_split(
        inputs,
        labels,
        test_size=portion,
        random_state=42,
        shuffle=True,
        stratify=labels,
    )
    X_train.extend(mini_train)
    train_raw, dev_raw = X_train, X_dev
    print(
        f"Dimensioni finali: Train={len(train_raw)}, Dev={len(dev_raw)}, Test={len(test_raw)}"
    )
    print("Creazione vocabolari...")
    words_train = sum([x["utterance"].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    all_slots = set(sum([line["slots"].split() for line in corpus], []))
    all_intents = set([line["intent"] for line in corpus])
    lang = Lang(words_train, all_intents, all_slots, cutoff=0)
    print(f"Vocabolario: {len(lang.word2id)} parole")
    print(f"Slot: {len(lang.slot2id)} etichette")
    print(f"Intenti: {len(lang.intent2id)} etichette")
    print("Creazione Dataset e DataLoader...")
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)
    batch_size_train, batch_size_eval = 128, 64
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_train, collate_fn=collate_fn, shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size_eval, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size_eval, collate_fn=collate_fn
    )
    print("DataLoader pronti.")

    # --- 5. Definizione Iperparametri ---
    # Parametri per la configurazione BASELINE
    hid_size = 200
    emb_size = 300
    lr = 0.001
    clip = 5
    n_epochs = 50
    patience = 5
    # Parametri che definiscono l'architettura (per il nome della cartella)
    # Mantienili qui anche se non usati direttamente per la baseline ModelIAS
    # Saranno utili per generare il nome della cartella
    bidirectional = False
    dropout_rate = 0.0


    out_slot, out_int, vocab_len = (
        len(lang.slot2id),
        len(lang.intent2id),
        len(lang.word2id),
    )
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    # --- Configurazione Specifica per questa Run (Baseline) ---
    config_name = "BASELINE"
    # Crea una stringa descrittiva dei parametri per il nome della cartella
    # Includi solo i parametri rilevanti per questa run, ma puoi aggiungere altri in futuro
    params_string = f"hid{hid_size}_emb{emb_size}_lr{lr}_p{patience}_bs{batch_size_train}"

    # Crea le cartelle di output per questa specifica run
    base_results_directory = "results"
    step_directory_name = "step_baseline" # Nome specifico per lo step 2.A - Baseline
    run_output_directory = create_run_output_dir(base_results_directory, step_directory_name, params_string)


    # --- 6. Addestramento e Valutazione (Modificato per salvare storia) ---
    print(f"\n--- Esecuzione Configurazione: {config_name} ---")

    # Istanzia il modello BASELINE
    model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN)
    model.to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_history_data = [] # Lista per salvare i dati per la history CSV

    best_f1_dev, current_patience_count, best_epoch_num = 0.0, patience, 0
    best_model_state_dict = None

    print(f"Inizio addestramento {config_name} per max {n_epochs} epoche...")
    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoca {epoch}/{n_epochs}")

        # Training loop
        loss_train_epoch = train_loop(
            train_loader, optimizer, criterion_slots, criterion_intents, model, clip
        )
        print(f"Loss Addestramento Media: {loss_train_epoch:.4f}")

        # Evaluation loop on Dev set
        f1_dev_epoch, acc_dev_epoch, loss_dev_epoch = eval_loop(
            dev_loader, criterion_slots, criterion_intents, model, lang
        )
        print(
            f"Risultati Dev: Loss={loss_dev_epoch:.4f}, Slot F1={f1_dev_epoch:.4f}, Intent Acc={acc_dev_epoch:.4f}"
        )

        # Salva i dati dell'epoca per la history
        epoch_history_data.append({
            'Epoch': epoch,
            'Train_Loss': loss_train_epoch,
            'Dev_Loss': loss_dev_epoch,
            'Dev_Slot_F1': f1_dev_epoch,
            'Dev_Intent_Acc': acc_dev_epoch
        })

        # Early Stopping Logic
        if f1_dev_epoch > best_f1_dev:
            best_f1_dev = f1_dev_epoch
            best_epoch_num = epoch
            current_patience_count = patience
            print(f"*** Nuovo miglior F1 Dev! Salvo stato modello (Epoca {epoch}). ***")
            best_model_state_dict = model.state_dict().copy()
            torch.save(best_model_state_dict, os.path.join(run_output_directory, 'best_model.pth'))
        else:
            current_patience_count -= 1
            print(f"Nessun miglioramento F1 Dev. Pazienza rimasta: {current_patience_count}")

        if current_patience_count <= 0:
            print(f"--- Early stopping attivato all'epoca {epoch}. ---")
            break

    print(f"\nAddestramento {config_name} completato.")
    if best_model_state_dict:
         print(f"Miglior F1 Slot su Dev set: {best_f1_dev:.4f} ottenuto all'epoca {best_epoch_num}")
    else:
         print("Attenzione: Nessun modello migliore salvato (addestramento interrotto presto o nessun miglioramento). Valuto con l'ultimo stato.")
         best_model_state_dict = model.state_dict().copy()


    # --- 7. Valutazione Finale sul Test Set ---
    print(f"\nValutazione {config_name} sul Test set...")
    if best_model_state_dict:
        print(f"Carico il modello migliore (o l'ultimo stato) per la valutazione sul Test set.")
        model.load_state_dict(best_model_state_dict)

    f1_test, acc_test, loss_test = eval_loop(
        test_loader, criterion_slots, criterion_intents, model, lang
    )

    # --- 8. Salvataggio Risultati e Plot ---
    print("\nSalvataggio storia, risultati finali e grafici...")

    # Salva la storia delle metriche (Train/Dev) per epoca
    save_epoch_history(run_output_directory, epoch_history_data)

    # Salva i risultati finali del Test set
    save_test_results(run_output_directory, f1_test, acc_test, loss_test)

    # Genera e salva il grafico delle Loss
    plt.figure(figsize=(10, 6))
    epochs_plot = [d['Epoch'] for d in epoch_history_data]
    train_losses_plot = [d['Train_Loss'] for d in epoch_history_data]
    dev_losses_plot = [d['Dev_Loss'] for d in epoch_history_data]

    plt.plot(epochs_plot, train_losses_plot, label='Train Loss')
    plt.plot(epochs_plot, dev_losses_plot, label='Dev Loss')
    plt.title(f'Andamento Loss - {config_name}')
    plt.xlabel('Epoche')
    plt.ylabel('Loss Media (Cross Entropy)')
    if best_epoch_num > 0:
        plt.axvline(
            x=best_epoch_num,
            color="r",
            linestyle="--",
            label=f"Best Dev F1 Epoch ({best_epoch_num})",
        )
    plt.legend()
    plt.grid(True)
    loss_plot_filepath = os.path.join(run_output_directory, f"loss_plot.png")
    plt.savefig(loss_plot_filepath)
    print(f"Grafico loss salvato in: {loss_plot_filepath}")
    plt.close()

    # Genera e salva il grafico delle Metriche (F1 e Accuracy)
    plt.figure(figsize=(10, 6))
    dev_f1s_plot = [d['Dev_Slot_F1'] for d in epoch_history_data]
    dev_accs_plot = [d['Dev_Intent_Acc'] for d in epoch_history_data]

    plt.plot(epochs_plot, dev_f1s_plot, label='Dev Slot F1')
    plt.plot(epochs_plot, dev_accs_plot, label='Dev Intent Accuracy')
    plt.title(f'Andamento Metriche Dev - {config_name}')
    plt.xlabel('Epoche')
    plt.ylabel('Score')
    if best_epoch_num > 0:
        plt.axvline(
            x=best_epoch_num,
            color="r",
            linestyle="--",
            label=f"Best Dev F1 Epoch ({best_epoch_num})",
        )
    plt.legend()
    plt.grid(True)
    metrics_plot_filepath = os.path.join(run_output_directory, f"metrics_plot.png")
    plt.savefig(metrics_plot_filepath)
    print(f"Grafico metriche salvato in: {metrics_plot_filepath}")
    plt.close()

    # --- 9. Stampa Risultati Finali con Tabulate ---
    print("\n--- Risultati Finali sul Test Set ---")
    test_results_data = [
        ["Slot Filling F1", f1_test],
        ["Intent Classification Accuracy", acc_test],
        ["Loss Media Test", loss_test]
    ]
    # Stampa la tabella formattata
    print(tabulate(test_results_data, headers=["Metrica", "Valore"], tablefmt="grid"))

    print("\nEsecuzione Baseline completata. Controllare la cartella 'results' per i dettagli.")

# --- END OF FILE main.py ---
# --- START OF FILE main.py ---

# Questo file è usato per eseguire le tue funzioni e stampare i risultati

import os
from pprint import pprint
import matplotlib.pyplot as plt  # Per il plot delle loss
from tqdm import tqdm  # Per la barra di progresso

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from collections import Counter  # Importato per lo split stratificato

# Importa tutto da utils.py (dati, Lang, Dataset, collate, init_weights, device)
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
    model.train()  # Modalità addestramento
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
    model.eval()  # Modalità valutazione
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

    # --- 5. Definizione Iperparametri (invariato) ---
    hid_size, emb_size = 200, 300
    lr, clip = 0.001, 5
    n_epochs, patience = 50, 5
    out_slot, out_int, vocab_len = (
        len(lang.slot2id),
        len(lang.intent2id),
        len(lang.word2id),
    )
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    # --- 6. Addestramento e Valutazione SOLO per il Modello BASELINE ---
    print("\n--- Esecuzione Configurazione: BASELINE ---")

    # Istanzia SOLO il modello BASELINE
    model = ModelIAS(
        hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN
    )
    model.to(device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses_train_hist, losses_dev_hist, sampled_epochs_hist = [], [], []
    best_f1_dev, current_patience_count, best_epoch_num = 0.0, patience, 0
    best_model_state_dict = None

    print(f"Inizio addestramento BASELINE per max {n_epochs} epoche...")
    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoca {epoch}/{n_epochs}")
        loss_train_epoch = train_loop(
            train_loader, optimizer, criterion_slots, criterion_intents, model, clip
        )
        losses_train_hist.append(loss_train_epoch)
        print(f"Loss Addestramento Media: {loss_train_epoch:.4f}")

        f1_dev_epoch, acc_dev_epoch, loss_dev_epoch = eval_loop(
            dev_loader, criterion_slots, criterion_intents, model, lang
        )
        losses_dev_hist.append(loss_dev_epoch)
        sampled_epochs_hist.append(epoch)
        print(
            f"Risultati Dev: Loss={loss_dev_epoch:.4f}, Slot F1={f1_dev_epoch:.4f}, Intent Acc={acc_dev_epoch:.4f}"
        )

        # Early Stopping Logic
        if f1_dev_epoch > best_f1_dev:
            best_f1_dev = f1_dev_epoch
            best_epoch_num = epoch
            current_patience_count = patience
            print(f"*** Nuovo miglior F1 Dev! Salvo stato modello (Epoca {epoch}). ***")
            best_model_state_dict = model.state_dict().copy()
        else:
            current_patience_count -= 1
            print(
                f"Nessun miglioramento F1 Dev. Pazienza rimasta: {current_patience_count}"
            )

        if current_patience_count <= 0:
            print(f"--- Early stopping attivato all'epoca {epoch}. ---")
            break

    print(f"\nAddestramento BASELINE completato.")
    if best_model_state_dict:
        print(
            f"Miglior F1 Slot su Dev set: {best_f1_dev:.4f} ottenuto all'epoca {best_epoch_num}"
        )
    else:
        print("Attenzione: Nessun modello migliore salvato.")

    # --- 7. Valutazione Finale sul Test Set (solo Baseline) ---
    print(f"Valutazione BASELINE sul Test set...")
    if best_model_state_dict:
        print(f"Carico il modello migliore dall'epoca {best_epoch_num}.")
        model.load_state_dict(best_model_state_dict)
    else:
        print("Valuto con l'ultimo stato del modello disponibile.")

    f1_test, acc_test, loss_test = eval_loop(
        test_loader, criterion_slots, criterion_intents, model, lang
    )
    print("\n--- Risultati Finali BASELINE sul Test Set ---")
    print(f"  Slot Filling F1: {f1_test:.4f}")
    print(f"  Intent Classification Accuracy: {acc_test:.4f}")
    print(f"  Loss Media Test: {loss_test:.4f}")

    # --- 8. Plot delle Loss (solo Baseline) ---
    plt.figure(figsize=(10, 6))
    plt.plot(
        sampled_epochs_hist,
        losses_train_hist[: len(sampled_epochs_hist)],
        label=f"Train Loss",
    )
    plt.plot(sampled_epochs_hist, losses_dev_hist, label=f"Dev Loss")
    plt.title(f"Andamento Loss - BASELINE")
    plt.xlabel("Epoche")
    plt.ylabel("Loss Media (Cross Entropy)")
    if best_epoch_num > 0:
        plt.axvline(
            x=best_epoch_num,
            color="r",
            linestyle="--",
            label=f"Best Dev F1 Epoch ({best_epoch_num})",
        )
    plt.legend()
    plt.grid(True)
    plot_filename = f"loss_plot_BASELINE.png"
    plt.savefig(plot_filename)
    print(f"Grafico loss BASELINE salvato in: {plot_filename}")
    plt.close()

    # --- 9. Riepilogo Risultati (Non necessario per singolo run) ---
    # Rimosso il riepilogo comparativo

    print("\nEsecuzione Baseline completata.")

# --- END OF FILE main.py ---

# --- START OF FILE main.py ---

# Questo file è usato per eseguire le tue funzioni e stampare i risultati
# Modificato per eseguire una GRID SEARCH su Dropout Rate, Learning Rate, Batch Size
# per l'architettura BIDIREZIONALE (Step 2.A - Grid Search)
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

# Importa il modello (con bidirezionalità/dropout configurabili) da model.py
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
    loss_array = []
    # Non usiamo tqdm qui per ridurre l'output annidato nella grid search
    for sample in data:
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
    """Esegue la valutazione."""
    model.eval()
    loss_array = []
    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []
    with torch.no_grad():
        # Non usiamo tqdm qui per ridurre l'output annidato nella grid search
        for sample in data:
            slots, intents = model(sample["utterances"], sample["slots_len"])
            loss_intent = criterion_intents(intents, sample["intents"])
            loss_slot = criterion_slots(slots, sample["y_slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
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
    try:
        results = evaluate(ref_slots, hyp_slots)
        slot_f1 = results.get("total", {}).get("f", 0.0)
    except Exception as ex:
        tqdm.write(f"\nAttenzione: Errore durante la valutazione conll: {ex}")
        slot_f1 = 0.0
    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=0, output_dict=True
    )
    intent_acc = report_intent.get("accuracy", 0.0)
    mean_loss = np.mean(loss_array) if loss_array else float("inf")
    return slot_f1, intent_acc, mean_loss


# --- Funzioni per la gestione dei risultati e delle cartelle (invariate) ---
def create_run_output_dir(base_results_dir, step_name, run_params_str):
    run_dir = os.path.join(base_results_dir, step_name, run_params_str)
    os.makedirs(run_dir, exist_ok=True)
    # Non stampiamo più qui, lo facciamo nel loop principale con tqdm.write
    return run_dir


def save_epoch_history(run_output_dir, history_data):
    csv_filepath = os.path.join(run_output_dir, "epoch_history.csv")
    if not history_data:  # Gestisci caso di storia vuota (es. errore all'inizio)
        print(f"Warning: Nessun dato storico da salvare per {run_output_dir}")
        return
    fieldnames = history_data[0].keys()  # Prendi le chiavi dal primo dizionario
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


# --- Funzione per generare e salvare Heatmap ---
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
    """Genera e salva una heatmap da un DataFrame pandas."""
    try:
        # Crea la tabella pivot, usando la media se ci sono duplicati (dovuti al terzo parametro non visualizzato)
        pivot_table = pd.pivot_table(
            df, values=value_col, index=index_col, columns=columns_col, aggfunc=np.mean
        )

        plt.figure(
            figsize=(
                len(pivot_table.columns) * 1.5 + 4,
                len(pivot_table.index) * 0.8 + 3,
            )
        )  # Adatta dimensione
        # Per la loss, un cmap diverso può essere migliore (es. 'coolwarm' o 'YlGnBu_r')
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

        filename = f"heatmap_{metric_safe_name}_vs_{idx_param.replace(' ', '')}_vs_{col_param.replace(' ', '')}.png"
        filepath = os.path.join(run_output_dir_base, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"  Heatmap salvata: {filepath}")
    except Exception as e:
        print(
            f"  Errore durante la generazione/salvataggio della heatmap per {metric} ({idx_param} vs {col_param}): {e}"
        )


# --- Blocco Principale di Esecuzione ---
if __name__ == "__main__":
    # --- 1. Caricamento e Preparazione Dati ---
    # Scommentiamo il blocco di caricamento e preparazione dati
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
    # train_test_split ora prende inputs e labels
    X_train, X_dev, _, _ = train_test_split(
        inputs,
        labels,  # Passa labels qui per stratify
        test_size=portion,
        random_state=42,
        shuffle=True,
        stratify=labels,  # Stratifica sulle labels
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
    print("Creazione Dataset...")
    # Crea le istanze dei Dataset
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)
    print("Dataset pronti.")

    # --- 5. Definizione Iperparametri FISSI e Liste per GRID SEARCH ---
    # Parametri fissi generali
    hid_size = 200
    emb_size = 300
    clip = 5
    n_epochs = 200
    patience = 3

    # Architettura fissa: BIDIREZIONALE
    bidirectional = True

    # Liste dei parametri da testare nella Grid Search
    # ATTENZIONE: Il numero totale di run sarà len(dropout)*len(lr)*len(batch)
    # Puoi modificare queste liste in base ai risultati dei tuning sequenziali se li fai prima
    dropout_rates_to_test = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    learning_rates_to_test = [
        0.0005,
        0.001,
        0.005,
    ]  # Puoi usare i migliori 3-4 dalla bidirezionale lr tuning
    batch_sizes_to_test = [
        64,
        128,
    ]  # Esempi: puoi aggiungere 32, 256 se la memoria lo permette

    out_slot, out_int, vocab_len = (
        len(lang.slot2id),
        len(lang.intent2id),
        len(lang.word2id),
    )
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    # --- 6. Inizio Grid Search ---
    base_results_directory = "results"
    step_directory_name = "step_bidrop_grid_search"  # Nome per questo step

    all_runs_test_results = []  # Lista per raccogliere i risultati di TUTTE le combinazioni

    # Calcola il numero totale di run per tqdm
    total_runs = (
        len(dropout_rates_to_test)
        * len(learning_rates_to_test)
        * len(batch_sizes_to_test)
    )
    print(f"\nInizio Grid Search per l'architettura Bidirezionale+Dropout.")
    print(f"Parametri da testare:")
    print(f"  - Dropout Rates: {dropout_rates_to_test}")
    print(f"  - Learning Rates: {learning_rates_to_test}")
    print(f"  - Batch Sizes: {batch_sizes_to_test}")
    print(f"Numero totale di combinazioni (run): {total_runs}")
    print("-" * 30)

    # Loop esterno principale per la Grid Search
    run_counter = 0
    # Non possiamo usare tqdm qui facilmente a causa dei loop annidati e della creazione dinamica dei dataloader

    for dropout_rate in dropout_rates_to_test:
        for lr in learning_rates_to_test:
            for batch_size_train in batch_sizes_to_test:
                run_counter += 1
                # Usiamo print normale o tqdm.write per questi messaggi fuori dalla barra epoca
                print(f"\n--- RUN {run_counter}/{total_runs} ---")
                print(
                    f"Parametri Correnti: Dropout={dropout_rate}, LR={lr}, BatchSize={batch_size_train}"
                )

                # --- Configurazione Specifica per questa Run ---
                config_name = f"BI_DR{dropout_rate}_LR{lr}_BS{batch_size_train}"
                params_string = f"hid{hid_size}_emb{emb_size}_lr{str(lr).replace('.', '')}_p{patience}_bs{batch_size_train}_bi{int(bidirectional)}_dr{str(dropout_rate).replace('.', '')}"

                # Usa tqdm.write (anche se fuori dal loop epoch_iterator in questo punto,
                # tqdm.write è più sicuro di print per i messaggi intermedi)
                run_output_directory = create_run_output_dir(
                    base_results_directory, step_directory_name, params_string
                )
                tqdm.write(f"Output directory per questa run: {run_output_directory}")

                # --- Creazione DataLoader DINAMICA (dipende dal batch_size corrente) ---
                # È importante ricreare i dataloader se il batch size cambia!
                current_train_loader = DataLoader(
                    train_dataset,  # train_dataset ora è definito
                    batch_size=batch_size_train,
                    collate_fn=collate_fn,
                    shuffle=True,
                )
                # Dev e Test loader possono mantenere il batch size fisso o variare anche quello
                # Manteniamo fisso quello di valutazione (64) come nel notebook originale per coerenza tra run
                dev_loader = DataLoader(
                    dev_dataset,
                    batch_size=64,
                    collate_fn=collate_fn,  # dev_dataset ora è definito
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=64,
                    collate_fn=collate_fn,  # test_dataset ora è definito
                )
                print("DataLoader pronti per questa run.")  # Aggiunto per feedback

                # --- Addestramento e Valutazione per la Combinazione Corrente ---
                tqdm.write(f"\nEsecuzione Configurazione: {config_name}")

                model = ModelIAS(
                    hid_size,
                    out_slot,
                    out_int,
                    emb_size,
                    vocab_len,
                    pad_index=PAD_TOKEN,
                    bidirectional=bidirectional,
                    dropout_rate=dropout_rate,
                )
                model.to(device)
                model.apply(init_weights)
                optimizer = optim.Adam(model.parameters(), lr=lr)

                epoch_history_data = []
                best_f1_dev, current_patience_count, best_epoch_num = 0.0, patience, 0
                best_model_state_dict = None

                tqdm.write(f"Inizio addestramento per max {n_epochs} epoche...")
                # Barra di progresso per le epoche di questa run
                # Usiamo leave=True per lasciare la barra visibile finché tutti i loop annidati non sono completi
                epoch_iterator = tqdm(
                    range(1, n_epochs + 1),
                    desc=f"Run {run_counter}/{total_runs} Training",
                    leave=True,
                )

                try:  # Aggiunto blocco try per catturare errori critici durante l'addestramento
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
                            dev_loader, criterion_slots, criterion_intents, model, lang
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

                        # Early Stopping Logic
                        if f1_dev_epoch > best_f1_dev:
                            best_f1_dev = f1_dev_epoch
                            best_epoch_num = epoch
                            current_patience_count = patience
                            tqdm.write(
                                f"  (Run {run_counter}) Epoca {epoch}: Nuovo miglior F1 Dev = {best_f1_dev:.4f}. Salvato stato."
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
                    # Salva lo stato attuale del modello anche in caso di errore, se esiste
                    try:
                        if model:
                            torch.save(
                                model.state_dict(),
                                os.path.join(
                                    run_output_directory, "last_state_on_error.pth"
                                ),
                            )
                    except Exception as save_e:
                        tqdm.write(
                            f"Errore durante salvataggio stato in caso di errore: {save_e}"
                        )
                    # Continua al prossimo run (ma potresti voler interrompere tutto)
                    continue  # Passa al prossimo batch size combination

                finally:  # Assicura che la barra di tqdm dell'epoca sia completata anche in caso di errore
                    epoch_iterator.close()

                tqdm.write(f"Addestramento {config_name} completato.")
                if not best_model_state_dict:
                    tqdm.write(
                        "  Attenzione: Nessun modello migliore salvato. Valuto con l'ultimo stato."
                    )
                    # In caso di errore o nessun miglioramento, best_model_state_dict potrebbe essere None.
                    # Assicuriamoci che ci sia qualcosa da caricare, anche se è l'ultimo stato prima dell'early stop.
                    if epoch_history_data:  # Se almeno un'epoca è stata completata
                        best_model_state_dict = (
                            model.state_dict().copy()
                        )  # Usa lo stato corrente
                    else:  # Se non è stata completata nemmeno un'epoca (errore subito)
                        tqdm.write(
                            "  Nessuno stato modello disponibile per la valutazione finale."
                        )
                        # Salta la valutazione finale per questa run e aggiungi risultati vuoti
                        all_runs_test_results.append(
                            {
                                "Config": config_name,
                                "Dropout Rate": dropout_rate,
                                "Learning Rate": lr,
                                "Batch Size Train": batch_size_train,
                                "Test Slot F1": 0.0,  # Risultati a 0 in caso di fallimento
                                "Test Intent Acc": 0.0,
                                "Test Loss": float("inf"),  # Loss infinita
                            }
                        )
                        # Pulisce memoria e passa al prossimo run
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
                        continue  # Passa al prossimo run

                # Valutazione Finale sul Test Set
                tqdm.write(f"Valutazione {config_name} sul Test set finale...")
                model.load_state_dict(
                    best_model_state_dict
                )  # Carica il migliore o l'ultimo

                f1_test, acc_test, loss_test = eval_loop(
                    test_loader, criterion_slots, criterion_intents, model, lang
                )

                tqdm.write(
                    f"  Risultati Test - {config_name}: F1={f1_test:.4f}, Acc={acc_test:.4f}, Loss={loss_test:.4f}"
                )

                # Salvataggio Risultati e Plot
                tqdm.write("Salvataggio storia, risultati finali e grafici...")
                save_epoch_history(run_output_directory, epoch_history_data)
                save_test_results(run_output_directory, f1_test, acc_test, loss_test)

                # Plot (i plot sono gli stessi di prima, salvati nella cartella della run)
                # ... (codice plot invariato) ...
                # Grafico delle Loss
                plt.figure(figsize=(10, 6))
                epochs_plot = (
                    [d["Epoch"] for d in epoch_history_data]
                    if epoch_history_data
                    else []
                )
                train_losses_plot = (
                    [d["Train_Loss"] for d in epoch_history_data]
                    if epoch_history_data
                    else []
                )
                dev_losses_plot = (
                    [d["Dev_Loss"] for d in epoch_history_data]
                    if epoch_history_data
                    else []
                )
                plt.plot(epochs_plot, train_losses_plot, label="Train Loss")
                plt.plot(epochs_plot, dev_losses_plot, label="Dev Loss")
                plt.title(f"Andamento Loss - {config_name}")
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
                loss_plot_filepath = os.path.join(
                    run_output_directory, f"loss_plot.png"
                )
                plt.savefig(loss_plot_filepath)
                plt.close()
                tqdm.write(f"  Grafico loss salvato: {loss_plot_filepath}")

                # Grafico delle Metriche
                plt.figure(figsize=(10, 6))
                dev_f1s_plot = (
                    [d["Dev_Slot_F1"] for d in epoch_history_data]
                    if epoch_history_data
                    else []
                )
                dev_accs_plot = (
                    [d["Dev_Intent_Acc"] for d in epoch_history_data]
                    if epoch_history_data
                    else []
                )
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
                    current_train_loader,  # Pulisci anche i dataloader specifici della run
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()

    # --- 11. Riepilogo Comparativo Finale e HEATMAPS ---
    print(
        "\n--- Riepilogo Comparativo Risultati Test (Grid Search Bidirezionale+Dropout) ---"
    )

    # Crea DataFrame pandas dai risultati raccolti
    df_results = pd.DataFrame(all_runs_test_results)

    # Stampa la tabella riassuntiva
    print(tabulate(df_results, headers="keys", tablefmt="grid", showindex=False))

    # Salva la tabella riassuntiva completa in un CSV
    summary_csv_path = os.path.join(
        base_results_directory, step_directory_name, "grid_search_summary.csv"
    )
    df_results.to_csv(summary_csv_path, index=False)
    print(f"\nRiepilogo completo salvato in: {summary_csv_path}")

    # --- Generazione Heatmaps ---
    print("\nGenerazione Heatmaps (valori medi dove applicabile)...")

    # Definisci le metriche da visualizzare
    metrics_to_plot = ["Test Slot F1", "Test Intent Acc", "Test Loss"]
    # Definisci le coppie di parametri per gli assi delle heatmap
    # Useremo sempre Batch Size o Learning Rate sull'asse X e Dropout Rate sull'asse Y
    # Per Batch Size vs LR, dovremo aggregare sul Dropout Rate
    param_pairs = [
        ("Dropout Rate", "Learning Rate", "Batch Size Train"),
        ("Dropout Rate", "Batch Size Train", "Learning Rate"),
        (
            "Learning Rate",
            "Batch Size Train",
            "Dropout Rate",
        ),  # Adattato per Batch vs LR
    ]

    # Directory base per le heatmap
    heatmap_dir = os.path.join(base_results_directory, step_directory_name, "_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    for metric in metrics_to_plot:
        metric_safe_name = metric.replace(" ", "_")  # Nome sicuro per file
        for idx_param, col_param, agg_param in param_pairs:
            # Assicurati che la combinazione di colonne esista nel DataFrame
            if (
                idx_param in df_results.columns
                and col_param in df_results.columns
                and agg_param in df_results.columns
            ):
                try:
                    # Crea la tabella pivot, aggregando con la media sulla dimensione non visualizzata
                    pivot_table = pd.pivot_table(
                        df_results,
                        values=metric,
                        index=idx_param,
                        columns=col_param,
                        aggfunc=np.mean,
                    )

                    # Adatta dimensione figura
                    fig_width = max(6, len(pivot_table.columns) * 1.5 + 3)
                    fig_height = max(4, len(pivot_table.index) * 0.8 + 2)
                    plt.figure(figsize=(fig_width, fig_height))

                    # Per la loss, un cmap diverso può essere migliore
                    cmap_val = (
                        "viridis" if "F1" in metric or "Acc" in metric else "YlGnBu_r"
                    )  # YlGnBu_r è invertito, per Loss decrescente
                    sns.heatmap(
                        pivot_table,
                        annot=True,
                        fmt=".4f",
                        cmap=cmap_val,
                        linewidths=0.5,
                    )
                    plt.title(
                        f"{metric} vs {idx_param} & {col_param}\n(Aggregato su {agg_param})",
                        fontsize=12,
                    )
                    plt.xlabel(col_param, fontsize=10)
                    plt.ylabel(idx_param, fontsize=10)
                    plt.xticks(rotation=45, ha="right")
                    plt.yticks(rotation=0)
                    plt.tight_layout()

                    filename = f"heatmap_{metric_safe_name}_vs_{idx_param.replace(' ', '')}_vs_{col_param.replace(' ', '')}.png"
                    filepath = os.path.join(heatmap_dir, filename)
                    plt.savefig(filepath)
                    plt.close()
                    print(f"  Heatmap salvata: {filepath}")
                except Exception as e:
                    print(
                        f"  Errore durante la generazione della heatmap per {metric} ({idx_param} vs {col_param}): {e}"
                    )
            else:
                print(
                    f"  Skipping heatmap for {metric} ({idx_param} vs {col_param}): Colonne non trovate nel DataFrame."
                )

    print("\nGrid Search completata. Controllare la cartella 'results' per i dettagli.")

# --- END OF FILE main.py ---

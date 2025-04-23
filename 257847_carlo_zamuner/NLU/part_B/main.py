# --- START OF FILE main.py ---

# Questo file è usato per eseguire le tue funzioni e stampare i risultati
# Implementazione per la Parte 2.B: Fine-tuning BERT per Joint Intent & Slot
# Corretto l'errore "assigned before global declaration"

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

# Non necessari per una singola run BERT, ma li lasciamo nel caso si voglia fare tuning/heatmap in futuro
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

# Importa il tokenizer di BERT (anche se importato in utils, meglio averlo qui per chiarezza)
from transformers import BertTokenizer


# --- Funzioni di Training e Valutazione ADATTATE per BERT ---
# Queste funzioni devono gestire l'input/output specifico di BERT

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=1.0): # Clipping tipico per BERT è più basso (es. 1.0)
    """Esegue un'epoca di addestramento per il modello BERT."""
    model.train() # Modalità addestramento
    total_loss = 0.0
    # tqdm per il progresso sui batch (utile anche con output ridotto)
    for batch in tqdm(data, desc="Training Batches", leave=False): # leave=False per non intasare l'output
        optimizer.zero_grad()

        # Passaggio forward: il modello BERT riceve input IDs, attention mask, token type IDs
        # e restituisce logits per intent e slot
        intent_logits, slot_logits = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids']
        )

        # --- Calcolo Loss Intent ---
        # intent_logits: (batch_size, num_intent_labels)
        # batch['intent_labels']: (batch_size)
        loss_intent = criterion_intents(intent_logits, batch['intent_labels'])

        # --- Calcolo Loss Slot ---
        # slot_logits: (batch_size, seq_len, num_slot_labels)
        # batch['slot_labels']: (batch_size, seq_len)
        # CrossEntropyLoss per task di Slot Filling (sequenza)
        # slot_logits deve essere (batch_size, num_slot_labels, seq_len)
        # batch['slot_labels'] deve essere (batch_size, seq_len)
        # Utilizziamo IGNORE_INDEX per ignorare i token di padding e i sub-token non-iniziali
        # La loss viene calcolata solo sui token validi e sui primi sub-token delle parole originali
        loss_slot = criterion_slots(slot_logits.permute(0, 2, 1), batch['slot_labels'])


        # Combina le loss
        loss = loss_intent + loss_slot # Multi-task learning semplice con somma delle loss
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # Aggiornamento pesi
        optimizer.step()

    # Evita divisione per zero se per qualche motivo il DataLoader è vuoto
    return total_loss / len(data) if len(data) > 0 else float('inf')

def eval_loop(data, criterion_slots, criterion_intents, model, lang, tokenizer):
    """Esegue la valutazione per il modello BERT."""
    model.eval() # Modalità valutazione
    total_loss = 0.0

    ref_intents, hyp_intents = [], []
    ref_slots_list, hyp_slots_list = [], [] # Liste per accumulare seq di slot (formato conll)

    ignore_index = IGNORE_INDEX # Indice per label slot da ignorare

    with torch.no_grad(): # Disabilita calcolo gradienti
        # tqdm per il progresso sui batch (utile anche con output ridotto)
        for batch in tqdm(data, desc="Evaluation Batches", leave=False):

            intent_logits, slot_logits = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids']
            )

            # Calcolo loss (utile per monitoraggio)
            loss_intent = criterion_intents(intent_logits, batch['intent_labels'])
            loss_slot = criterion_slots(slot_logits.permute(0, 2, 1), batch['slot_labels'])
            loss = loss_intent + loss_slot
            total_loss += loss.item()

            # --- Intent Inference ---
            out_intent_ids = torch.argmax(intent_logits, dim=1).cpu().tolist()
            hyp_intents.extend([lang.id2intent.get(idx, 'UNK_INTENT') for idx in out_intent_ids])
            ref_intents.extend([lang.id2intent.get(idx, 'UNK_INTENT') for idx in batch['intent_labels'].cpu().tolist()])

            # --- Slot Inference e Allineamento Sub-token ---
            predicted_slot_ids = torch.argmax(slot_logits, dim=-1).cpu().tolist() # (batch_size, seq_len)

            # Per ogni sequenza nel batch:
            for i in range(batch['input_ids'].size(0)):
                # Recupera le info necessarie dal batch (fornite da Dataset/Collate)
                word_id_sequence = batch['word_ids'][i] # Mapping subtoken -> id parola originale
                attention_mask_sequence = batch['attention_mask'][i].cpu().tolist() # Per identificare padding/speciali
                original_slot_labels_list = batch['original_slot_labels'][i] # Etichette slot originali (lista di stringhe)
                original_words_list = batch['original_words'][i] # Parole originali (lista di stringhe)

                predicted_slot_id_sequence = predicted_slot_ids[i] # Predizioni ID slot per i sub-token

                current_pred_slots_labels = []
                current_ref_slots_labels = []
                current_original_words = [] # Parole originali corrispondenti alle predizioni/ref


                last_word_id = None # Per identificare il primo sub-token di ogni parola

                # Itera sui sub-token (con i loro word_ids) per allineare predizioni e ground truth alle parole originali
                for token_idx, word_id in enumerate(word_id_sequence):
                    # Processa solo token reali (non padding) e che corrispondono a una parola originale (non CLS/SEP)
                    if word_id is not None and attention_mask_sequence[token_idx] == 1:
                         if word_id != last_word_id: # Questo è il PRIMO sub-token di una parola originale
                              # Recupera l'etichetta ground truth originale per questa parola
                              # E la parola originale stessa
                              if word_id < len(original_slot_labels_list):
                                   current_ref_slots_labels.append(original_slot_labels_list[word_id])
                                   current_original_words.append(original_words_list[word_id])

                                   # Recupera la predizione ID slot per questo PRIMO sub-token
                                   predicted_id = predicted_slot_id_sequence[token_idx]
                                   # Mappa l'ID predetto all'etichetta slot stringa
                                   predicted_slot_label = lang.id2slot.get(predicted_id, 'UNK_SLOT_PRED') # Fallback sicuro
                                   current_pred_slots_labels.append(predicted_slot_label)
                                   # Nota: Stiamo usando la predizione del PRIMO sub-token per l'intera parola,
                                   # come suggerito dal paper.
                              else:
                                   # Questo caso non dovrebbe succedere con Dataset ben formato, ma per robustezza
                                   pass # Salta se word_id è fuori dai limiti delle label originali
                    last_word_id = word_id # Aggiorna last_word_id

                # Ora le liste current_original_words, current_ref_slots_labels, current_pred_slots_labels
                # contengono le parole originali e le loro etichette (reali e predette) allineate,
                # saltando i sub-token non iniziali e i token speciali/padding.
                # Formatta per lo script conll (lista di tuple (parola, etichetta))
                # Assicurati che le lunghezze siano uguali (dovrebbero esserlo se la logica sopra funziona)
                if len(current_original_words) == len(current_ref_slots_labels) and len(current_original_words) == len(current_pred_slots_labels):
                    ref_slots_list.append(list(zip(current_original_words, current_ref_slots_labels)))
                    hyp_slots_list.append(list(zip(current_original_words, current_pred_slots_labels)))
                else:
                    tqdm.write(f"\nWarning: Mismatch in lengths for conll eval in batch {i}.")
                    tqdm.write(f"  Original words: {len(current_original_words)}, Ref labels: {len(current_ref_slots_labels)}, Pred labels: {len(current_pred_slots_labels)}")
                    # Aggiungi comunque liste vuote o parziali per non bloccare evaluate()
                    min_len = min(len(current_original_words), len(current_ref_slots_labels), len(current_pred_slots_labels))
                    ref_slots_list.append(list(zip(current_original_words[:min_len], current_ref_slots_labels[:min_len])))
                    hyp_slots_list.append(list(zip(current_original_words[:min_len], current_pred_slots_labels[:min_len])))


    # --- Calcolo Metriche ---
    # Slot Filling F1 (usando lo script conll)
    try:
        results = evaluate(ref_slots_list, hyp_slots_list)
        slot_f1 = results.get('total', {}).get('f', 0.0)
    except Exception as ex:
        tqdm.write(f"\nAttenzione: Errore durante la valutazione conll: {ex}")
        slot_f1 = 0.0 # Fallback

    # Intent Classification Accuracy (usando sklearn)
    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=0, output_dict=True
    )
    intent_acc = report_intent.get("accuracy", 0.0)

    mean_loss = total_loss / len(data) if len(data) > 0 else float('inf')

    return slot_f1, intent_acc, mean_loss


# --- Funzioni per la gestione dei risultati e delle cartelle (simili a prima) ---
# useremo queste funzioni, non c'è bisogno di riscriverle se sono già definite e funzionano

def create_run_output_dir(base_results_dir, step_name, run_params_str):
    """Crea la struttura di directory per salvare i risultati di una run."""
    run_dir = os.path.join(base_results_dir, step_name, run_params_str)
    os.makedirs(run_dir, exist_ok=True)
    tqdm.write(f"Risultati per questa run verranno salvati in: {run_dir}")
    return run_dir

def save_epoch_history(run_output_dir, history_data):
    """Salva la lista di dizionari history_data in un file CSV."""
    csv_filepath = os.path.join(run_output_dir, 'epoch_history.csv')
    if not history_data: return # Non salvare se vuoto
    fieldnames = history_data[0].keys()
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_data)

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

    # --- 1. Caricamento e Preparazione Dati ---
    print("Caricamento dati ATIS...")
    data_dir = os.path.join("dataset", "ATIS")
    train_raw_all = load_data(os.path.join(data_dir, "train.json")) # Carica tutto il train per lo split
    test_raw = load_data(os.path.join(data_dir, "test.json"))
    print(f"Campioni Train caricati (prima dello split): {len(train_raw_all)}")
    print(f"Campioni Test caricati: {len(test_raw)}")

    print("Creazione Dev set...")
    portion = 0.10
    intents_list_all = [x["intent"] for x in train_raw_all]

    # --- Gestione Intenti Rari e Split Stratificato ---
    # Raccogli inputs e labels per lo split, separando gli intenti rari
    inputs_split, labels_split, mini_train_split = [], [], []
    count_y_split = Counter(intents_list_all)
    for i, intent in enumerate(intents_list_all):
         if count_y_split[intent] > 1:
             inputs_split.append(train_raw_all[i])
             labels_split.append(intent)
         else:
             mini_train_split.append(train_raw_all[i]) # Metti intenti rari nel training set

    # Esegui lo split stratificato solo sui dati con intenti non rari
    X_train_split, X_dev_split, _, _ = train_test_split(
        inputs_split,
        labels_split, # Stratifica su questa lista di labels
        test_size=portion,
        random_state=42, # Seed per riproducibilità
        shuffle=True,
        stratify=labels_split, # Stratifica usando le labels
    )

    # Riunisci i dati di training aggiungendo gli intenti rari
    train_raw = X_train_split # train_raw ora contiene il set di training effettivo (parte stratificata)
    train_raw.extend(mini_train_split) # Aggiungi gli intenti rari al training set
    dev_raw = X_dev_split     # dev_raw contiene il set di development

    print(f"Dimensioni finali (dopo split e gestione rari): Train={len(train_raw)}, Dev={len(dev_raw)}, Test={len(test_raw)}")


    # --- Creazione Vocabolari Etichette (Lang) ---
    # Estrae tutti gli slot e intenti da *tutti* i set (train+dev+test) per assicurare copertura per Lang
    # Non usiamo words_train qui per costruire il vocabolario parole, solo label per Lang
    corpus_all = train_raw + dev_raw + test_raw # Corpus completo per le label
    all_slots_corpus = set(sum([line['slots'].split() for line in corpus_all], []))
    all_intents_corpus = set([line['intent'] for line in corpus_all])

    # Lang ora gestisce solo label
    lang = Lang(all_intents_corpus, all_slots_corpus)
    print(f"Slot: {len(lang.slot2id)} etichette")
    print(f"Intenti: {len(lang.intent2id)} etichette")

    # --- 2. Inizializzazione Tokenizer BERT ---
    print("\nInizializzazione BertTokenizer...")
    tokenizer_name = 'bert-base-uncased' # Nome del modello BERT da usare
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    print(f"BertTokenizer vocab size: {len(tokenizer)}")
    print(f"BertTokenizer [PAD] token ID: {tokenizer.pad_token_id}")
    # Assicurati che l'ID speciale per ignorare sia diverso dagli ID reali e dal padding
    # IGNORE_INDEX = -100 è una scelta sicura.

    # --- 3. Creazione Dataset e DataLoader (ADATTATI per BERT) ---
    # Usiamo la nuova classe IntentsAndSlotsBERT e la nuova collate_fn_bert
    print("Creazione Dataset e DataLoader (BERT compatible)...")

    # Max length come nel paper o notebook originale (50 nel paper)
    max_seq_length = 50 # BERT-base max è 512, ma 50 è comune per ATIS/Snips

    # Passiamo anche le liste di parole e label originali al Dataset per l'eval loop
    train_dataset = IntentsAndSlotsBERT(train_raw, lang, tokenizer, max_len=max_seq_length)
    dev_dataset = IntentsAndSlotsBERT(dev_raw, lang, tokenizer, max_len=max_seq_length)
    test_dataset = IntentsAndSlotsBERT(test_raw, lang, tokenizer, max_len=max_seq_length)

    # Batch size (128 è usato nel paper per fine-tuning, ma lo abbiamo testato)
    # Decidi quale batch size usare per questo run singolo (es. il migliore dalla grid search)
    batch_size_train_fixed = 128 # Usiamo 128 come nel paper e nella grid search testata
    batch_size_eval_fixed = 128  # Usiamo lo stesso batch size per eval per coerenza

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train_fixed, collate_fn=collate_fn_bert, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size_eval_fixed, collate_fn=collate_fn_bert)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval_fixed, collate_fn=collate_fn_bert)

    print("DataLoader BERT-compatible pronti.")
    print(f"Lunghezza massima sequenza (dopo tokenizzazione/padding): {max_seq_length}")
    print(f"Batch size (Train/Eval): {batch_size_train_fixed}/{batch_size_eval_fixed}")


    # --- 4. Definizione Modello, Ottimizzatore, Loss (ADATTATI per BERT) ---
    # Usiamo la nuova classe JointBERTModel
    print("\nInizializzazione Modello BERT...")

    # Ottieni numero di label da lang object
    num_intent_labels = len(lang.intent2id)
    num_slot_labels = len(lang.slot2id) # Include 'pad'

    # Iperparametri per fine-tuning BERT
    # Decidi quale LR e Dropout Rate usare per questo run singolo (es. i migliori dalla grid search)
    lr_fixed = 5e-5 # LR dal paper o il migliore dalla grid search (es. 0.00005 o 0.001 o 0.005)
    bert_dropout_rate_fixed = 0.1 # Dropout dal paper o il migliore dalla grid search (es. 0.1, 0.4, 0.5)
    clip = 1.0 # Clipping consigliato per fine-tuning BERT

    # Istanzia il modello BERT con i parametri fissi scelti
    model = JointBERTModel(num_intent_labels, num_slot_labels, bert_model_name=tokenizer_name, dropout_rate=bert_dropout_rate_fixed)
    model.to(device)

    # Ottimizzatore (Adam)
    optimizer = optim.Adam(model.parameters(), lr=lr_fixed)

    # Criteri di Loss
    criterion_intents = nn.CrossEntropyLoss()
    criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX) # Usa IGNORE_INDEX


    print("Modello BERT inizializzato.")
    print(f"Hyperparameters: LR={lr_fixed}, Dropout={bert_dropout_rate_fixed}, BatchSize={batch_size_train_fixed}, MaxLen={max_seq_length}")


    # --- 5. Ciclo di Addestramento e Valutazione ---
    # Usiamo parametri di addestramento/early stopping ragionevoli per fine-tuning
    n_epochs = 50 # Aumentato un po' per early stopping
    patience = 10 # Aumentato un po' per stabilità


    base_results_directory = "results"
    # Nome dello step per questo run BERT
    step_directory_name = "step_bert_finetuning_final" # Nome specifico per il run finale con i migliori parametri

    # Stringa parametri per il nome della cartella di questo run finale
    params_string = f"lr{str(lr_fixed).replace('.', '')}_dr{str(bert_dropout_rate_fixed).replace('.', '')}_bs{batch_size_train_fixed}_maxlen{max_seq_length}"
    run_output_directory = create_run_output_dir(base_results_directory, step_directory_name, params_string)


    # --- 6. Addestramento e Valutazione (BERT) ---
    print(f"\n--- Inizio Fine-tuning BERT ---")

    epoch_history_data = []

    best_f1_dev, current_patience_count, best_epoch_num = 0.0, patience, 0
    best_model_state_dict = None


    tqdm.write(f"Inizio fine-tuning per max {n_epochs} epoche...")
    epoch_iterator = tqdm(range(1, n_epochs + 1), desc=f"BERT Fine-tuning", leave=True)

    try:
        for epoch in epoch_iterator:
            loss_train_epoch = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip)

            # eval_loop ora riceve anche il tokenizer
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
                     # Stampa meno frequentemente
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
        # In caso di errore critico, imposta risultati finali a 0/inf
        f1_test, acc_test, loss_test = 0.0, 0.0, float('inf')
        best_model_state_dict = None # Non usare uno stato potenzialmente corrotto

    finally:
         epoch_iterator.close()

    # Se non è stato trovato un modello migliore valido durante l'addestramento riuscito,
    # controlla se c'è uno stato dall'ultima epoca (se l'addestramento non è fallito prima)
    if best_model_state_dict is None and epoch_history_data:
         tqdm.write("Attenzione: Nessun miglioramento F1 Dev trovato. Valuto con l'ultimo stato completato.")
         best_model_state_dict = model.state_dict().copy()
         # Potrebbe anche essere utile salvare questo ultimo stato
         torch.save(best_model_state_dict, os.path.join(run_output_directory, "last_state_valid.pth"))
    elif best_model_state_dict is None and not epoch_history_data:
         tqdm.write("Attenzione: Addestramento fallito o interrotto subito. Impossibile eseguire valutazione finale.")


    # --- 7. Valutazione Finale sul Test Set ---
    # Esegui solo se è stato trovato almeno uno stato modello valido (migliore o ultimo)
    if best_model_state_dict:
        print(f"\nValutazione finale sul Test set...")
        try:
            # Ricarica il modello pulito con gli stessi parametri
            model_final_eval = JointBERTModel(num_intent_labels, num_slot_labels, bert_model_name=tokenizer_name, dropout_rate=bert_dropout_rate_fixed)
            model_final_eval.load_state_dict(best_model_state_dict)
            model_final_eval.to(device)
            # Esegui la valutazione finale
            f1_test, acc_test, loss_test = eval_loop(test_loader, criterion_slots, criterion_intents, model_final_eval, lang, tokenizer)
            del model_final_eval # Pulisci dopo l'uso

        except Exception as e:
            print(f"\nERRORE durante la valutazione finale sul Test set: {e}")
            f1_test, acc_test, loss_test = 0.0, 0.0, float('inf') # Imposta risultati a 0 in caso di errore


    else: # Caso in cui non è stato trovato nessuno stato modello valido
         print("\nSkipping final test set evaluation due to critical error during training.")
         # I risultati sono già stati impostati a 0/inf nel blocco try/except

    print("\n--- Risultati Finali sul Test Set ---")
    config_name_final = f"BERT_DR{bert_dropout_rate_fixed}_LR{lr_fixed}_BS{batch_size_train_fixed}_MaxLen{max_seq_length}"
    print(f"Configurazione: {config_name_final}")
    test_results_data = [
        ["Slot Filling F1", f1_test],
        ["Intent Classification Accuracy", acc_test],
        ["Loss Media Test", loss_test]
    ]
    print(tabulate(test_results_data, headers=["Metrica", "Valore"], tablefmt="grid"))

    # --- 8. Salvataggio Risultati e Plot ---
    print("\nSalvataggio storia, risultati finali e grafici...")

    # Salva risultati solo se validi
    if best_model_state_dict or epoch_history_data: # Se c'è storia (anche se non c'è un "migliore" salvato)
         save_epoch_history(run_output_directory, epoch_history_data)
         save_test_results(run_output_directory, f1_test, acc_test, loss_test)
    else:
         print("Nessun dato valido da salvare.")


    # Genera e salva il grafico delle Loss
    if epoch_history_data: # Plot solo se ci sono dati nella storia
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

        # Genera e salva il grafico delle Metriche
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
    # del model è già stato fatto alla fine della valutazione

    print("\nEsecuzione Parte 2.B completata.")

# --- END OF FILE main.py ---
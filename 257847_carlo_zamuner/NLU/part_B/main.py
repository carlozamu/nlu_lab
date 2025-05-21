import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from collections import Counter 
from tabulate import tabulate 
import copy 

from utils import device, IGNORE_INDEX, Lang, IntentsAndSlotsBERT, collate_fn_bert, load_data, train_test_split
from model import JointBERTModel
from functions import train_loop_bert, eval_loop_bert
from conll import evaluate as conll_evaluate

from transformers import BertTokenizerFast

# ==============================================================================
# --- MAIN EXECUTION BLOCK  ---
# ==============================================================================
if __name__ == "__main__":

    bert_model_name_config = 'bert-base-uncased'
    learning_rate_config = 5e-5
    dropout_rate_config = 0.3
    batch_size_train_config = 128
    batch_size_eval_config = 128

    max_epochs_config = 50
    early_stop_patience_config = 10
    grad_clip_config = 1.0
    max_seq_length_config = 50
    
    config_run_name_str = f"BERT_Best_{bert_model_name_config.split('/')[-1]}_DR{dropout_rate_config}_LR{learning_rate_config}_BS{batch_size_train_config}"
    print(f"\n--- Running Configuration: {config_run_name_str} ---")
    print(f"Device: {device}")

    # --- Data Loading and Preprocessing  ---
    print("Loading ATIS data...")
    data_dir_path = os.environ.get("ATIS_DATA_DIR", os.path.join("dataset", "ATIS"))
    if not os.path.exists(os.path.join(data_dir_path, "train.json")):
        data_dir_path = os.path.join("dataset", "ATIS")
        if not os.path.exists(os.path.join(data_dir_path, "train.json")):
            print(f"CRITICAL ERROR: Dataset not found at {data_dir_path} or ./dataset/ATIS.")
            exit()
            
    tmp_train_raw = load_data(os.path.join(data_dir_path, "train.json"))
    test_raw = load_data(os.path.join(data_dir_path, "test.json"))

    # Create dev set
    portion = 0.10
    intents_all = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents_all)
    labels_strat, inputs_strat, mini_train_strat = [], [], []
    for i, intent_val in enumerate(intents_all):
        if count_y[intent_val] > 1:
            inputs_strat.append(tmp_train_raw[i]); labels_strat.append(intent_val)
        else: mini_train_strat.append(tmp_train_raw[i])
    if len(inputs_strat) > 1 and len(set(labels_strat)) > 1:
        X_train_data, X_dev_data, _, _ = train_test_split(inputs_strat, labels_strat, test_size=portion, random_state=42, shuffle=True, stratify=labels_strat)
        train_raw = X_train_data + mini_train_strat; dev_raw = X_dev_data
    else:
        num_dev = int(len(tmp_train_raw) * portion)
        if num_dev > 0 and len(tmp_train_raw) - num_dev > 0: train_raw, dev_raw = train_test_split(tmp_train_raw, test_size=portion, random_state=42, shuffle=True)
        else: train_raw = tmp_train_raw; dev_raw = []
    
    # Create Lang object (for BERT, only manages label vocabularies)
    corpus_all_data = train_raw + (dev_raw if dev_raw else []) + test_raw
    slots_for_lang = set(sum([line['slots'].split() for line in corpus_all_data], []))
    intents_for_lang = set([line['intent'] for line in corpus_all_data])
    lang = Lang(intents_for_lang, slots_for_lang) # Adapted Lang for BERT
    
    # Initialize BERT Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_name_config)

    # Create Datasets (BERT specific)
    train_dataset = IntentsAndSlotsBERT(train_raw, lang, tokenizer, max_len=max_seq_length_config)
    dev_dataset = IntentsAndSlotsBERT(dev_raw, lang, tokenizer, max_len=max_seq_length_config) if dev_raw else None
    test_dataset = IntentsAndSlotsBERT(test_raw, lang, tokenizer, max_len=max_seq_length_config)
    print("Data preparation and Dataset creation finished.")

    # DataLoaders (BERT specific collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train_config, collate_fn=collate_fn_bert, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size_eval_config, collate_fn=collate_fn_bert) if dev_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size_eval_config, collate_fn=collate_fn_bert)

    # --- Model, Optimizer, Loss (BERT specific) ---
    print("Initializing BERT Model, Optimizer, and Loss criteria...")
    num_intent_labels_model = len(lang.intent2id)
    num_slot_labels_model = len(lang.slot2id)

    model = JointBERTModel(
        num_intent_labels=num_intent_labels_model,
        num_slot_labels=num_slot_labels_model,
        bert_model_name=bert_model_name_config,
        dropout_rate=dropout_rate_config
    ).to(device)
    # Note: No model.apply(init_weights) for BERT pre-trained body. New heads are auto-initialized.

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate_config) # AdamW for BERT
    criterion_intents = nn.CrossEntropyLoss()
    criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX) # Use IGNORE_INDEX
    print("Initialization complete.")

    # --- Training Loop (BERT specific) ---
    print("Starting fine-tuning...")
    best_f1_dev_run = 0.0
    current_patience_run = early_stop_patience_config
    best_model_state_dict_run = copy.deepcopy(model.state_dict())

    if not dev_loader:
        tqdm.write(f"Fine-tuning for {max_epochs_config} epochs (no dev set for early stopping)...")
        for epoch_idx in tqdm(range(1, max_epochs_config + 1), desc=f"Fine-tuning {config_run_name_str}", unit="epoch"):
            train_loop_bert(train_loader, optimizer, criterion_slots, criterion_intents, model, grad_clip_config)
        best_model_state_dict_run = copy.deepcopy(model.state_dict())
    else:
        epoch_pbar = tqdm(range(1, max_epochs_config + 1), desc=f"Fine-tuning {config_run_name_str}", unit="epoch")
        for epoch_num in epoch_pbar:
            train_loss_val = train_loop_bert(train_loader, optimizer, criterion_slots, criterion_intents, model, grad_clip_config)
            f1_dev_val, acc_dev_val, loss_dev_val = eval_loop_bert(dev_loader, criterion_slots, criterion_intents, model, lang, tokenizer) # Pass tokenizer to eval_loop_bert
            
            epoch_pbar.set_postfix({
                "TrainLoss": f"{train_loss_val:.4f}", "DevLoss": f"{loss_dev_val:.4f}",
                "DevF1": f"{f1_dev_val:.4f}", "DevAcc": f"{acc_dev_val:.4f}"
            })

            if f1_dev_val > best_f1_dev_run:
                best_f1_dev_run = f1_dev_val
                current_patience_run = early_stop_patience_config
                best_model_state_dict_run = copy.deepcopy(model.state_dict())
            else:
                current_patience_run -= 1
            if current_patience_run <= 0:
                tqdm.write(f"\nEarly stopping at epoch {epoch_num}.")
                break
        epoch_pbar.close()
    print("--- Fine-tuning Finished ---")
    
    # --- Final Evaluation on Test Set ---
    if best_model_state_dict_run:
        print("\nEvaluating best model on Test Set...")
        model.load_state_dict(best_model_state_dict_run)
        test_f1, test_acc, test_loss = eval_loop_bert(test_loader, criterion_slots, criterion_intents, model, lang, tokenizer) # Pass tokenizer
        
        print("\n--- FINAL TEST RESULTS (BERT Best Config) ---")
        results_display_data = [
            ["Metric", "Value"],
            ["Configuration", config_run_name_str],
            ["Test Slot F1", f"{test_f1:.4f}"],
            ["Test Intent Accuracy", f"{test_acc:.4f}"],
            ["Test Loss (average)", f"{test_loss:.4f}"]
        ]
        print(tabulate(results_display_data, headers="firstrow", tablefmt="grid"))
    else:
        print("CRITICAL ERROR: No model state available for final BERT testing.")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print(f"\n--- Script Finished: {config_run_name_str} ---")

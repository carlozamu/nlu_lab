
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

from utils import * # load_data, train_test_split, Lang, IntentsAndSlotsBERT, collate_fn_bert, device, IGNORE_INDEX
from model import JointBERTModel
try:
    from conll import evaluate as conll_evaluate
except ImportError:
    print("CRITICAL ERROR: 'conll.py' not found.")
    exit()

from transformers import BertTokenizerFast

# ==============================================================================
# --- Training and Evaluation Loops for BERT ---
# ==============================================================================

def train_loop_bert(data_loader, optimizer, criterion_slots, criterion_intents, model, grad_clip=1.0):
    """Executes one training epoch for the BERT model."""
    model.train()
    total_epoch_loss = 0.0
    for batch in tqdm(data_loader, desc="Training", leave=False, unit="batch"):
        optimizer.zero_grad()
        intent_logits, slot_logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )
        loss_intent = criterion_intents(intent_logits, batch["intent_labels"])
        loss_slot = criterion_slots(slot_logits.permute(0, 2, 1), batch["slot_labels"]) # Align for CrossEntropy
        current_loss = loss_intent + loss_slot
        total_epoch_loss += current_loss.item()
        current_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    return total_epoch_loss / len(data_loader) if data_loader else float("inf")

def eval_loop_bert(data_loader, criterion_slots, criterion_intents, model, lang_obj, tokenizer_obj):
    """Performs evaluation for the BERT model, handling slot alignment."""
    model.eval()
    total_epoch_loss = 0.0
    all_ref_intents, all_hyp_intents = [], []
    all_ref_slot_sequences, all_hyp_slot_sequences = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False, unit="batch"):
            intent_logits, slot_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )
            loss_intent = criterion_intents(intent_logits, batch["intent_labels"])
            loss_slot = criterion_slots(slot_logits.permute(0, 2, 1), batch["slot_labels"])
            total_epoch_loss += (loss_intent + loss_slot).item()

            pred_intent_ids = torch.argmax(intent_logits, dim=1).cpu().tolist()
            all_hyp_intents.extend([lang_obj.id2intent.get(idx, "UNK_INTENT") for idx in pred_intent_ids])
            all_ref_intents.extend([lang_obj.id2intent.get(idx, "UNK_INTENT") for idx in batch["intent_labels"].cpu().tolist()])

            pred_slot_ids_batch = torch.argmax(slot_logits, dim=-1).cpu().tolist()
            for i in range(batch["input_ids"].size(0)):
                word_ids_sample = batch["word_ids"][i]
                attn_mask_sample = batch["attention_mask"][i].cpu().tolist()
                original_slots_sample = batch["original_slot_labels"][i]
                original_words_sample = batch["original_words"][i]
                pred_slot_ids_sample = pred_slot_ids_batch[i]
                aligned_pred_slots = []
                last_word_idx = -1
                for token_idx, word_idx in enumerate(word_ids_sample):
                    if word_idx is not None and attn_mask_sample[token_idx] == 1:
                        if word_idx != last_word_idx: # First sub-token of a new word
                            if word_idx < len(original_words_sample):
                                aligned_pred_slots.append(lang_obj.id2slot.get(pred_slot_ids_sample[token_idx], "O"))
                        last_word_idx = word_idx
                
                min_len = min(len(original_words_sample), len(original_slots_sample), len(aligned_pred_slots))
                if min_len > 0: # Ensure alignment is possible
                    all_ref_slot_sequences.append(list(zip(original_words_sample[:min_len], original_slots_sample[:min_len])))
                    all_hyp_slot_sequences.append(list(zip(original_words_sample[:min_len], aligned_pred_slots[:min_len])))
    slot_f1 = 0.0
    if all_ref_slot_sequences and all_hyp_slot_sequences:
        try:
            slot_f1 = conll_evaluate(all_ref_slot_sequences, all_hyp_slot_sequences).get("total", {}).get("f", 0.0)
        except Exception: pass # Minimal error handling for CoNLL
    intent_acc = classification_report(all_ref_intents, all_hyp_intents, zero_division=0, output_dict=True).get("accuracy", 0.0)
    mean_epoch_loss = total_epoch_loss / len(data_loader) if data_loader else float("inf")
    return slot_f1, intent_acc, mean_epoch_loss

# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================

if __name__ == "__main__":
    
    # --- 1. Data Loading and Preprocessing ---
    data_dir = os.environ.get("ATIS_DATA_DIR", os.path.join("dataset", "ATIS"))
    if not os.path.exists(os.path.join(data_dir, "train.json")):
        print(f"CRITICAL ERROR: Dataset not found at {data_dir}.")
        exit()
    train_raw_full = load_data(os.path.join(data_dir, "train.json"))
    test_raw = load_data(os.path.join(data_dir, "test.json"))

    dev_portion = 0.10
    intents_list_full = [x["intent"] for x in train_raw_full]
    unique_intent_counts = Counter(intents_list_full)
    stratify_inputs, stratify_labels, single_instance_samples = [], [], []
    for i, intent_item in enumerate(intents_list_full):
        if unique_intent_counts[intent_item] > 1:
            stratify_inputs.append(train_raw_full[i]); stratify_labels.append(intent_item)
        else: single_instance_samples.append(train_raw_full[i])
    if len(stratify_inputs) > 1 and len(set(stratify_labels)) > 1:
        train_stratified, dev_stratified, _, _ = train_test_split(stratify_inputs, stratify_labels, test_size=dev_portion, random_state=42, shuffle=True, stratify=stratify_labels)
        train_raw = train_stratified + single_instance_samples; dev_raw = dev_stratified
    else:
        num_dev_samples = int(len(train_raw_full) * dev_portion)
        if num_dev_samples > 0 and len(train_raw_full) - num_dev_samples > 0: train_raw, dev_raw = train_test_split(train_raw_full, test_size=dev_portion, random_state=42, shuffle=True)
        else: train_raw = train_raw_full; dev_raw = []

    corpus_for_labels = train_raw + (dev_raw if dev_raw else []) + test_raw
    all_slot_labels = set(sum([s["slots"].split() for s in corpus_for_labels], []))
    all_intent_labels = set([s["intent"] for s in corpus_for_labels])
    lang_obj = Lang(all_intent_labels, all_slot_labels) # Lang for BERT: only labels

    # --- 2. BERT Fine-tuning Parameters (Best Configuration) ---
    BERT_MODEL_NAME = 'bert-base-uncased'; LEARNING_RATE = 5e-5; DROPOUT_RATE = 0.3
    BATCH_SIZE_TRAIN = 128; BATCH_SIZE_EVAL = 128; MAX_EPOCHS = 50
    EARLY_STOP_PATIENCE = 10; GRAD_CLIP = 1.0; MAX_SEQ_LENGTH = 50
    config_run_name = f"BestBERT_{BERT_MODEL_NAME.split('/')[-1]}_DR{DROPOUT_RATE}_LR{LEARNING_RATE}_BS{BATCH_SIZE_TRAIN}"
    print(f"\n--- Running: {config_run_name} ---")

    # --- 3. Tokenizer, Datasets, DataLoaders ---
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    train_dataset = IntentsAndSlotsBERT(train_raw, lang_obj, tokenizer, max_len=MAX_SEQ_LENGTH)
    dev_dataset = IntentsAndSlotsBERT(dev_raw, lang_obj, tokenizer, max_len=MAX_SEQ_LENGTH) if dev_raw else None
    test_dataset = IntentsAndSlotsBERT(test_raw, lang_obj, tokenizer, max_len=MAX_SEQ_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, collate_fn=collate_fn_bert, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE_EVAL, collate_fn=collate_fn_bert) if dev_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_EVAL, collate_fn=collate_fn_bert)

    # --- 4. Model, Optimizer, Criterion ---
    model = JointBERTModel(len(lang_obj.intent2id), len(lang_obj.slot2id), BERT_MODEL_NAME, DROPOUT_RATE).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion_intents = nn.CrossEntropyLoss()
    criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # --- 5. Training Loop ---
    best_dev_f1 = 0.0; patience_counter = EARLY_STOP_PATIENCE
    best_model_state_for_test = model.state_dict().copy()

    if not dev_loader:
        tqdm.write(f"Training for {MAX_EPOCHS} epochs (no dev set)...")
        for epoch in tqdm(range(1, MAX_EPOCHS + 1), desc=f"Training", unit="epoch"):
            train_loop_bert(train_loader, optimizer, criterion_slots, criterion_intents, model, GRAD_CLIP)
        best_model_state_for_test = model.state_dict().copy()
    else:
        epoch_pbar = tqdm(range(1, MAX_EPOCHS + 1), desc=f"Fine-tuning {BERT_MODEL_NAME.split('/')[-1]}", unit="epoch")
        for epoch in epoch_pbar:
            train_loss = train_loop_bert(train_loader, optimizer, criterion_slots, criterion_intents, model, GRAD_CLIP)
            dev_f1, dev_acc, dev_loss = eval_loop_bert(dev_loader, criterion_slots, criterion_intents, model, lang_obj, tokenizer)
            epoch_pbar.set_postfix({"TrainLoss": f"{train_loss:.4f}", "DevLoss": f"{dev_loss:.4f}", "DevF1": f"{dev_f1:.4f}", "DevAcc": f"{dev_acc:.4f}"})
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1; patience_counter = EARLY_STOP_PATIENCE; best_model_state_for_test = model.state_dict().copy()
            else:
                patience_counter -= 1
            if patience_counter <= 0: tqdm.write(f"\nEarly stopping at epoch {epoch}."); break
        epoch_pbar.close()
    
    # --- 6. Final Evaluation on Test Set ---
    if best_model_state_for_test:
        model.load_state_dict(best_model_state_for_test)
        test_f1, test_acc, test_loss = eval_loop_bert(test_loader, criterion_slots, criterion_intents, model, lang_obj, tokenizer)
        print("\n--- FINAL TEST RESULTS (BERT Best Config) ---")
        results_data = [["Metric", "Value"], ["Configuration", config_run_name], ["Test Slot F1", f"{test_f1:.4f}"], ["Test Intent Acc", f"{test_acc:.4f}"], ["Test Loss", f"{test_loss:.4f}"]]
        print(tabulate(results_data, headers="firstrow", tablefmt="grid"))
    else:
        print("CRITICAL ERROR: No model state for final BERT testing.")

    print(f"\nRun finished: {config_run_name}")


# (Bidirectional + Dropout: DR=0.5, LR=0.005, BS=128).

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

# Project-specific imports
from utils import PAD_TOKEN, Lang, IntentsAndSlots, collate_fn, init_weights, device, load_data, train_test_split
from model import ModelIAS
try:
    from conll import evaluate
except ImportError:
    # This error is critical, so a print is justified
    print("CRITICAL ERROR: 'conll.py' not found. Please ensure it is in the directory.")
    exit()

# ==============================================================================
# --- Training and Evaluation Loops (Unchanged from previous "best run" version) ---
# ==============================================================================

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5.0):
    """Executes one training epoch."""
    model.train()
    epoch_losses = []
    for batch in data:
        optimizer.zero_grad()
        slot_logits, intent_logits = model(batch["utterances"], batch["slots_len"])
        loss_intent = criterion_intents(intent_logits, batch["intents"])
        loss_slot = criterion_slots(slot_logits, batch["y_slots"])
        loss = loss_intent + loss_slot
        epoch_losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return np.mean(epoch_losses) if epoch_losses else float("inf")

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    """Performs evaluation on a dataset."""
    model.eval()
    epoch_losses = []
    ref_intents, hyp_intents = [], []
    ref_slots, hyp_slots = [], []
    with torch.no_grad():
        for batch in data:
            slot_logits, intent_logits = model(batch["utterances"], batch["slots_len"])
            loss_intent = criterion_intents(intent_logits, batch["intents"])
            loss_slot = criterion_slots(slot_logits, batch["y_slots"])
            loss = loss_intent + loss_slot
            epoch_losses.append(loss.item())
            out_intents_ids = torch.argmax(intent_logits, dim=1).cpu().tolist()
            hyp_intents.extend([lang.id2intent.get(idx, "UNK_INTENT") for idx in out_intents_ids])
            ref_intents.extend([lang.id2intent.get(idx, "UNK_INTENT") for idx in batch["intents"].cpu().tolist()])
            predicted_slot_ids = torch.argmax(slot_logits, dim=1)
            for i in range(predicted_slot_ids.size(0)):
                length = batch["slots_len"][i].item()
                utt_word_ids = batch["utterances"][i][:length].cpu().tolist()
                utterance_words = [lang.id2word.get(word_id, lang.word2id.get("unk", "UNK")) for word_id in utt_word_ids]
                true_slot_ids = batch["y_slots"][i][:length].cpu().tolist()
                true_slot_labels = [lang.id2slot.get(slot_id, lang.slot2id.get("pad", "UNK_SLOT")) for slot_id in true_slot_ids]
                pred_slot_ids_sample = predicted_slot_ids[i][:length].cpu().tolist()
                pred_slot_labels = [lang.id2slot.get(slot_id, lang.slot2id.get("pad", "UNK_SLOT")) for slot_id in pred_slot_ids_sample]
                ref_slots.append(list(zip(utterance_words, true_slot_labels)))
                hyp_slots.append(list(zip(utterance_words, pred_slot_labels)))
    try:
        slot_results = evaluate(ref_slots, hyp_slots)
        slot_f1 = slot_results.get("total", {}).get("f", 0.0)
    except Exception as e:
        # Keep a minimal warning for this critical evaluation step
        tqdm.write(f"\nWarning: CoNLL evaluation error: {e}")
        slot_f1 = 0.0
    intent_report = classification_report(ref_intents, hyp_intents, zero_division=0, output_dict=True)
    intent_acc = intent_report.get("accuracy", 0.0)
    mean_loss = np.mean(epoch_losses) if epoch_losses else float("inf")
    return slot_f1, intent_acc, mean_loss

# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================

if __name__ == "__main__":
    
    # --- 1. Data Loading and Preprocessing ---
    # Minimal print statements during setup
    data_dir = os.environ.get("ATIS_DATA_DIR", os.path.join("dataset", "ATIS"))
    if not os.path.exists(os.path.join(data_dir, "train.json")): # Check one file
        print(f"CRITICAL ERROR: Dataset not found at {data_dir}. Please set ATIS_DATA_DIR or place data in ./dataset/ATIS")
        exit()
        
    train_raw_full = load_data(os.path.join(data_dir, "train.json"))
    test_raw = load_data(os.path.join(data_dir, "test.json"))

    dev_portion = 0.10
    intents_list_full = [x["intent"] for x in train_raw_full]
    unique_intent_counts = Counter(intents_list_full)
    
    stratify_inputs, stratify_labels, single_instance_samples = [], [], []
    for i, intent_item in enumerate(intents_list_full):
        if unique_intent_counts[intent_item] > 1:
            stratify_inputs.append(train_raw_full[i])
            stratify_labels.append(intent_item)
        else:
            single_instance_samples.append(train_raw_full[i])
            
    if len(stratify_inputs) > 1 and len(set(stratify_labels)) > 1 :
        train_stratified, dev_stratified, _, _ = train_test_split(
            stratify_inputs, stratify_labels,
            test_size=dev_portion, random_state=42, shuffle=True, stratify=stratify_labels
        )
        train_raw = train_stratified + single_instance_samples
        dev_raw = dev_stratified
    else:
        num_dev_samples = int(len(train_raw_full) * dev_portion)
        if num_dev_samples > 0 and len(train_raw_full) - num_dev_samples > 0 :
            train_raw, dev_raw = train_test_split(train_raw_full, test_size=dev_portion, random_state=42, shuffle=True)
        else:
            train_raw = train_raw_full
            dev_raw = [] # Dev set might be empty if dataset is too small

    words_for_vocab = sum([sample["utterance"].split() for sample in train_raw], [])
    corpus_for_labels = train_raw + (dev_raw if dev_raw else []) + test_raw
    all_slot_labels = set(sum([sample["slots"].split() for sample in corpus_for_labels], []))
    all_intent_labels = set([sample["intent"] for sample in corpus_for_labels])
    lang = Lang(words_for_vocab, all_intent_labels, all_slot_labels, cutoff=0)

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang) if dev_raw else None
    test_dataset = IntentsAndSlots(test_raw, lang)

    # --- 2. BEST LSTM Configuration Parameters ---
    LEARNING_RATE = 0.005
    DROPOUT_RATE = 0.5
    BATCH_SIZE_TRAIN = 128
    HIDDEN_SIZE = 200
    EMBEDDING_SIZE = 300
    GRAD_CLIP = 5.0
    MAX_EPOCHS = 50 # Adjust as needed; original grid search used more
    EARLY_STOP_PATIENCE = 10
    IS_BIDIRECTIONAL = True
    config_name = f"BestLSTM_DR{DROPOUT_RATE}_LR{LEARNING_RATE}_BS{BATCH_SIZE_TRAIN}"

    # --- 3. DataLoader Creation ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn) if dev_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # --- 4. Model, Optimizer, Criterion ---
    model = ModelIAS(
        hidden_size=HIDDEN_SIZE, out_slot=len(lang.slot2id), out_int=len(lang.intent2id),
        emb_size=EMBEDDING_SIZE, vocab_len=len(lang.word2id), pad_index=PAD_TOKEN,
        bidirectional=IS_BIDIRECTIONAL, dropout_rate=DROPOUT_RATE
    ).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    # --- 5. Training Loop ---
    best_dev_f1 = 0.0
    patience_counter = EARLY_STOP_PATIENCE
    best_model_state_for_test = model.state_dict().copy() # Initialize with initial weights

    if not dev_loader: # No dev set, train for all epochs
        tqdm.write(f"Training {config_name} for {MAX_EPOCHS} epochs (no dev set for early stopping)...") # tqdm.write for consistency
        for epoch in tqdm(range(1, MAX_EPOCHS + 1), desc=f"Training {config_name}"):
            train_loss_epoch = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, GRAD_CLIP)
            # Optionally, update tqdm postfix if desired, e.g. tqdm.set_postfix({"TrainLoss": f"{train_loss_epoch:.4f}"})
        best_model_state_for_test = model.state_dict().copy()
    else: # Normal training with early stopping
        epoch_pbar = tqdm(range(1, MAX_EPOCHS + 1), desc=f"Training {config_name}")
        for epoch in epoch_pbar:
            train_loss_epoch = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, GRAD_CLIP)
            dev_f1_epoch, dev_acc_epoch, dev_loss_epoch = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
            
            epoch_pbar.set_postfix({
                "TrainLoss": f"{train_loss_epoch:.4f}", "DevLoss": f"{dev_loss_epoch:.4f}", 
                "DevF1": f"{dev_f1_epoch:.4f}", "DevAcc": f"{dev_acc_epoch:.4f}"
            })

            if dev_f1_epoch > best_dev_f1:
                best_dev_f1 = dev_f1_epoch
                patience_counter = EARLY_STOP_PATIENCE
                best_model_state_for_test = model.state_dict().copy()
            else:
                patience_counter -= 1
            
            if patience_counter <= 0:
                tqdm.write(f"\nEarly stopping at epoch {epoch}.") # Use tqdm.write
                break
        epoch_pbar.close()
    
    # --- 6. Final Evaluation on Test Set ---
    if best_model_state_for_test:
        model.load_state_dict(best_model_state_for_test)
        test_f1_final, test_acc_final, test_loss_final = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        
        print("\n--- FINAL TEST RESULTS ---") # This final print is important
        results_data = [
            ["Configuration", config_name],
            ["Test Slot F1", f"{test_f1_final:.4f}"],
            ["Test Intent Accuracy", f"{test_acc_final:.4f}"],
            ["Test Loss", f"{test_loss_final:.4f}"]
        ]
        print(tabulate(results_data, headers="firstrow", tablefmt="grid"))
    else:
        print("CRITICAL ERROR: No model state available for final testing.") # Critical error print

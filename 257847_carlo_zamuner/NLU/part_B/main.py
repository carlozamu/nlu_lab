import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
from tabulate import tabulate
import copy
from functions import train_loop_bert, eval_loop_bert
from utils import (
    device,
    IGNORE_INDEX,
    Lang,
    IntentsAndSlotsBERT,
    collate_fn_bert,
    load_data,
)
from model import JointBERTModel
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # ========== HYPERPARAMETERS ==========
    # These control the model architecture and training process
    BERT_MODEL_NAME = "bert-base-uncased"  # Pre-trained BERT model to use
    LEARNING_RATE = 5e-5                   # Learning rate for Adam optimizer
    DROPOUT_RATE = 0.3                     # Dropout for regularization
    BATCH_SIZE_TRAIN = 128                 # Training batch size
    BATCH_SIZE_EVAL = 128                  # Evaluation batch size
    MAX_EPOCHS = 50                        # Maximum training epochs
    EARLY_STOP_PATIENCE = 10               # Stop training if no improvement for N epochs
    CLIP = 1.0                            # Gradient clipping value
    MAX_SEQ_LENGTH = 50                   # Maximum sequence length for BERT

    config_name = f"BERT_Best_{BERT_MODEL_NAME.split('/')[-1]}_DR{DROPOUT_RATE}_LR{LEARNING_RATE}_BS{BATCH_SIZE_TRAIN}"
    print(f"\n--- Running: {config_name} (Device: {device}) ---")

    # ========== DATA LOADING AND PREPROCESSING ==========
    # Load ATIS dataset - a benchmark for intent classification and slot filling
    data_dir = os.environ.get("ATIS_DATA_DIR", os.path.join("dataset", "ATIS"))
    if not os.path.exists(os.path.join(data_dir, "train.json")):
        print(f"CRITICAL ERROR: Dataset not found at {data_dir}.")
        exit()
    
    tmp_train_raw = load_data(os.path.join(data_dir, "train.json"))
    test_raw = load_data(os.path.join(data_dir, "test.json"))

    # Create train/dev split with stratification to maintain class balance
    dev_portion = 0.10  # 10% for development set
    intents_full = [x["intent"] for x in tmp_train_raw]
    intent_counts = Counter(intents_full)
    
    # Separate intents with multiple instances from single instances
    strat_inputs, strat_labels, single_instance = [], [], []
    for i, intent_val in enumerate(intents_full):
        if intent_counts[intent_val] > 1:
            strat_inputs.append(tmp_train_raw[i])
            strat_labels.append(intent_val)
        else:
            single_instance.append(tmp_train_raw[i])

    # Stratified split to maintain intent distribution in train/dev
    if len(strat_inputs) > 1 and len(set(strat_labels)) > 1:
        train_s, dev_s, _, _ = train_test_split(
            strat_inputs,
            strat_labels,
            test_size=dev_portion,
            random_state=42,
            shuffle=True,
            stratify=strat_labels,
        )
        train_raw = train_s + single_instance  # Add single instances to training
        dev_raw = dev_s
    else:
        # Fallback: simple random split if stratification not possible
        num_dev_fb = int(len(tmp_train_raw) * dev_portion)
        if num_dev_fb > 0 and len(tmp_train_raw) - num_dev_fb > 0:
            train_raw, dev_raw = train_test_split(
                tmp_train_raw, test_size=dev_portion, random_state=42, shuffle=True
            )
        else:
            train_raw = tmp_train_raw
            dev_raw = []

    # Build vocabularies from all data (train + dev + test)
    corpus_labels = train_raw + (dev_raw if dev_raw else []) + test_raw
    slots_vocab = set(sum([line["slots"].split() for line in corpus_labels], []))
    intents_vocab = set([line["intent"] for line in corpus_labels])

    # Create language object that maps labels to IDs
    lang = Lang(intents_vocab, slots_vocab)

    # ========== TOKENIZER, DATASETS, DATALOADERS ==========
    # Initialize BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    # Create datasets that handle BERT tokenization and alignment
    train_dataset = IntentsAndSlotsBERT(
        train_raw, lang, tokenizer, max_len=MAX_SEQ_LENGTH
    )
    dev_dataset = (
        IntentsAndSlotsBERT(dev_raw, lang, tokenizer, max_len=MAX_SEQ_LENGTH)
        if dev_raw
        else None
    )
    test_dataset = IntentsAndSlotsBERT(
        test_raw, lang, tokenizer, max_len=MAX_SEQ_LENGTH
    )

    # Create data loaders for batching and shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        collate_fn=collate_fn_bert,
        shuffle=True,
    )
    dev_loader = (
        DataLoader(dev_dataset, batch_size=BATCH_SIZE_EVAL, collate_fn=collate_fn_bert)
        if dev_dataset
        else None
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE_EVAL, collate_fn=collate_fn_bert
    )

    # ========== MODEL, OPTIMIZER, LOSS ==========
    # Initialize the joint BERT model for intent + slot prediction
    model = JointBERTModel(
        num_intent_labels=len(lang.intent2id),
        num_slot_labels=len(lang.slot2id),
        bert_model_name=BERT_MODEL_NAME,
        dropout_rate=DROPOUT_RATE,
    ).to(device)
    
    # AdamW optimizer (Adam with weight decay) - good for transformers
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Loss functions for the two tasks
    criterion_intents = nn.CrossEntropyLoss()  # Standard classification loss
    criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)  # Ignore padding tokens

    # ========== TRAINING LOOP ==========
    # Training with early stopping based on development set performance
    best_f1_dev = 0.0
    patience_counter = EARLY_STOP_PATIENCE
    best_model_state = copy.deepcopy(model.state_dict())

    if not dev_loader:
        # No development set: train for full epochs without early stopping
        tqdm.write(
            f"Fine-tuning for {MAX_EPOCHS} epochs (no dev set for early stopping)..."
        )
        for epoch in tqdm(range(1, MAX_EPOCHS + 1), desc=f"Fine-tuning", unit="epoch"):
            train_loop_bert(
                train_loader,
                optimizer,
                criterion_slots,
                criterion_intents,
                model,
                CLIP,
            )
        best_model_state = copy.deepcopy(model.state_dict())
    else:
        # Training with development set for early stopping
        epoch_pbar = tqdm(
            range(1, MAX_EPOCHS + 1),
            desc=f"Fine-tuning {BERT_MODEL_NAME.split('/')[-1]}",
            unit="epoch",
        )
        
        for epoch_num in epoch_pbar:
            # Training phase
            train_loss = train_loop_bert(
                train_loader,
                optimizer,
                criterion_slots,
                criterion_intents,
                model,
                CLIP,
            )
            
            # Evaluation phase
            dev_f1, dev_acc, dev_loss = eval_loop_bert(
                dev_loader, criterion_slots, criterion_intents, model, lang, tokenizer
            )

            # Update progress bar with current metrics
            epoch_pbar.set_postfix(
                {
                    "TrainLoss": f"{train_loss:.4f}",
                    "DevLoss": f"{dev_loss:.4f}",
                    "DevF1": f"{dev_f1:.4f}",
                    "DevAcc": f"{dev_acc:.4f}",
                }
            )
            
            # Early stopping logic: save best model based on F1 score
            if dev_f1 > best_f1_dev:
                best_f1_dev = dev_f1
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = EARLY_STOP_PATIENCE  # Reset patience
            else:
                patience_counter -= 1
                
            if patience_counter <= 0:
                tqdm.write(f"\nEarly stopping at epoch {epoch_num}.")
                break
        
        epoch_pbar.close()

    # ========== FINAL EVALUATION ON TEST SET ==========
    # Load the best model and evaluate on test set
    if best_model_state:
        model.load_state_dict(best_model_state)
        test_f1, test_acc, test_loss = eval_loop_bert(
            test_loader, criterion_slots, criterion_intents, model, lang, tokenizer
        )

        # Display final results in a nice table format
        print("\n--- FINAL TEST RESULTS (BERT Best Config) ---")
        results_data = [
            ["Metric", "Value"],
            ["Configuration", config_name],
            ["Test Slot F1", f"{test_f1:.4f}"],        # F1 for entity extraction
            ["Test Intent Accuracy", f"{test_acc:.4f}"],  # Accuracy for intent classification
            ["Test Loss (average)", f"{test_loss:.4f}"],
        ]
        print(tabulate(results_data, headers="firstrow", tablefmt="grid"))
    else:
        print("CRITICAL ERROR: No model state available for final BERT testing.")

    print(f"\n--- Script Finished: {config_name} ---")
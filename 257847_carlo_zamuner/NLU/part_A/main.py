# --- START OF FILE main_lstm_best_notebook_style.py ---
# This script executes a single run of the best LSTM configuration for ATIS
# (Bidirectional + Dropout: DR=0.5, LR=0.005, BS=128).
# The main execution block is styled to resemble the original
# '05_intent_and_slot_filling.py' notebook, while relying on
# imported classes and functions from utils.py, model.py, and functions.py.

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
from sklearn.model_selection import train_test_split
from utils import device, PAD_TOKEN, Lang, IntentsAndSlots, collate_fn, load_data, train_test_split
from model import ModelIAS
from functions import init_weights, train_loop, eval_loop
from conll import evaluate 
from pprint import pprint

# ==============================================================================
# --- MAIN EXECUTION BLOCK  ---
# ==============================================================================
if __name__ == "__main__":

    hid_size = 200
    emb_size = 300
    lr = 0.005
    clip = 5.0
    n_epochs = 50       
    patience = 10       
    
    is_bidirectional_model = True
    dropout_rate_model = 0.5
    batch_size_model_train = 128 
    
    config_name_run = f"LSTM_Best_DR{dropout_rate_model}_LR{lr}_BS{batch_size_model_train}"
    print(f"\n--- Running Configuration: {config_name_run} ---")
    print(f"Device: {device}") # Print device used

    # --- Data Loading and Preprocessing  --- 
    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))

    # Create dev set 
    portion = 0.10

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]

    # Intent distributions
    print('Train:')
    pprint({k:round(v/len(y_train),3)*100 for k, v in sorted(Counter(y_train).items())})
    print('Dev:'), 
    pprint({k:round(v/len(y_dev),3)*100 for k, v in sorted(Counter(y_dev).items())})
    print('Test:') 
    pprint({k:round(v/len(y_test),3)*100 for k, v in sorted(Counter(y_test).items())})
    print('='*89)
    # Dataset size
    print('TRAIN size:', len(train_raw))
    print('DEV size:', len(dev_raw))
    print('TEST size:', len(test_raw))
    
    # Create Lang object 
    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not want unk labels, however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(words, intents, slots, cutoff=0)

    # Create Datasets (as in notebook)
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang) if dev_raw else None
    test_dataset = IntentsAndSlots(test_raw, lang)
    print("Data preparation and Dataset creation finished.")

    # DataLoaders 
    train_loader = DataLoader(train_dataset, batch_size=batch_size_model_train, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn) if dev_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # --- Model, Optimizer, Loss ---
    print("Initializing Model, Optimizer, and Loss criteria...")
    out_slot_size = len(lang.slot2id)
    out_int_size = len(lang.intent2id)
    vocab_len_size = len(lang.word2id)

    model = ModelIAS( 
        hidden_size=hid_size, 
        num_slot_classes=out_slot_size,  
        num_intent_classes=out_int_size, 
        embedding_size=emb_size,   
        vocab_size=vocab_len_size,     
        pad_token_idx=PAD_TOKEN,
        is_bidirectional=is_bidirectional_model,
        dropout_p=dropout_rate_model
    ).to(device)
    model.apply(init_weights) 

    optimizer = optim.Adam(model.parameters(), lr=lr) 
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN) 
    criterion_intents = nn.CrossEntropyLoss() 
    print("Initialization complete.")

    # --- Training Loop ---
    runs = 5
    slot_f1s, intent_acc = [], []
    
    for x in tqdm(range(0, runs)):
        model = ModelIAS(hid_size, out_slot_size, out_int_size, emb_size, 
                        vocab_len_size, pad_index=PAD_TOKEN).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        patience_counter = patience
        best_f1 = 0
        best_model_state = None

        for epoch in range(1, n_epochs + 1):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model)
            
            if epoch % 5 == 0:
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    best_f1 = f1
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = patience
                else:
                    patience_counter -= 1
                    
                if patience_counter <= 0:  # Early stopping
                    break

        # Load best model and evaluate on test set
        if best_model_state:
            model.load_state_dict(best_model_state)
            results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                   criterion_intents, model, lang)
            intent_acc.append(intent_test['accuracy'])
            slot_f1s.append(results_test['total']['f'])

    # Calculate and print final results
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    
    print("\n--- FINAL RESULTS ---")
    results_display_data = [
        ["Metric", "Mean", "Std"],
        ["Slot F1", f"{slot_f1s.mean():.3f}", f"{slot_f1s.std():.3f}"],
        ["Intent Accuracy", f"{intent_acc.mean():.3f}", f"{intent_acc.std():.3f}"]
    ]
    print(tabulate(results_display_data, headers="firstrow", tablefmt="grid"))
    print(f"\n--- Script Finished: {config_name_run} ---")
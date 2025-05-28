import os
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
from tabulate import tabulate
import copy
from utils import (
    device,
    PAD_TOKEN,
    Lang,
    IntentsAndSlots,
    collate_fn,
    load_data,
    train_test_split,
)
from model import ModelIAS
from functions import init_weights, train_loop, eval_loop

if __name__ == "__main__":
    HID_SIZE = 200
    EMB_SIZE = 300
    LR = 0.005
    CLIP = 5.0
    N_EPOCHS = 50
    PATIENCE = 10
    IS_BIDIRECTIONAL = True
    DROPOUT_RATE = 0.5
    BATCH_SIZE_TRAIN = 128

    config_name = f"LSTM_Best_DR{DROPOUT_RATE}_LR{LR}_BS{BATCH_SIZE_TRAIN}"
    print(f"\n--- Running: {config_name} (Device: {device}) ---")

    # --- Data Loading and Preprocessing ---
    data_dir = os.environ.get("ATIS_DATA_DIR", os.path.join("dataset", "ATIS"))
    if not os.path.exists(os.path.join(data_dir, "train.json")):
        print(f"CRITICAL ERROR: Dataset not found at {data_dir}.")
        exit()
    tmp_train_raw = load_data(os.path.join(data_dir, "train.json"))
    test_raw = load_data(os.path.join(data_dir, "test.json"))

    dev_portion = 0.10
    intents_full = [x["intent"] for x in tmp_train_raw]
    intent_counts = Counter(intents_full)
    strat_inputs, strat_labels, single_instance = [], [], []
    for i, intent_val in enumerate(intents_full):
        if intent_counts[intent_val] > 1:
            strat_inputs.append(tmp_train_raw[i])
            strat_labels.append(intent_val)
        else:
            single_instance.append(tmp_train_raw[i])

    # Assuming stratification is always possible for ATIS or a simple split is acceptable
    if len(strat_inputs) > 1 and len(set(strat_labels)) > 1:
        train_s, dev_s, _, _ = train_test_split(
            strat_inputs,
            strat_labels,
            test_size=dev_portion,
            random_state=42,
            shuffle=True,
            stratify=strat_labels,
        )
        train_raw = train_s + single_instance
        dev_raw = dev_s
    else:  # Fallback for very small/uniform datasets (less likely for ATIS)
        train_raw, dev_raw = train_test_split(
            tmp_train_raw, test_size=dev_portion, random_state=42, shuffle=True
        )
        if not dev_raw:  # If dev_raw is still empty, this is problematic
            print(
                "Warning: Development set could not be created effectively. Results might be unreliable."
            )
            dev_raw = test_raw[
                : max(1, int(len(test_raw) * 0.1))
            ]  # Create a tiny dev from test as last resort - not ideal

    words_vocab = sum([x["utterance"].split() for x in train_raw], [])
    corpus_labels = train_raw + dev_raw + test_raw  # dev_raw is now assumed to exist
    slots_vocab = set(sum([line["slots"].split() for line in corpus_labels], []))
    intents_vocab = set([line["intent"] for line in corpus_labels])
    lang = Lang(words_vocab, intents_vocab, slots_vocab, cutoff=0)

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)  # dev_dataset is assumed to exist
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_TRAIN, collate_fn=collate_fn, shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=64, collate_fn=collate_fn
    )  # Eval batch size
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # --- Training (Multiple Runs) ---
    num_runs = 5
    all_slot_f1s, all_intent_accs = [], []

    for run_idx in tqdm(range(num_runs), desc="Total Runs"):
        tqdm.write(f"\n--- Run {run_idx + 1}/{num_runs} ---")
        model = ModelIAS(
            hidden_size=HID_SIZE,
            num_slot_classes=len(lang.slot2id),
            num_intent_classes=len(lang.intent2id),
            embedding_size=EMB_SIZE,
            vocab_size=len(lang.word2id),
            pad_token_idx=PAD_TOKEN,
            is_bidirectional=IS_BIDIRECTIONAL,
            dropout_p=DROPOUT_RATE,
        ).to(device)
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        current_patience = PATIENCE
        best_f1_run = 0.0
        best_model_state_run = copy.deepcopy(model.state_dict())

        epoch_pbar = tqdm(
            range(1, N_EPOCHS + 1), desc=f"Run {run_idx + 1} Epochs", leave=False
        )
        for epoch_num in epoch_pbar:
            train_losses = train_loop(
                train_loader, optimizer, criterion_slots, criterion_intents, model, CLIP
            )
            avg_train_loss = np.mean(train_losses) if train_losses else float("inf")

            dev_slots_res, dev_intent_rep, dev_avg_loss = eval_loop(
                dev_loader, criterion_slots, criterion_intents, model, lang
            )
            dev_f1 = dev_slots_res.get("total", {}).get("f", 0.0)
            dev_acc = dev_intent_rep.get("accuracy", 0.0)

            epoch_pbar.set_postfix(
                {
                    "TrainLoss": f"{avg_train_loss:.4f}",
                    "DevLoss": f"{dev_avg_loss:.4f}",
                    "DevF1": f"{dev_f1:.4f}",
                    "DevAcc": f"{dev_acc:.4f}",
                }
            )
            if dev_f1 > best_f1_run:
                best_f1_run = dev_f1
                best_model_state_run = copy.deepcopy(model.state_dict())
                current_patience = PATIENCE
            else:
                current_patience -= 1
            if current_patience <= 0:
                tqdm.write(f"Run {run_idx + 1}: Early stopping epoch {epoch_num}.")
                break
        epoch_pbar.close()

        if best_model_state_run:
            model.load_state_dict(best_model_state_run)
            test_slots_res, test_intent_rep, _ = eval_loop(
                test_loader, criterion_slots, criterion_intents, model, lang
            )
            all_intent_accs.append(test_intent_rep.get("accuracy", 0.0))
            all_slot_f1s.append(test_slots_res.get("total", {}).get("f", 0.0))
        else:
            all_intent_accs.append(0.0)
            all_slot_f1s.append(0.0)

    # --- Final Results ---
    final_slot_f1s_np = np.asarray(all_slot_f1s)
    final_intent_accs_np = np.asarray(all_intent_accs)

    print("\n--- FINAL AVG RESULTS (Multiple Runs) ---")
    results_table_data = [
        ["Metric", "Mean", "Std"],
        [
            "Slot F1",
            f"{final_slot_f1s_np.mean():.4f}",
            f"{final_slot_f1s_np.std():.4f}",
        ],
        [
            "Intent Accuracy",
            f"{final_intent_accs_np.mean():.4f}",
            f"{final_intent_accs_np.std():.4f}",
        ],
    ]
    print(tabulate(results_table_data, headers="firstrow", tablefmt="grid"))
    print(f"\n--- Script Finished: {config_name} ---")

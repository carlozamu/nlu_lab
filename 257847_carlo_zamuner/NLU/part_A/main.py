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
    hid_size = 200
    emb_size = 300
    lr = 0.005
    clip = 5
    n_epochs = 50
    patience = 10
    bidirectional = True
    dropout = 0.5
    batch_size = 128
    print(
        f"\n--- Running with bidirectional={bidirectional}, dropout={dropout} (Device: {device}) ---"
    )
    tmp_train_raw = load_data(os.path.join("dataset", "ATIS", "train.json"))
    test_raw = load_data(os.path.join("dataset", "ATIS", "test.json"))
    print("Train samples:", len(tmp_train_raw))
    print("Test samples:", len(test_raw))
    portion = 0.10
    intents = [x["intent"] for x in tmp_train_raw]
    count_y = Counter(intents)
    labels = []
    inputs = []
    mini_train = []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    X_train, X_dev, y_train, y_dev = train_test_split(
        inputs,
        labels,
        test_size=portion,
        random_state=42,
        shuffle=True,
        stratify=labels,
    )
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev
    words = sum([x["utterance"].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    slots = set(sum([line["slots"].split() for line in corpus], []))
    intents = set([line["intent"] for line in corpus])
    lang = Lang(words, intents, slots, cutoff=0)
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
    )
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    runs = 5
    slot_f1s, intent_acc = [], []
    for run_idx in tqdm(range(0, runs), desc="Total Runs"):
        print(f"\n--- Run {run_idx + 1}/{runs} ---")
        model = ModelIAS(
            hid_size,
            out_slot,
            out_int,
            emb_size,
            vocab_len,
            pad_index=PAD_TOKEN,
            bidirectional=bidirectional,
            dropout=dropout,
        ).to(device)
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()
        patience_count = patience
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        epoch_pbar = tqdm(
            range(1, n_epochs + 1), desc=f"Run {run_idx + 1} Epochs", leave=False
        )
        for x in epoch_pbar:
            loss = train_loop(
                train_loader,
                optimizer,
                criterion_slots,
                criterion_intents,
                model,
                clip=clip,
            )
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(
                dev_loader, criterion_slots, criterion_intents, model, lang
            )
            losses_dev.append(np.asarray(loss_dev).mean())
            f1 = results_dev["total"]["f"]
            epoch_pbar.set_postfix(
                {
                    "F1": f"{f1:.3f}",
                    "Acc": f"{intent_res['accuracy']:.3f}",
                    "Loss": f"{np.mean(loss):.3f}",
                }
            )
            if f1 > best_f1:
                best_f1 = f1
                patience_count = patience
            else:
                patience_count -= 1
            if patience_count <= 0:
                print(f"Early stopping at epoch {x} with best F1: {best_f1:.3f}")
                break
        results_test, intent_test, _ = eval_loop(
            test_loader, criterion_slots, criterion_intents, model, lang
        )
        intent_acc.append(intent_test["accuracy"])
        slot_f1s.append(results_test["total"]["f"])
    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print("\n--- FINAL RESULTS ---")
    print("Slot F1", round(slot_f1s.mean(), 3), "+-", round(slot_f1s.std(), 3))
    print("Intent Acc", round(intent_acc.mean(), 3), "+-", round(intent_acc.std(), 3))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import copy
from functools import partial
import os
from tqdm import tqdm

from functions import train_loop, eval_loop, init_weights
from utils import read_file, get_vocab, Lang, PennTreeBank, collate_fn
from model import LM_LSTM_With_Dropout

if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    experiment_step = "Part1A_Step3_Best"
    model_save_dir = "bin"

    lr = 0.001
    batch_size = 16
    hid = 300
    emb = 200
    clip = 5
    n_epochs = 100
    patience = 3
    emb_dropout = 0.1
    fc_dropout = 0.1

    train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)
    print("Data loaded.")

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    print(f"\n--- Running Configuration: {experiment_step} ---")
    print(f"   LR={lr}, BS={batch_size}, Optim=AdamW")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(
            collate_fn, pad_token_id=lang.word2id["<pad>"], device=DEVICE
        ),
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size * 2,
        collate_fn=partial(
            collate_fn, pad_token_id=lang.word2id["<pad>"], device=DEVICE
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        collate_fn=partial(
            collate_fn, pad_token_id=lang.word2id["<pad>"], device=DEVICE
        ),
    )

    print("Initializing Model, Optimizer, Loss...")
    model = LM_LSTM_With_Dropout(
        emb_size=emb,
        hidden_size=hid,
        output_size=vocab_len,
        pad_index=lang.word2id["<pad>"],
        emb_dropout=emb_dropout,
        fc_dropout=fc_dropout,
    ).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    print("Starting Training...")
    patience_counter = patience
    best_ppl = math.inf
    best_model = None
    sampled_epochs = []
    losses_train = []
    losses_dev = []

    pbar = tqdm(range(1, n_epochs + 1))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(loss)
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(loss_dev)
            pbar.set_description("PPL: %f" % ppl_dev)

            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to("cpu")
                patience_counter = patience
            else:
                patience_counter -= 1

            if patience_counter <= 0:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    if best_model is not None:
        best_model.to(DEVICE)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print("Test ppl: ", final_ppl)

        # Save the best model
        # print("\nSaving best model...")
        # os.makedirs(model_save_dir, exist_ok=True)
        # model_filename = f"{experiment_step}_lr{lr}_bs{batch_size}.pt"
        # model_save_path = os.path.join(model_save_dir, model_filename)
        # try:
        # torch.save(best_model.state_dict(), model_save_path)
        # print(f"Model saved to: {model_save_path}")
        # except Exception as e:
        # print(f"Error saving model: {e}")
    else:
        print("\nNo best model found.")

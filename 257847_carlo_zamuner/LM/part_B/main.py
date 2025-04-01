from functions import *
from utils import *
from model import *
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from tabulate import tabulate
from functools import partial
import math
import numpy as np
import csv
from torch.utils.data import DataLoader
import os

if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=128,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),
    )

    fixed_lr = 0.001
    fixed_wd = 0.01
    n_epochs = 100
    vocab_len = len(lang.word2id)
    hid_size = emb_size = 400
    clip = 5

    emb_dropouts = [0.1, 0.3, 0.5]
    fc_dropouts = [0.1, 0.3, 0.5]
    alphas = [0.8, 0.9, 0.95]

    for emb_dp in emb_dropouts:
        for fc_dp in fc_dropouts:
            for alpha in alphas:
                print(
                    f"\n=== Running: emb_dp={emb_dp}, fc_dp={fc_dp}, alpha={alpha} ==="
                )

                model = LM_LSTM_Dropout_Tied(
                    emb_size,
                    hid_size,
                    vocab_len,
                    pad_index=lang.word2id["<pad>"],
                    emb_dropout=emb_dp,
                    fc_dropout=fc_dp,
                ).to(DEVICE)
                model.apply(init_weights)

                optimizer = optim.AdamW(
                    model.parameters(), lr=fixed_lr, weight_decay=fixed_wd
                )
                criterion_train = nn.CrossEntropyLoss(
                    ignore_index=lang.word2id["<pad>"]
                )
                criterion_eval = nn.CrossEntropyLoss(
                    ignore_index=lang.word2id["<pad>"], reduction="sum"
                )

                patience = 3
                sampled_epochs = []
                losses_train = []
                losses_dev = []
                ppls_dev = []
                best_dev_ppl = math.inf
                best_model = None
                avg_model = None

                pbar = tqdm(range(1, n_epochs + 1), desc="Training")
                for epoch in pbar:
                    train_loss = train_loop(
                        train_loader, optimizer, criterion_train, model, clip=clip
                    )
                    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

                    if epoch == 1:
                        avg_model = copy.deepcopy(model)
                    else:
                        for param, avg_param in zip(
                            model.parameters(), avg_model.parameters()
                        ):
                            avg_param.data = (
                                alpha * avg_param.data + (1 - alpha) * param.data
                            )

                    ppl_avg, _ = eval_loop(dev_loader, criterion_eval, avg_model)
                    pbar.set_description(f"Avg PPL: {ppl_avg:.2f}")

                    sampled_epochs.append(epoch)
                    losses_train.append(train_loss)
                    losses_dev.append(loss_dev)
                    ppls_dev.append(ppl_avg)

                    if ppl_avg < best_dev_ppl:
                        best_dev_ppl = ppl_avg
                        best_model = copy.deepcopy(avg_model).to("cpu")
                        patience = 3
                    else:
                        patience -= 1
                        if patience <= 0:
                            print("Early stopping triggered.")
                            break

                best_model.to(DEVICE)
                final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
                print(
                    f"Final Test PPL (emb_dp={emb_dp}, fc_dp={fc_dp}, alpha={alpha}): {final_ppl}"
                )

                exp_dir = f"results/regularized/emb{emb_dp}_fc{fc_dp}_alpha{alpha}"
                os.makedirs(exp_dir, exist_ok=True)

                with open(
                    os.path.join(exp_dir, "training_summary.csv"), "w", newline=""
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(["Epoch", "Train Loss", "Dev Loss", "Dev PPL"])
                    for i in range(len(sampled_epochs)):
                        writer.writerow(
                            [
                                sampled_epochs[i],
                                losses_train[i],
                                losses_dev[i],
                                ppls_dev[i],
                            ]
                        )

                plt.figure()
                plt.plot(sampled_epochs, losses_train, label="Train Loss")
                plt.plot(sampled_epochs, losses_dev, label="Dev Loss")
                plt.legend()
                plt.title("Loss Evolution")
                plt.savefig(os.path.join(exp_dir, "loss.png"))
                plt.close()

                plt.figure()
                plt.plot(sampled_epochs, ppls_dev, label="Dev PPL", color="red")
                plt.title("Dev PPL Evolution")
                plt.savefig(os.path.join(exp_dir, "ppl.png"))
                plt.close()

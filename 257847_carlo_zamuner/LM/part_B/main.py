from functions import *
from utils import *
from model import *
import torch.optim as optim
from tqdm import tqdm
import copy
from functools import partial
import math
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import csv
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

    batch_size = 64
    lr = 0.01
    wd = 0.01
    n_epochs = 100
    vocab_len = len(lang.word2id)
    hid_size = emb_size = 300
    clip = 5
    emb_dp = 0.5
    fc_dp = 0.5
    alpha = 0.95

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size * 2,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], DEVICE=DEVICE),
    )

    print(
        f"\n=== Running configuration: emb_dp={emb_dp}, fc_dp={fc_dp}, alpha={alpha}, batch_size={batch_size} ===\n"
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

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion_train = torch.nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = torch.nn.CrossEntropyLoss(
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

    print("Inizio Training...\n")
    pbar = tqdm(range(1, n_epochs + 1), desc="Training")
    for epoch in pbar:
        train_loss = train_loop(
            train_loader, optimizer, criterion_train, model, clip=clip
        )
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

        if epoch == 1:
            avg_model = copy.deepcopy(model)
        else:
            for param, avg_param in zip(model.parameters(), avg_model.parameters()):
                avg_param.data = alpha * avg_param.data + (1 - alpha) * param.data

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
    print(f"\nFinal Test PPL: {final_ppl:.2f}")

    print("\nRisultati:")
    print(f" - Best Dev PPL: {best_dev_ppl:.2f}")
    print(f" - Final Test PPL: {final_ppl:.2f}")
    print(f" - Epochs Trained: {sampled_epochs[-1]}")

    # Salvataggio dei risultati in CSV
    os.makedirs("results/analysis", exist_ok=True)
    with open("results/analysis/training_log.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Dev Loss", "Dev PPL"])
        for i in range(len(sampled_epochs)):
            writer.writerow(
                [sampled_epochs[i], losses_train[i], losses_dev[i], ppls_dev[i]]
            )

    # Grafico: Train vs Dev Loss
    plt.figure(figsize=(10, 6))
    plt.plot(sampled_epochs, losses_train, label="Train Loss", color="blue")
    plt.plot(sampled_epochs, losses_dev, label="Dev Loss", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Dev Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/analysis/loss_over_epochs.png")
    plt.close()

    # Grafico: Dev PPL
    plt.figure(figsize=(10, 6))
    plt.plot(sampled_epochs, ppls_dev, label="Dev PPL", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Dev Perplexity over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/analysis/ppl_over_epochs.png")
    plt.close()

    # Istogramma: distribuzione delle Dev Loss
    plt.figure(figsize=(8, 5))
    plt.hist(losses_dev, bins=20, color="orange", edgecolor="black")
    plt.title("Distribution of Dev Loss")
    plt.xlabel("Dev Loss")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/analysis/dev_loss_distribution.png")
    plt.close()

    # Scatter Plot: Loss vs PPL
    plt.figure(figsize=(8, 6))
    plt.scatter(losses_dev, ppls_dev, c="purple")
    plt.xlabel("Dev Loss")
    plt.ylabel("Dev PPL")
    plt.title("Dev Loss vs Dev PPL")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/analysis/loss_vs_ppl.png")
    plt.close()

    print("\nTutti i grafici e log salvati in 'results/analysis/'")

# This file is used to run your functions and print the results
# Please write your functions or classes in the functions.py

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
    # Check if CUDA is available, otherwise use CPU
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Load datasets once (they will be used in all experiments)
    train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")

    # Build vocab and Lang object (computed on training set)
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    # Create datasets
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # DataLoader instantiation (the same for all experiments)
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

    # Parametri fissi per questa grid search
    fixed_lr = 0.001
    fixed_wd = 0.01
    n_epochs = 100
    vocab_len = len(lang.word2id)

    # Stampiamo i parametri comuni (esclusi quelli che variano)
    common_params = [
        ["Learning Rate", fixed_lr],
        ["Weight Decay", fixed_wd],
        ["Epochs", n_epochs],
        ["Device", DEVICE],
    ]
    print("\nCommon Training Parameters for Optimization:")
    print(tabulate(common_params, headers=["Parameter", "Value"], tablefmt="pretty"))

    # Definiamo le griglie per i nuovi iperparametri:
    optim_hid_sizes = [200, 300, 400]
    optim_emb_sizes = [200, 300, 400]
    optim_clips = [3, 5, 7]

    overall_optim_results = []  # per salvare tuple: (hid_size, emb_size, clip, best_dev_ppl)

    # Grid search sugli iperparametri del modello
    for hid in optim_hid_sizes:
        for emb in optim_emb_sizes:
            for clip_val in optim_clips:
                print(
                    f"\n{'=' * 40}\nStarting experiment with hid_size = {hid}, emb_size = {emb}, clip = {clip_val}\n{'=' * 40}"
                )
                exp_params = [
                    ["Hidden Size", hid],
                    ["Embedding Size", emb],
                    ["Gradient Clip", clip_val],
                ]
                print(
                    tabulate(
                        exp_params, headers=["Parameter", "Value"], tablefmt="pretty"
                    )
                )

                # Inizializza il modello usando LM_LSTM_Dropout (che include i due dropout)
                model = LM_LSTM_Dropout(
                    emb, hid, vocab_len, pad_index=lang.word2id["<pad>"]
                ).to(DEVICE)
                model.apply(init_weights)

                # Inizializza l'ottimizzatore (AdamW) con i parametri fissi fixed_lr e fixed_wd
                optimizer = optim.AdamW(
                    model.parameters(), lr=fixed_lr, weight_decay=fixed_wd
                )
                criterion_train = nn.CrossEntropyLoss(
                    ignore_index=lang.word2id["<pad>"]
                )
                criterion_eval = nn.CrossEntropyLoss(
                    ignore_index=lang.word2id["<pad>"], reduction="sum"
                )

                # Inizializza metriche per questo esperimento
                patience = 3
                sampled_epochs = []
                losses_train = []
                losses_dev = []
                ppls_dev = []
                best_dev_ppl = math.inf
                best_model = None

                # Training loop con tqdm
                pbar = tqdm(range(1, n_epochs + 1), desc="Training")
                for epoch in pbar:
                    train_loss = train_loop(
                        train_loader, optimizer, criterion_train, model, clip=clip_val
                    )
                    ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

                    pbar.set_description(f"PPL: {ppl_dev:.2f}")

                    sampled_epochs.append(epoch)
                    losses_train.append(train_loss)
                    losses_dev.append(loss_dev)
                    ppls_dev.append(ppl_dev)

                    if ppl_dev < best_dev_ppl:
                        best_dev_ppl = ppl_dev
                        best_model = copy.deepcopy(model).to("cpu")
                        patience = 3
                    else:
                        patience -= 1
                        if patience <= 0:
                            print("Early stopping triggered.")
                            break

                best_model.to(DEVICE)

                # Valutazione finale sul test set per questa combinazione
                final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
                print(
                    f"\nFinal Test PPL for hid={hid}, emb={emb}, clip={clip_val}: {final_ppl}"
                )

                overall_optim_results.append((hid, emb, clip_val, best_dev_ppl))

                # Definisci il percorso per salvare CSV e grafici per questo esperimento
                config_dir = f"results/optimizing/hid{hid}_emb{emb}_clip{clip_val}"
                os.makedirs(config_dir, exist_ok=True)

                # Salva i dati in un file CSV
                csv_filename = os.path.join(config_dir, "training_summary.csv")
                with open(csv_filename, mode="w", newline="") as csv_file:
                    writer = csv.writer(csv_file)
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
                print(f"CSV file saved as {csv_filename}")

                # Genera e salva il grafico dell'evoluzione della loss (Train and Dev Loss)
                plt.figure(figsize=(10, 6))
                plt.plot(
                    sampled_epochs, losses_train, label="Train Loss", color="tab:blue"
                )
                plt.plot(
                    sampled_epochs, losses_dev, label="Dev Loss", color="tab:green"
                )
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Evolution of Loss (hid={hid}, emb={emb}, clip={clip_val})")
                plt.legend()
                plt.grid(True)
                loss_plot_filename = os.path.join(config_dir, "loss_evolution.png")
                plt.savefig(loss_plot_filename)
                print(f"Loss evolution plot saved as {loss_plot_filename}")
                plt.close()

                # Genera e salva il grafico dell'evoluzione del Dev PPL
                plt.figure(figsize=(10, 6))
                plt.plot(sampled_epochs, ppls_dev, label="Dev PPL", color="tab:red")
                plt.xlabel("Epoch")
                plt.ylabel("Dev PPL")
                plt.title(
                    f"Evolution of Dev PPL (hid={hid}, emb={emb}, clip={clip_val})"
                )
                plt.legend()
                plt.grid(True)
                ppl_plot_filename = os.path.join(config_dir, "ppl_evolution.png")
                plt.savefig(ppl_plot_filename)
                print(f"Dev PPL evolution plot saved as {ppl_plot_filename}")
                plt.close()

    # Salva i risultati complessivi della grid search di ottimizzazione in un CSV
    optim_csv_filename = "results/optimizing/optimization_results.csv"
    os.makedirs("results/optimizing", exist_ok=True)
    with open(optim_csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Hidden Size", "Embedding Size", "Clip", "Best Dev PPL"])
        for hid_val, emb_val, clip_val, best_ppl_val in overall_optim_results:
            writer.writerow([hid_val, emb_val, clip_val, best_ppl_val])
    print(f"\nOptimization results saved as {optim_csv_filename}")

    # (Opzionale) Genera una heatmap dei risultati (ad esempio, fissando clip e visualizzando hid vs emb)
    # Per semplificare, possiamo generare una heatmap per un clip fisso, ad esempio clip=5.
    fixed_clip = 5
    heatmap = np.zeros((len(optim_hid_sizes), len(optim_emb_sizes)))
    for i, hid in enumerate(optim_hid_sizes):
        for j, emb in enumerate(optim_emb_sizes):
            for hid_val, emb_val, clip_val, best_ppl_val in overall_optim_results:
                if clip_val == fixed_clip and hid_val == hid and emb_val == emb:
                    heatmap[i, j] = best_ppl_val

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="viridis", aspect="auto", origin="lower")
    plt.colorbar(label="Best Dev PPL")
    plt.xticks(ticks=np.arange(len(optim_emb_sizes)), labels=optim_emb_sizes)
    plt.yticks(ticks=np.arange(len(optim_hid_sizes)), labels=optim_hid_sizes)
    plt.xlabel("Embedding Size")
    plt.ylabel("Hidden Size")
    plt.title(f"Optimization Heatmap (Clip = {fixed_clip})")
    grid_plot_filename = "results/optimizing/heatmap_clip5.png"
    plt.savefig(grid_plot_filename)
    plt.show()
    print(f"Optimization heatmap saved as {grid_plot_filename}")

    # Stampa un riepilogo finale dei risultati di ottimizzazione
    print(
        "\nOverall Optimization Results (Hidden Size, Embedding Size, Clip, Best Dev PPL):"
    )
    for hid_val, emb_val, clip_val, best_ppl_val in overall_optim_results:
        print(
            f"hid: {hid_val}, emb: {emb_val}, clip: {clip_val} -> Best Dev PPL: {best_ppl_val}"
        )

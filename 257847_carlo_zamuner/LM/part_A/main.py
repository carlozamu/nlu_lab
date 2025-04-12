from functions import *
from utils import *
from model import *
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
from tabulate import tabulate
from functools import partial
import math
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Impostazione del device
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Caricamento dei dati e creazione del vocabolario
    train_raw = read_file("./dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("./dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("./dataset/PennTreeBank/ptb.test.txt")

    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    # Creazione dei dataset
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Parametri fissi (configurazione ideale individuata)
    wd = 0.01
    n_epochs = 100
    vocab_len = len(lang.word2id)
    lr = 0.001
    batch_size = 16
    hid = 300
    emb = 200
    clip_val = 5

    # Stampa della configurazione attuale
    exp_params = [
        ["Learning Rate", lr],
        ["Batch Size (train)", batch_size],
        ["Hidden Size", hid],
        ["Embedding Size", emb],
        ["Gradient Clip", clip_val],
    ]
    print("\nConfigurazione dell'Esperimento:")
    print(tabulate(exp_params, headers=["Parametro", "Valore"], tablefmt="pretty"))

    # Creazione dei DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(
            collate_fn,
            pad_token=lang.word2id["<pad>"],
            DEVICE=DEVICE,
        ),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size * 2,
        collate_fn=partial(
            collate_fn,
            pad_token=lang.word2id["<pad>"],
            DEVICE=DEVICE,
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        collate_fn=partial(
            collate_fn,
            pad_token=lang.word2id["<pad>"],
            DEVICE=DEVICE,
        ),
    )

    # Inizializzazione del modello
    model = LM_LSTM_Dropout(emb, hid, vocab_len, pad_index=lang.word2id["<pad>"]).to(
        DEVICE
    )
    model.apply(init_weights)

    # Definizione dell'ottimizzatore e dei criteri di perdita
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    # Variabili per il monitoraggio
    patience = 3
    sampled_epochs = []
    losses_train = []
    losses_dev = []
    ppls_dev = []
    best_dev_ppl = math.inf
    best_model = None

    print("\nInizio Training...")
    pbar = tqdm(range(1, n_epochs + 1), desc="Training")
    for epoch in pbar:
        train_loss = train_loop(
            train_loader, optimizer, criterion_train, model, clip=clip_val
        )
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

        pbar.set_description(f"Epoch {epoch} - Dev PPL: {ppl_dev:.2f}")

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

    # Valutazione sul test set del modello migliore
    best_model.to(DEVICE)
    final_test_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print(f"\nFinal Test PPL: {final_test_ppl:.2f}")
    print("Esperimento completato.\n")

    # Stampa riepilogativa dei risultati
    print("Risultati:")
    print(f" - Best Dev PPL: {best_dev_ppl:.2f}")
    print(f" - Final Test PPL: {final_test_ppl:.2f}")
    print(f" - Epochs Trained: {sampled_epochs[-1]}")

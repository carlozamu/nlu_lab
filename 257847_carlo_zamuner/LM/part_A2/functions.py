import torch
import torch.nn as nn
import math


# Funzione originale train_loop
def train_loop(data, optimizer, criterion, model, clip=5):  # Accetta 'clip'
    model.train()
    loss_array = []
    number_of_tokens = []
    for sample in data:  # Qui 'data' è il data_loader
        optimizer.zero_grad()
        output = model(sample["source"])
        # SENZA RESHAPE
        loss = criterion(output, sample["target"])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # Usa 'clip'
        optimizer.step()
    if sum(number_of_tokens) == 0:
        return 0.0
    return sum(loss_array) / sum(number_of_tokens)


# Funzione originale eval_loop
def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():
        for sample in data:  # Qui 'data' è il data_loader
            output = model(sample["source"])
            # SENZA RESHAPE
            loss = eval_criterion(output, sample["target"])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
    if sum(number_of_tokens) == 0:
        return float("inf"), float("inf")
    avg_loss = sum(loss_array) / sum(number_of_tokens)
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")
    return ppl, avg_loss


# Funzione originale init_weights
def init_weights(mat):  # Accetta i moduli passati da apply
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(
                            param[idx * mul : (idx + 1) * mul]
                        )
                elif "weight_hh" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif "bias" in name:
                    param.data.fill_(0)
        elif type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

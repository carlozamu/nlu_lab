import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np


# RNN implementation 
# class LM_RNN(nn.Module):
#     def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
#                 emb_dropout=0.1, n_layers=1):
#         super(LM_RNN, self).__init__()
#         self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
#         self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
#         self.pad_token = pad_index
#         self.output = nn.Linear(hidden_size, output_size)
#     def forward(self, input_sequence):
#         emb = self.embedding(input_sequence)
#         rnn_out, _  = self.rnn(emb)
#         output = self.output(rnn_out).permute(0,2,1)
#         return output

class LM_LSTM_Base(nn.Module):
    """
    Modello Language Model con LSTM.
    Questa versione implementa solo il Punto 1 della Part 1.A:
    - Sostituisce RNN con LSTM.
    - NON include i dropout aggiuntivi del Punto 2.
    """
    def __init__(
        self,
        emb_size,
        hidden_size,
        output_size,
        pad_index=0,
        n_layers=1, # Parametro n_layers mantenuto
    ):
        super(LM_LSTM_Base, self).__init__()
        # Embedding layer (senza dropout successivo per ora)
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        # LSTM layer: batch_first=True per facilitare la gestione dei batch
        # Sostituisce nn.RNN del baseline
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)

        # Linear layer per proiettare lo stato nascosto nello spazio del vocabolario
        # (Senza dropout precedente per ora)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        # Passa l'input nell'embedding
        emb = self.embedding(input_sequence)
        # NESSUN dropout sull'embedding qui (modifica del Punto 2)

        # Passaggio attraverso l'LSTM
        lstm_out, _ = self.lstm(emb)

        # NESSUN dropout sull'output LSTM qui (modifica del Punto 2)
        out = lstm_out # Passa direttamente l'output LSTM al layer lineare

        # Proietta nello spazio del vocabolario e permuta le dimensioni per la loss
        output = self.output(out).permute(0, 2, 1)
        return output

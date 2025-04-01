import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from functions import VariationalDropout


class LM_LSTM_Dropout_Tied(nn.Module):
    def __init__(
        self,
        emb_size,
        hidden_size,
        vocab_size,
        pad_index,
        emb_dropout=0.5,
        fc_dropout=0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_index)
        # Utilizza la classe VariationalDropout:
        self.input_dropout = VariationalDropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.output_dropout = VariationalDropout(fc_dropout)

        if hidden_size != emb_size:
            self.hid2emb = nn.Linear(hidden_size, emb_size)
            self.decoder = nn.Linear(emb_size, vocab_size)
        else:
            self.hid2emb = None
            self.decoder = nn.Linear(hidden_size, vocab_size)

        # Weight tying: i pesi del decoder sono condivisi con quelli dell'embedding
        self.decoder.weight = self.embedding.weight

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.input_dropout(embedded)
        # Aggiungi questa riga:
        self.lstm.flatten_parameters()
        out, _ = self.lstm(embedded)
        out = self.output_dropout(out)
        if self.hid2emb is not None:
            out = self.hid2emb(out)
        output = self.decoder(out).permute(0, 2, 1)
        return output

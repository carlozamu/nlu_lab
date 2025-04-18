import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np


# Helper Class: Locked Dropout (Variational Dropout per attivazioni)
# Variational Dropout di Gal & Ghahramani (2016) / Sez 4.2
class LockedDropout(nn.Module):
    """
    Applica lo stesso dropout mask a tutti i timestep.
    Utile per applicare dropout a sequenze RNN senza distruggere l'informazione temporale.
    """

    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        if not self.training or self.dropout == 0.0:
            return x

        # Maschera con shape [batch_size, 1, feature_dim]
        # Viene applicata tramite broadcasting a tutti i 'seq_len' timesteps.
        mask_shape = (x.size(0), 1, x.size(2))
        # Crea maschera (1 dove si tiene, 0 dove si droppa)
        keep_prob = 1 - self.dropout
        mask = torch.bernoulli(x.data.new(*mask_shape).fill_(keep_prob))

        # Applica maschera e scala (inverted dropout)
        # Scaling factor = 1 / keep_prob
        mask = mask / keep_prob
        masked_x = x * mask

        return masked_x


# Modello LSTM Base (A1 / B1 Start)
class LM_LSTM_Base(nn.Module):
    # ... (codice invariato) ...
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_Base, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        out = lstm_out
        output = self.output(out).permute(0, 2, 1)
        return output


# Modello LSTM con Weight Tying (Step B1)
class LM_LSTM_WeightTying(nn.Module):
    # ... (codice invariato) ...
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_WeightTying, self).__init__()
        assert emb_size == hidden_size, (
            "Weight Tying requires embedding size to be equal to hidden size."
        )
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output_logits = self.output(lstm_out)
        output = output_logits.permute(0, 2, 1)
        return output


# NUOVO: Modello LSTM con Variational Dropout e Weight Tying (Step B2)
class LM_LSTM_VarDropWT(nn.Module):
    """
    Modello Language Model con LSTM, Weight Tying e Variational (Locked) Dropout.
    Implementa il secondo passo della Parte B.
    Applica Locked Dropout a embedding output e LSTM output.
    Richiede emb_size == hidden_size.
    """

    def __init__(
        self,
        emb_size,
        hidden_size,
        output_size,
        pad_index=0,
        n_layers=1,
        dropoute=0.1,
        dropouto=0.4,
    ):  # Esempio tassi dropout
        super(LM_LSTM_VarDropWT, self).__init__()

        assert emb_size == hidden_size, (
            "Weight Tying requires embedding size == hidden size."
        )

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Layers
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            emb_size, hidden_size, n_layers, batch_first=True, dropout=0
        )  # Disabilita dropout standard LSTM
        self.output = nn.Linear(hidden_size, output_size)

        # Variational (Locked) Dropout Layers
        self.lockdrop_e = LockedDropout(dropoute)  # Per embedding output
        self.lockdrop_o = LockedDropout(dropouto)  # Per LSTM output

        # Weight Tying
        self.output.weight = self.embedding.weight

        # Conserva tassi dropout per riferimento
        self.dropoute_rate = dropoute
        self.dropouto_rate = dropouto

    def forward(self, input_sequence, hidden=None):  # Aggiunto hidden opzionale
        # Embedding
        emb = self.embedding(input_sequence)
        # Applica Locked Dropout all'embedding
        emb_dropped = self.lockdrop_e(emb)

        # LSTM
        # L'input all'LSTM è l'embedding con dropout applicato
        lstm_out, hidden_out = self.lstm(emb_dropped, hidden)
        # lstm_out shape: [batch_size, seq_len, hidden_size]

        # Applica Locked Dropout all'output dell'LSTM
        lstm_out_dropped = self.lockdrop_o(lstm_out)

        # Output Layer (con pesi condivisi)
        output_logits = self.output(lstm_out_dropped)
        # output_logits shape: [batch_size, seq_len, output_size]

        # Permuta per CrossEntropyLoss
        output = output_logits.permute(0, 2, 1)

        # Restituisce anche hidden state per usi futuri (es. generazione)
        return output, hidden_out

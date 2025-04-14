import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np


# Baseline LSTM (Step 1 - senza dropout aggiuntivi)
class LM_LSTM_Base(nn.Module):
    """Modello Language Model con LSTM base (Step 1)."""

    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_Base, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        # Nessun dropout applicato qui
        out = lstm_out
        output = self.output(out).permute(0, 2, 1)
        return output


# Modello LSTM con Dropout (Step 2: Punto 1 + Punto 2)
class LM_LSTM_With_Dropout(nn.Module):
    """
    Modello Language Model con LSTM e Dropout aggiuntivi.
    Implementa i Punti 1 e 2 della Part 1.A:
    - Usa LSTM.
    - Aggiunge dropout dopo embedding e prima del layer lineare finale.
    """

    def __init__(
        self,
        emb_size,
        hidden_size,
        output_size,
        pad_index=0,
        emb_dropout=0.1,  # Dropout rate per embedding
        fc_dropout=0.1,  # Dropout rate pre-fully connected
        n_layers=1,
    ):
        super(LM_LSTM_With_Dropout, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # --- Dropout dopo embedding ---
        self.dropout_emb = nn.Dropout(emb_dropout)

        # LSTM layer
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)

        # --- Dropout prima del layer lineare finale ---
        self.dropout_fc = nn.Dropout(fc_dropout)

        # Linear layer
        self.output = nn.Linear(hidden_size, output_size)

        # Conserviamo i valori di dropout per riferimento (es. stampa configurazione)
        self.emb_dropout_rate = emb_dropout
        self.fc_dropout_rate = fc_dropout

    def forward(self, input_sequence):
        # Passa l'input nell'embedding
        emb = self.embedding(input_sequence)
        # --- Applica dropout embedding ---
        emb_dropped = self.dropout_emb(emb)

        # Passaggio attraverso l'LSTM (usa embedding con dropout)
        lstm_out, _ = self.lstm(emb_dropped)

        # --- Applica dropout prima del layer lineare ---
        out_dropped = self.dropout_fc(lstm_out)

        # Proietta nello spazio del vocabolario (usa output LSTM con dropout)
        output = self.output(out_dropped).permute(0, 2, 1)
        return output

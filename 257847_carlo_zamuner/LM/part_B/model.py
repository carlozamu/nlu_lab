import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np


# Baseline LSTM (Step A1 / B1 Start)
class LM_LSTM_Base(nn.Module):
    """Modello Language Model con LSTM base (Step A1 / B1 Start)."""

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


# Modello LSTM con Dropout (Step A2/A3) - Riferimento
class LM_LSTM_With_Dropout(nn.Module):
    """Modello Language Model con LSTM e Dropout aggiuntivi (Step A2/A3)."""

    # ... (codice invariato rispetto a prima) ...
    def __init__(
        self,
        emb_size,
        hidden_size,
        output_size,
        pad_index=0,
        emb_dropout=0.1,
        fc_dropout=0.1,
        n_layers=1,
    ):
        super(LM_LSTM_With_Dropout, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.dropout_emb = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.dropout_fc = nn.Dropout(fc_dropout)
        self.output = nn.Linear(hidden_size, output_size)
        self.emb_dropout_rate = emb_dropout
        self.fc_dropout_rate = fc_dropout

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb_dropped = self.dropout_emb(emb)
        lstm_out, _ = self.lstm(emb_dropped)
        out_dropped = self.dropout_fc(lstm_out)
        output = self.output(out_dropped).permute(0, 2, 1)
        return output


# NUOVO: Modello LSTM con Weight Tying (Step B1)
class LM_LSTM_WeightTying(nn.Module):
    """
    Modello Language Model con LSTM e Weight Tying.
    Implementa il primo passo della Parte B: LSTM base + Weight Tying.
    Richiede emb_size == hidden_size.
    """

    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_WeightTying, self).__init__()

        # !! Vincolo per Weight Tying !!
        assert emb_size == hidden_size, (
            "Weight Tying requires embedding size to be equal to hidden size."
        )

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Layer di Embedding
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        # Layer LSTM
        # Nota: L'input size dell'LSTM è emb_size, l'hidden size è hidden_size
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)

        # Layer Lineare di Output
        # L'input è hidden_size, l'output è output_size
        self.output = nn.Linear(hidden_size, output_size)

        # --- Implementazione del Weight Tying ---
        # Condividi i pesi tra embedding e output lineare
        self.output.weight = self.embedding.weight
        # -----------------------------------------

    def forward(self, input_sequence):
        # Passaggio nell'embedding
        emb = self.embedding(input_sequence)
        # Nessun dropout qui (è il modello base + tying)

        # Passaggio nell'LSTM
        lstm_out, _ = self.lstm(emb)
        # lstm_out ha shape [batch_size, seq_len, hidden_size]

        # Passaggio nel layer lineare di output (con pesi condivisi)
        # Nessun dropout qui
        output_logits = self.output(lstm_out)
        # output_logits ha shape [batch_size, seq_len, output_size]

        # Permuta per CrossEntropyLoss ([batch_size, output_size, seq_len])
        output = output_logits.permute(0, 2, 1)
        return output

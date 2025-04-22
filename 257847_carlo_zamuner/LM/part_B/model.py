import torch
import torch.nn as nn


# Helper Class: Locked Dropout (Variational Dropout)
class LockedDropout(nn.Module):
    """Applica la stessa maschera di dropout a tutti i timesteps."""

    def __init__(self, dropout=0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):  # x shape: [batch, seq_len, features]
        if not self.training or self.dropout == 0.0:
            return x
        mask_shape = (x.size(0), 1, x.size(2))
        keep_prob = 1.0 - self.dropout
        mask = torch.bernoulli(
            torch.full(mask_shape, keep_prob, device=x.device, dtype=x.dtype)
        )
        mask = mask / keep_prob
        return x * mask


# Modello LSTM con Weight Tying e Variational Dropout (Step B3 / FineTune)
class LM_LSTM_VarDropWT(nn.Module):
    """LSTM con Weight Tying e Variational Dropout."""

    def __init__(
        self,
        emb_size,
        hidden_size,
        output_size,
        pad_index=0,
        n_layers=1,
        dropoute=0.1,
        dropouto=0.4,
    ):
        super(LM_LSTM_VarDropWT, self).__init__()
        assert emb_size == hidden_size, "WT richiede emb_size == hidden_size."
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            emb_size, hidden_size, n_layers, batch_first=True, dropout=0
        )
        self.output = nn.Linear(hidden_size, output_size)
        self.lockdrop_e = LockedDropout(dropoute)
        self.lockdrop_o = LockedDropout(dropouto)
        self.output.weight = self.embedding.weight  # Weight Tying

    def forward(self, input_sequence, hidden=None):
        emb = self.embedding(input_sequence)
        emb_dropped = self.lockdrop_e(emb)
        lstm_out, hidden_out = self.lstm(emb_dropped, hidden)
        lstm_out_dropped = self.lockdrop_o(lstm_out)
        output_logits = self.output(lstm_out_dropped)
        output = output_logits.permute(0, 2, 1)
        return output, hidden_out  # Restituisce output e hidden


# --- Modelli per altri step (mantenerli se necessario) ---
# Modello LSTM Base (Step A1)
class LM_LSTM_Base(nn.Module):
    """LSTM base senza dropout aggiuntivo (per Step A1)."""

    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_Base, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output


# Modello LSTM con Dropout standard (Step A2/A3)
class LM_LSTM_With_Dropout(nn.Module):
    """LSTM con Dropout standard."""

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

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb_dropped = self.dropout_emb(emb)
        lstm_out, _ = self.lstm(emb_dropped)
        out_dropped = self.dropout_fc(lstm_out)
        output_logits = self.output(out_dropped)
        output = output_logits.permute(0, 2, 1)
        return output  # Restituisce solo output


# Modello LSTM con Weight Tying (Step B1)
class LM_LSTM_WeightTying(nn.Module):
    """LSTM con Weight Tying (per Step B1)."""

    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, n_layers=1):
        super(LM_LSTM_WeightTying, self).__init__()
        assert emb_size == hidden_size, "WT richiede emb_size == hidden_size."
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

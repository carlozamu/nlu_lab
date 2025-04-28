import torch
import torch.nn as nn

# Modello LSTM con Dropout standard (per Step A2 e A3)
class LM_LSTM_With_Dropout(nn.Module):
    """LSTM con Dropout standard dopo embedding e prima dell'output lineare."""
    def __init__(
        self, emb_size, hidden_size, output_size, pad_index=0,
        emb_dropout=0.1, fc_dropout=0.1, n_layers=1
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
        return output

# Modello LM_LSTM_Base (Step A1) - Mantenere se model.py è unico
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

# Mantenere anche le classi per Parte B se model.py è unico
# class LockedDropout(...): ...
# class LM_LSTM_VarDropWT(...): ...
# class LM_LSTM_WeightTying(...): ...
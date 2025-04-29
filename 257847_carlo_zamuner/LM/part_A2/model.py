import torch
import torch.nn as nn


# Modello LSTM con Dropout standard (per Step A2 e A3)
class LM_LSTM_With_Dropout(nn.Module):
    """LSTM con Dropout standard dopo embedding e prima del layer lineare."""

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
        output = output_logits.permute(0, 2, 1)  # Output 3D [B, V, S]
        return output


# Mantenere anche le altre classi se model.py è unico
class LM_LSTM_Base(nn.Module):
    # ... definizione ...
    pass


# ... (altre classi modello originali) ...

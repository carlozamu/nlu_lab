import torch
import torch.nn as nn

# Mantenuto solo il modello necessario per A1

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
        lstm_out, _ = self.lstm(emb) # lstm_out shape: [B, S, H]
        output_logits = self.output(lstm_out) # output_logits shape: [B, S, V]
        output = output_logits.permute(0, 2, 1) # output shape: [B, V, S]
        return output # Restituisce solo output
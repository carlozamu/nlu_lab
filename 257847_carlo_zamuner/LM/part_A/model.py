import torch
import torch.nn as nn
# Rimosso: optim, F, math, numpy

# Mantenuto solo il modello per Step A3
# Se servono gli altri per eseguire Step A1/A2, vanno mantenuti.


# Modello LSTM con Dropout (usato per Step A2 e A3)
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
        # Rimosse variabili _rate se non usate
        # self.emb_dropout_rate = emb_dropout
        # self.fc_dropout_rate = fc_dropout

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb_dropped = self.dropout_emb(emb)
        lstm_out, _ = self.lstm(emb_dropped)
        out_dropped = self.dropout_fc(lstm_out)
        output_logits = self.output(out_dropped)
        # Permuta per CrossEntropyLoss
        output = output_logits.permute(0, 2, 1)
        return output  # Restituisce solo output (corretto per questo modello)


# Se necessario, mantenere anche LM_LSTM_Base:
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


# Se necessario, mantenere anche LM_LSTM_VarDropWT e LockedDropout per Parte B:
# (Codice omesso qui per focalizzarci su Parte A, ma andrebbe incluso se model.py è unico)

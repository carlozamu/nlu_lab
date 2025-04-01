import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from functions import variational_dropout

# RNN Elman version
# We are not going to use this since for efficiently purposes it's better to use the RNN layer provided by pytorch

# RNN cell implementation (not used in final code)
# class RNN_cell(nn.Module):
#     def __init__(self,  hidden_size, input_size, output_size, vocab_size, dropout=0.1):
#         super(RNN_cell, self).__init__()

#         self.W = nn.Linear(input_size, hidden_size, bias=False)
#         self.U = nn.Linear(hidden_size, hidden_size)
#         self.V = nn.Linear(hidden_size, vocab_size)
#         self.vocab_size = vocab_size
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, prev_hidden, word):
#         input_emb = self.W(word)
#         prev_hidden_rep = self.U(prev_hidden)
#         # ht = σ(Wx + Uht-1 + b)
#         hidden_state = self.sigmoid(input_emb + prev_hidden_rep)
#         # yt = σ(Vht + b)
#         output = self.output(hidden_state)
#         return hidden_state, output


# RNN implementation (not used in final code)
# class LM_RNN(nn.Module):
#     def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
#                 emb_dropout=0.1, n_layers=1):
#         super(LM_RNN, self).__init__()
#         # Token ids to vectors, we will better see this in the next lab
#         self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
#         # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
#         self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
#         self.pad_token = pad_index
#         # Linear layer to project the hidden layer to our output space
#         self.output = nn.Linear(hidden_size, output_size)

#     def forward(self, input_sequence):
#         emb = self.embedding(input_sequence)
#         rnn_out, _  = self.rnn(emb)
#         output = self.output(rnn_out).permute(0,2,1)
#         return output


# class LM_LSTM_Dropout(nn.Module):
#     def __init__(
#         self,
#         emb_size,
#         hidden_size,
#         output_size,
#         pad_index=0,
#         emb_dropout=0.1,
#         fc_dropout=0.1,
#         n_layers=1,
#     ):
#         super(LM_LSTM_Dropout, self).__init__()
#         # Embedding layer con dropout subito dopo
#         self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
#         self.dropout_emb = nn.Dropout(emb_dropout)

#         # LSTM layer: batch_first=True per facilitare la gestione dei batch
#         self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)

#         # Dropout applicato prima dell'ultimo layer lineare
#         self.dropout_fc = nn.Dropout(fc_dropout)

#         # Linear layer per proiettare lo stato nascosto nello spazio del vocabolario
#         self.output = nn.Linear(hidden_size, output_size)

#     def forward(self, input_sequence):
#         # Passa l'input nell'embedding e applica dropout
#         emb = self.embedding(input_sequence)
#         emb = self.dropout_emb(emb)

#         # Passaggio attraverso l'LSTM
#         lstm_out, _ = self.lstm(emb)

#         # Applica dropout prima del layer lineare
#         out = self.dropout_fc(lstm_out)

#         # Proietta nello spazio del vocabolario e permuta le dimensioni per la loss
#         output = self.output(out).permute(0, 2, 1)
#         return output


class LM_LSTM_Dropout_Tied(nn.Module):
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
        super(LM_LSTM_Dropout_Tied, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.dropout_emb = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.dropout_fc = nn.Dropout(fc_dropout)
        self.output = nn.Linear(hidden_size, output_size)
        # Weight tying: condividi i pesi tra embedding e output
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.dropout_emb(emb)
        # Assicurati che i pesi siano contigui per migliorare le prestazioni
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(emb)
        out = self.dropout_fc(lstm_out)
        output = self.output(out).permute(0, 2, 1)
        return output

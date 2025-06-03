import torch
import torch.nn as nn

class LockedDropout(nn.Module):
 
    def __init__(self, dropout_rate: float = 0.5):
      
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
     
        if not self.training or self.dropout_rate == 0.0:
            return x

       
        mask_shape = (x.size(0), 1, x.size(2))
        keep_probability = 1.0 - self.dropout_rate

      
        mask = torch.bernoulli(
            torch.full(mask_shape, keep_probability, device=x.device, dtype=x.dtype)
        )

        mask = mask / keep_probability
        return x * mask


class LM_LSTM_Base(nn.Module):
    
    def __init__(
        self, emb_size: int, hidden_size: int, output_size: int,
        pad_index: int = 0, n_layers: int = 1
    ):
        super(LM_LSTM_Base, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output_logits = self.output(lstm_out)
        output = output_logits.permute(0, 2, 1)
        return output

class LM_LSTM_With_Dropout(nn.Module):
    
    def __init__(
        self, emb_size: int, hidden_size: int, output_size: int,
        pad_index: int = 0, emb_dropout: float = 0.1, fc_dropout: float = 0.1,
        n_layers: int = 1
    ):
        super(LM_LSTM_With_Dropout, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.dropout_emb = nn.Dropout(emb_dropout) # Dropout after embedding
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.dropout_fc = nn.Dropout(fc_dropout)   # Dropout before final linear layer
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_sequence)
        emb_dropped = self.dropout_emb(emb)
        lstm_out, _ = self.lstm(emb_dropped)
        out_dropped = self.dropout_fc(lstm_out)
        output_logits = self.output(out_dropped)
        output = output_logits.permute(0, 2, 1)
        return output


class LM_LSTM_WeightTying(nn.Module):
    
    def __init__(
        self, emb_size: int, hidden_size: int, output_size: int,
        pad_index: int = 0, n_layers: int = 1
    ):
        super(LM_LSTM_WeightTying, self).__init__()
        assert emb_size == hidden_size
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight 

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output_logits = self.output(lstm_out)
        output = output_logits.permute(0, 2, 1)
        return output


class LM_LSTM_VarDropWT(nn.Module):

    def __init__(
        self,
        emb_size: int,      
        hidden_size: int,   
        output_size: int,   
        pad_index: int = 0, 
        n_layers: int = 1, 
        dropoute: float = 0.1, 
        dropouto: float = 0.4 
    ):
        super(LM_LSTM_VarDropWT, self).__init__()
        
        assert emb_size == hidden_size, \
            "For this Weight Tying setup, embedding_size must equal hidden_size."

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        self.lstm = nn.LSTM(
            emb_size, hidden_size, n_layers, batch_first=True, dropout=0
        )
        self.output = nn.Linear(hidden_size, output_size)

        
        self.lockdrop_e = LockedDropout(dropoute) 
        self.lockdrop_o = LockedDropout(dropouto) 

        
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence: torch.Tensor,
                hidden: tuple[torch.Tensor, torch.Tensor] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
       
        emb = self.embedding(input_sequence)            
        emb_dropped = self.lockdrop_e(emb)              

        lstm_out, hidden_out = self.lstm(emb_dropped, hidden)

        lstm_out_dropped = self.lockdrop_o(lstm_out)

        output_logits = self.output(lstm_out_dropped)

        output = output_logits.permute(0, 2, 1)
        return output, hidden_out
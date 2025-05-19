
import torch
import torch.nn as nn

# ==============================================================================
# --- MODEL DEFINITION: LM_LSTM_With_Dropout (for Steps A2 & A3) ---
# ==============================================================================
class LM_LSTM_With_Dropout(nn.Module):
    """
    LSTM Language Model with standard Dropout layers.
    Used for Part 1.A, Steps 2 and 3.
    - Applies Dropout after embeddings.
    - Applies Dropout before the final linear output layer.
    """
    def __init__(
        self,
        emb_size: int,      # Dimensionality of word embeddings
        hidden_size: int,   # Dimensionality of LSTM hidden states
        output_size: int,   # Vocabulary size (number of output classes)
        pad_index: int = 0, # Index of the padding token
        emb_dropout: float = 0.1, # Dropout rate after embedding layer
        fc_dropout: float = 0.1,  # Dropout rate before final linear layer
        n_layers: int = 1        # Number of LSTM layers
    ):
        super(LM_LSTM_With_Dropout, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.dropout_emb = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.dropout_fc = nn.Dropout(fc_dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        Args:
            input_sequence (torch.Tensor): Tensor of input token IDs.
                                           Shape: (batch_size, sequence_length)
        Returns:
            torch.Tensor: Logits over the vocabulary for each token in the sequence.
                          Shape: (batch_size, output_size, sequence_length)
        """
        # input_sequence shape: (batch_size, sequence_length)
        emb = self.embedding(input_sequence)    # (batch_size, sequence_length, emb_size)
        emb_dropped = self.dropout_emb(emb)     # (batch_size, sequence_length, emb_size)

        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        # hidden and cell states are ignored here
        lstm_out, _ = self.lstm(emb_dropped)

        out_dropped = self.dropout_fc(lstm_out) # (batch_size, sequence_length, hidden_size)

        # output_logits shape: (batch_size, sequence_length, output_size)
        output_logits = self.output(out_dropped)

        # Permute for CrossEntropyLoss, which expects (N, C, ...)
        # N=batch_size, C=output_size, ...=sequence_length
        # Output shape: (batch_size, output_size, sequence_length)
        output = output_logits.permute(0, 2, 1)
        return output

# ==============================================================================
# --- MODEL DEFINITION: LM_LSTM_Base (for Step A1) ---
# ==============================================================================
class LM_LSTM_Base(nn.Module):
    """
    Basic LSTM Language Model without additional Dropout layers.
    Used for Part 1.A, Step 1.
    """
    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        output_size: int,
        pad_index: int = 0,
        n_layers: int = 1
    ):
        super(LM_LSTM_Base, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        Args:
            input_sequence (torch.Tensor): Tensor of input token IDs.
                                           Shape: (batch_size, sequence_length)
        Returns:
            torch.Tensor: Logits over the vocabulary.
                          Shape: (batch_size, output_size, sequence_length)
        """
        emb = self.embedding(input_sequence)    # (batch_size, sequence_length, emb_size)
        lstm_out, _ = self.lstm(emb)            # (batch_size, sequence_length, hidden_size)
        output_logits = self.output(lstm_out)   # (batch_size, sequence_length, output_size)
        output = output_logits.permute(0, 2, 1) # (batch_size, output_size, sequence_length)
        return output

# ==============================================================================
# --- MODELS FOR PART B (Keep if this is a unified model file) ---
# ==============================================================================

class LockedDropout(nn.Module):
    """
    Implements Locked Dropout (Variational Dropout for activations).
    Applies the same dropout mask across all timesteps of a sequence.
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, feature_dim)
        if not self.training or self.dropout == 0.0:
            return x
        # Mask shape: (batch_size, 1, feature_dim) for broadcasting
        mask_shape = (x.size(0), 1, x.size(2))
        keep_prob = 1.0 - self.dropout
        # Create mask on the same device and dtype as x
        mask = torch.bernoulli(torch.full(mask_shape, keep_prob, device=x.device, dtype=x.dtype))
        # Apply inverted dropout scaling
        mask = mask / keep_prob
        return x * mask

class LM_LSTM_WeightTying(nn.Module):
    """LSTM Language Model with Weight Tying (for Part 1.B, Step 1)."""
    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        output_size: int,
        pad_index: int = 0,
        n_layers: int = 1
    ):
        super(LM_LSTM_WeightTying, self).__init__()
        # For direct weight tying, embedding size must equal hidden size if LSTM output is directly fed to tied linear layer
        assert emb_size == hidden_size, "Weight Tying in this setup requires embedding_size == hidden_size."
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        # Tie weights: output layer's weights are the embedding layer's weights
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_sequence)   
        lstm_out, _ = self.lstm(emb)           
        output_logits = self.output(lstm_out)  
        output = output_logits.permute(0, 2, 1) 
        return output

class LM_LSTM_VarDropWT(nn.Module):
    """
    LSTM LM with Weight Tying and Variational Dropout (Locked Dropout).
    Used for Part 1.B, Step 3 and Fine-Tuning.
    """
    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        output_size: int,
        pad_index: int = 0,
        n_layers: int = 1,
        dropoute: float = 0.1, # Variational dropout rate for embeddings
        dropouto: float = 0.4  # Variational dropout rate for LSTM outputs
    ):
        super(LM_LSTM_VarDropWT, self).__init__()
        assert emb_size == hidden_size, "Weight Tying requires embedding_size == hidden_size."
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # LSTM's internal dropout is disabled (dropout=0) as VarDrop is applied externally
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True, dropout=0)
        self.output = nn.Linear(hidden_size, output_size)

        # Variational (Locked) Dropout layers
        self.lockdrop_e = LockedDropout(dropoute) # Applied to embeddings
        self.lockdrop_o = LockedDropout(dropouto) # Applied to LSTM output

        # Apply Weight Tying
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence: torch.Tensor, hidden: tuple = None) -> tuple[torch.Tensor, tuple]:
        """
        Args:
            input_sequence (torch.Tensor): (batch_size, sequence_length)
            hidden (tuple, optional): Initial hidden and cell states for LSTM.
        Returns:
            tuple[torch.Tensor, tuple]: Output logits (B, V, S) and final LSTM states.
        """
        emb = self.embedding(input_sequence)          
        emb_dropped = self.lockdrop_e(emb)             
        lstm_out, hidden_out = self.lstm(emb_dropped, hidden) 
        lstm_out_dropped = self.lockdrop_o(lstm_out)    
        output_logits = self.output(lstm_out_dropped)   
        output = output_logits.permute(0, 2, 1)         
        return output, hidden_out
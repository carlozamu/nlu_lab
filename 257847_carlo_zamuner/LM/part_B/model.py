
import torch
import torch.nn as nn

# ==============================================================================
# --- HELPER CLASS: LockedDropout (Variational Dropout) ---
# ==============================================================================
class LockedDropout(nn.Module):
    """
    Implements Locked Dropout.
    Applies the same dropout mask across all timesteps of a sequence.
    This is a form of Variational Dropout for RNN activations.
    """
    def __init__(self, dropout_rate: float = 0.5):
        """
        Args:
            dropout_rate (float): Probability of an element to be zeroed.
        """
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
                              Shape: (batch_size, sequence_length, feature_dim)
        Returns:
            torch.Tensor: Output tensor with dropout applied.
                          Shape: (batch_size, sequence_length, feature_dim)
        """
        if not self.training or self.dropout_rate == 0.0:
            return x

        # Create a mask with shape (batch_size, 1, feature_dim)
        # This mask will be broadcast across the sequence_length dimension.
        mask_shape = (x.size(0), 1, x.size(2))
        keep_probability = 1.0 - self.dropout_rate

        # Generate a Bernoulli mask on the same device and dtype as the input tensor x
        mask = torch.bernoulli(
            torch.full(mask_shape, keep_probability, device=x.device, dtype=x.dtype)
        )

        # Apply inverted dropout scaling: scale by 1 / keep_probability
        mask = mask / keep_probability
        return x * mask

# ==============================================================================
# --- MODEL DEFINITION: LM_LSTM_VarDropWT (for Part B, Step 3 & Fine-Tuning) ---
# ==============================================================================
class LM_LSTM_VarDropWT(nn.Module):
    """
    LSTM Language Model with Weight Tying and Variational Dropout (Locked Dropout).
    """
    def __init__(
        self,
        emb_size: int,      # Dimensionality of word embeddings
        hidden_size: int,   # Dimensionality of LSTM hidden states
        output_size: int,   # Vocabulary size (number of output classes)
        pad_index: int = 0, # Index of the padding token
        n_layers: int = 1,  # Number of LSTM layers
        dropoute: float = 0.1, # Variational dropout rate for embeddings
        dropouto: float = 0.4  # Variational dropout rate for LSTM outputs
    ):
        super(LM_LSTM_VarDropWT, self).__init__()
        # Weight Tying in this configuration typically requires embedding_size == hidden_size
        assert emb_size == hidden_size, \
            "For this Weight Tying setup, embedding_size must equal hidden_size."

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # LSTM's internal dropout is set to 0 as Variational Dropout is applied externally
        self.lstm = nn.LSTM(
            emb_size, hidden_size, n_layers, batch_first=True, dropout=0
        )
        self.output = nn.Linear(hidden_size, output_size)

        # Variational (Locked) Dropout layers
        self.lockdrop_e = LockedDropout(dropoute) # Applied after embedding
        self.lockdrop_o = LockedDropout(dropouto) # Applied after LSTM output

        # Apply Weight Tying: share weights between embedding and output projection
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence: torch.Tensor,
                hidden: tuple[torch.Tensor, torch.Tensor] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Defines the forward pass of the model.

        Args:
            input_sequence (torch.Tensor): Tensor of input token IDs.
                                           Shape: (batch_size, sequence_length)
            hidden (tuple, optional): Initial hidden and cell states for the LSTM.
                                      Defaults to None (zero-initialized states).
        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                - Output logits over the vocabulary.
                  Shape: (batch_size, output_size, sequence_length)
                - Final hidden and cell states of the LSTM.
        """
        # input_sequence shape: (batch_size, sequence_length)
        emb = self.embedding(input_sequence)            # (batch_size, sequence_length, emb_size)
        emb_dropped = self.lockdrop_e(emb)              # (batch_size, sequence_length, emb_size)

        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, hidden_out = self.lstm(emb_dropped, hidden)

        lstm_out_dropped = self.lockdrop_o(lstm_out)    # (batch_size, sequence_length, hidden_size)

        # output_logits shape: (batch_size, sequence_length, output_size)
        output_logits = self.output(lstm_out_dropped)

        # Permute for CrossEntropyLoss, which expects (N, C, ...)
        # Output shape: (batch_size, output_size, sequence_length)
        output = output_logits.permute(0, 2, 1)
        return output, hidden_out

# ==============================================================================
# --- MODEL DEFINITION: LM_LSTM_Base (for Part A, Step 1) ---
# ==============================================================================
class LM_LSTM_Base(nn.Module):
    """Basic LSTM Language Model without additional explicit Dropout layers."""
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

# ==============================================================================
# --- MODEL DEFINITION: LM_LSTM_With_Dropout (for Part A, Steps 2 & 3) ---
# ==============================================================================
class LM_LSTM_With_Dropout(nn.Module):
    """LSTM Language Model with standard Dropout layers."""
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

# ==============================================================================
# --- MODEL DEFINITION: LM_LSTM_WeightTying (for Part B, Step 1) ---
# ==============================================================================
class LM_LSTM_WeightTying(nn.Module):
    """LSTM Language Model with Weight Tying."""
    def __init__(
        self, emb_size: int, hidden_size: int, output_size: int,
        pad_index: int = 0, n_layers: int = 1
    ):
        super(LM_LSTM_WeightTying, self).__init__()
        assert emb_size == hidden_size, \
            "For this Weight Tying setup, embedding_size must equal hidden_size."
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight # Apply Weight Tying

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output_logits = self.output(lstm_out)
        output = output_logits.permute(0, 2, 1)
        return output
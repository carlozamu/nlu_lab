
import torch
import torch.nn as nn
from transformers import BertModel

# ==============================================================================
# --- BERT-based Model for Joint Intent Classification and Slot Filling ---
# ==============================================================================

class JointBERTModel(nn.Module):
    """
    BERT-based model for joint Intent Classification and Slot Filling.
    """
    def __init__(self, num_intent_labels, num_slot_labels, bert_model_name='bert-base-uncased', dropout_rate=0.1):
        """
        Args:
            num_intent_labels (int): Number of intent classes.
            num_slot_labels (int): Number of slot classes.
            bert_model_name (str): Pre-trained BERT model name.
            dropout_rate (float): Dropout rate for task-specific layers.
        """
        super(JointBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.intent_classifier = nn.Linear(bert_hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(bert_hidden_size, num_slot_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Args:
            input_ids (Tensor): Sub-token IDs (batch_size, seq_len).
            attention_mask (Tensor): Attention mask (batch_size, seq_len).
            token_type_ids (Tensor): Token type IDs (batch_size, seq_len).
        Returns:
            intent_logits (Tensor): (batch_size, num_intent_labels).
            slot_logits (Tensor): (batch_size, seq_len, num_slot_labels).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False # Returns tuple: (last_hidden_state, pooler_output, ...)
        )

        last_hidden_state = outputs[0] # (batch_size, seq_len, bert_hidden_size)
        
        # Intent classification using [CLS] token representation
        cls_token_output = last_hidden_state[:, 0, :]
        intent_logits = self.intent_classifier(self.dropout(cls_token_output))

        # Slot filling using all token representations
        slot_logits = self.slot_classifier(self.dropout(last_hidden_state))
        
        return intent_logits, slot_logits

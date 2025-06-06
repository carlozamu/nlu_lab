import torch.nn as nn
from transformers import BertModel


class JointBERTModel(nn.Module):
    """
    Joint BERT model for simultaneous intent classification and slot filling.
    
    Architecture:
    1. BERT encoder: processes input text and creates contextualized representations
    2. Intent classifier: uses [CLS] token to classify the overall intent
    3. Slot classifier: uses all token representations to label each word
    
    This is called "joint" because both tasks share the same BERT encoder,
    allowing them to benefit from each other during training.
    """
    
    def __init__(
        self,
        num_intent_labels,
        num_slot_labels,
        bert_model_name="bert-base-uncased",
        dropout_rate=0.1,
    ):
        super(JointBERTModel, self).__init__()
        
        # Load pre-trained BERT model (frozen or fine-tuned)
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size  # Usually 768 for base models
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Intent classification head: [CLS] token -> intent probabilities
        # Input: BERT hidden size (768) -> Output: number of intent classes
        self.intent_classifier = nn.Linear(bert_hidden_size, num_intent_labels)
        
        # Slot classification head: each token -> slot label probabilities  
        # Input: BERT hidden size (768) -> Output: number of slot labels (BIO tags)
        self.slot_classifier = nn.Linear(bert_hidden_size, num_slot_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Forward pass through the joint model.
        """
        
        # Pass through BERT encoder to get contextualized representations
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,  # Return tuple instead of dict
        )

        # Extract hidden states: [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs[0]

        # INTENT CLASSIFICATION
        # Use [CLS] token (first token) representation for sentence-level classification
        cls_token_output = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        intent_logits = self.intent_classifier(self.dropout(cls_token_output))

        # SLOT CLASSIFICATION  
        # Use all token representations for word-level classification
        slot_logits = self.slot_classifier(self.dropout(last_hidden_state))

        return intent_logits, slot_logits
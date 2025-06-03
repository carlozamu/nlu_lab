import torch.nn as nn
from transformers import BertModel


class JointBERTModel(nn.Module):
    

    def __init__(
        self,
        num_intent_labels,
        num_slot_labels,
        bert_model_name="bert-base-uncased",
        dropout_rate=0.1,
    ):
        
        super(JointBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.intent_classifier = nn.Linear(bert_hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(bert_hidden_size, num_slot_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )

        last_hidden_state = outputs[0]

        # Intent classification using [CLS] token representation
        cls_token_output = last_hidden_state[:, 0, :]
        intent_logits = self.intent_classifier(self.dropout(cls_token_output))

        # Slot filling using all token representations
        slot_logits = self.slot_classifier(self.dropout(last_hidden_state))

        return intent_logits, slot_logits

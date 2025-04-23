# --- START OF FILE model.py ---

import torch
import torch.nn as nn
# Importiamo il modello pre-addestrato di BERT
from transformers import BertModel

# Qui è dove definisci l'architettura del tuo modello usando PyTorch
# MODELLO BASATO SU BERT PER JOINT INTENT & SLOT

class JointBERTModel(nn.Module):
    ''' Modello BERT per Joint Intent Classification e Slot Filling. '''
    def __init__(self, num_intent_labels, num_slot_labels, bert_model_name='bert-base-uncased', dropout_rate=0.1):
        '''
        Inizializza il modello BERT e i layer di output per Intent e Slot.
        Args:
            num_intent_labels (int): Numero di classi di intent.
            num_slot_labels (int): Numero di classi di slot.
            bert_model_name (str): Nome del modello BERT pre-addestrato da caricare.
            dropout_rate (float): Tasso di dropout (usato nei layer di output).
        '''
        super(JointBERTModel, self).__init__()
        # Carica il modello BERT pre-addestrato
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Ottieni la dimensione dello stato nascosto di BERT
        # bert-base-uncased ha hidden_size = 768
        bert_hidden_size = self.bert.config.hidden_size

        # Layer di Dropout (applicato prima dei layer lineari di output)
        self.dropout = nn.Dropout(dropout_rate)

        # Layer di output per Intent Classification
        # Prende l'output del token [CLS]
        self.intent_classifier = nn.Linear(bert_hidden_size, num_intent_labels)

        # Layer di output per Slot Filling
        # Prende l'output di tutti i token (tranne [CLS])
        self.slot_classifier = nn.Linear(bert_hidden_size, num_slot_labels)


    def forward(self, input_ids, attention_mask, token_type_ids):
        '''
        Passaggio forward del modello BERT con layer di output.
        Args:
            input_ids (Tensor): IDs dei sub-token (batch_size, seq_len).
            attention_mask (Tensor): Mask per distinguere token reali da padding (batch_size, seq_len).
            token_type_ids (Tensor): IDs per distinguere segmenti (batch_size, seq_len).
        Returns:
            tuple: (intent_logits, slot_logits)
                   - intent_logits (Tensor): Logits per le classi di intent (batch_size, num_intent_labels).
                   - slot_logits (Tensor): Logits per le classi di slot (batch_size, seq_len, num_slot_labels).
        '''
        # Passa gli input a BERT
        # outputs è una tupla, il primo elemento [0] è l'output dello strato finale
        # last_hidden_state: (batch_size, seq_len, bert_hidden_size)
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=False) # Imposta False per ottenere la tupla come nel notebook originale

        last_hidden_state = outputs[0]

        # --- Intent Classification ---
        # Prendi l'output del token [CLS] (il primo nella sequenza)
        # [CLS] output: (batch_size, bert_hidden_size)
        cls_output = last_hidden_state[:, 0, :]

        # Applica dropout e layer lineare per l'intento
        intent_logits = self.intent_classifier(self.dropout(cls_output))


        # --- Slot Filling ---
        # Prendi gli output per tutti i token *tranne* il [CLS] iniziale
        # Altri token output: (batch_size, seq_len - 1, bert_hidden_size)
        # Nota: Dobbiamo mantenere la dimensione della sequenza per l'output slot
        # Il layer lineare `slot_classifier` applica la trasformazione all'ultima dimensione
        # Quindi `last_hidden_state` completo può essere passato
        # Applica dropout e layer lineare per gli slot a *tutti* gli output della sequenza
        # L'allineamento sub-token/parola e l'ignorare i token non-iniziali
        # e il padding verranno gestiti nella funzione di loss nel train_loop
        # o nella gestione dell'output nell'eval_loop.
        slot_logits = self.slot_classifier(self.dropout(last_hidden_state))
        # slot_logits dimensione: (batch_size, seq_len, num_slot_labels)


        # Nota: Non applichiamo Softmax qui, lo facciamo nella funzione di loss (CrossEntropyLoss)

        return intent_logits, slot_logits

# --- END OF FILE model.py ---
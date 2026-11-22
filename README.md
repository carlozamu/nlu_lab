# NLU Lab

Laboratori ed esame del corso *Natural Language Understanding* (UniTrento).

## Contenuto

| Cartella | Cosa c'è |
|---|---|
| `labs/` | Notebook del corso: corpus e lessico, metodologia sperimentale, n-gram, LM neurali, intent & slot filling, POS e NER |
| `257847_carlo_zamuner/LM/` | Language modeling su Penn Treebank. Part A: LSTM base e con dropout. Part B: regolarizzazioni (locked dropout) |
| `257847_carlo_zamuner/NLU/` | Intent classification e slot filling. Part A: modello LSTM. Part B: Joint BERT |
| `dataset/` | Penn Treebank e dati per i laboratori |

Ogni parte ha `main.py` (entrypoint), `model.py`, `functions.py`, `utils.py`; le relazioni sono in `NLUP1.pdf` e `NLUP2.pdf`.

## Esecuzione

```bash
pip install -r requirements.txt
cd 257847_carlo_zamuner/LM/part_A && python main.py
```

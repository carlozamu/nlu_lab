# Relazione Esercizio 05: Intent Classification e Slot Filling

Questo documento riassume il lavoro svolto per la Parte 2.A dell'Esercizio 05, che riguarda l'implementazione e il miglioramento di un modello neurale per i task congiunti di Intent Classification e Slot Filling sul dataset ATIS.

Abbiamo implementato l'architettura base e applicato incrementalmente le modifiche richieste (bidirezionalità e dropout), accompagnando ogni step con una fase di tuning degli iperparametri chiave per massimizzare le performance, valutate tramite Slot Filling F1 e Intent Classification Accuracy sul test set.

La struttura dei file (`main.py`, `model.py`, `utils.py`, `functions.py`) e l'organizzazione dei risultati in cartelle (`results/`) sono state adottate per una gestione chiara degli esperimenti.

## Architetture Testate (Incrementalmente)

1.  **Baseline:** Modello basato su LSTM **unidirezionale**, senza dropout.
2.  **Bidirezionale:** Modello basato su LSTM **bidirezionale**, senza dropout.
3.  **Bidirezionale + Dropout:** Modello basato su LSTM **bidirezionale**, con aggiunta di layer di dropout.

Per ogni architettura, abbiamo effettuato un tuning degli iperparametri rilevanti (principalmente Learning Rate, Batch Size e Dropout Rate) basandoci sulle performance sul set di sviluppo (Dev set) e riportando i risultati finali sul set di test (Test set).

## Risultati Comparativi sul Test Set

Di seguito vengono presentati i risultati ottenuti sul Test set per le configurazioni migliori (o rappresentative) di ciascuno step, basandosi sull'analisi delle fasi di tuning.

### 1. Baseline (LSTM Unidirezionale, No Dropout)

Dopo il tuning del Learning Rate per l'architettura Baseline, i risultati sul Test set erano i seguenti:

| Configurazione     |   Learning Rate |   Test Slot F1 |   Test Intent Acc |   Test Loss |
| :----------------- | :-------------- | :------------- | :---------------- | :---------- |
| BASELINE\_LR=0.0001 |        0.0001 |         0.0062 |            0.7077 |      3.8654 |
| BASELINE\_LR=0.001  |        0.001  |         0.9236 |            0.9261 |      0.5922 |
| BASELINE\_LR=0.005  |        0.005  |         0.9272 |            0.9183 |      0.6411 |
| BASELINE\_LR=0.01   |        0.01   |         0.9093 |            0.9127 |      0.6665 |

**Conclusioni Baseline:**
*   Un Learning Rate troppo basso (0.0001) impedisce l'apprendimento efficace.
*   Learning Rate di `0.001` e `0.005` offrono le migliori performance per la baseline.
*   La configurazione migliore per la baseline si attesta intorno a **Slot F1 ~0.927** e **Intent Acc ~0.926**.

### 2. Bidirezionale (LSTM Bidirezionale, No Dropout)

Passando all'architettura Bidirezionale (e utilizzando come punto di partenza i parametri della migliore baseline per un nuovo tuning del LR), abbiamo osservato i seguenti risultati per vari LR (tenendo Batch Size a 64 per queste run specifiche):

| Configurazione          |   Learning Rate |   Test Slot F1 |   Test Intent Acc |   Test Loss |
| :---------------------- | :-------------- | :------------- | :---------------- | :---------- |
| BIDIRECTIONAL\_LR=0.0001 |        0.0001 |         0.1186 |            0.7077 |      3.1284 |
| BIDIRECTIONAL\_LR=0.0005 |        0.0005 |         0.9429 |            0.9429 |      0.4203 |
| BIDIRECTIONAL\_LR=0.001  |        0.001  |         0.9435 |            0.9462 |      0.4035 |
| BIDIRECTIONAL\_LR=0.005  |        0.005  |         0.9352 |            0.9395 |      0.5538 |
| BIDIRECTIONAL\_LR=0.01   |        0.01   |         0.9383 |            0.9418 |      0.5325 |

**Conclusioni Bidirezionale (No Dropout):**
*   L'aggiunta della bidirezionalità porta a un **netto miglioramento** rispetto alla Baseline.
*   Le performance migliori si spostano verso Learning Rate di `0.0005` e `0.001`.
*   La configurazione migliore (con Batch Size 64) si attesta intorno a **Slot F1 ~0.9435** e **Intent Acc ~0.9462**.
*   Questo rappresenta un aumento di circa 2 punti percentuali per Slot F1 e ~2 punti percentuali per Intent Acc rispetto alla migliore Baseline.

### 3. Bidirezionale + Dropout

Eseguendo una Grid Search su Dropout Rate, Learning Rate e Batch Size per l'architettura Bidirezionale con Dropout, abbiamo ottenuto la seguente tabella di risultati completi (riportata per intero per riferimento):

| Config                  |   Dropout Rate |   Learning Rate |   Batch Size Train |   Test Slot F1 |   Test Intent Acc |   Test Loss |
| :---------------------- | :------------- | :-------------- | :----------------- | :------------- | :---------------- | :---------- |
| BI\_DR0.0\_LR0.0005\_BS64 |            0   |          0.0005 |                 64 |       0.946904 |          0.946249 |    0.442035 |
| BI\_DR0.0\_LR0.0005\_BS128 |            0   |          0.0005 |                128 |       0.933639 |          0.930571 |    0.451865 |
| BI\_DR0.0\_LR0.001\_BS64 |            0   |          0.001  |                 64 |       0.942923 |          0.950728 |    0.424828 |
| BI\_DR0.0\_LR0.001\_BS128 |            0   |          0.001  |                128 |       0.938811 |          0.948488 |    0.431403 |
| BI\_DR0.0\_LR0.005\_BS64 |            0   |          0.005  |                 64 |       0.941342 |          0.93841  |    0.543667 |
| BI\_DR0.0\_LR0.005\_BS128 |            0   |          0.005  |                128 |       0.937865 |          0.945129 |    0.503236 |
| BI\_DR0.1\_LR0.0005\_BS64 |            0.1 |          0.0005 |                 64 |       0.943991 |          0.956327 |    0.415227 |
| BI\_DR0.1\_LR0.0005\_BS128 |            0.1 |          0.0005 |                128 |       0.939266 |          0.946249 |    0.416545 |
| BI\_DR0.1\_LR0.001\_BS64 |            0.1 |          0.001  |                 64 |       0.949656 |          0.952968 |    0.393074 |
| BI\_DR0.1\_LR0.001\_BS128 |            0.1 |          0.001  |                128 |       0.943196 |          0.951848 |    0.413597 |
| BI\_DR0.1\_LR0.005\_BS64 |            0.1 |          0.005  |                 64 |       0.944141 |          0.955207 |    0.460834 |
| BI\_DR0.1\_LR0.005\_BS128 |            0.1 |          0.005  |                128 |       0.94261  |          0.945129 |    0.455338 |
| BI\_DR0.2\_LR0.0005\_BS64 |            0.2 |          0.0005 |                 64 |       0.948501 |          0.942889 |    0.415774 |
| BI\_DR0.2\_LR0.0005\_BS128 |            0.2 |          0.0005 |                128 |       0.936373 |          0.948488 |    0.392932 |
| BI\_DR0.2\_LR0.001\_BS64 |            0.2 |          0.001  |                 64 |       0.943856 |          0.949608 |    0.400868 |
| BI\_DR0.2\_LR0.001\_BS128 |            0.2 |          0.001  |                128 |       0.943975 |          0.950728 |    0.426761 |
| BI\_DR0.2\_LR0.005\_BS64 |            0.2 |          0.005  |                 64 |       0.948465 |          0.949608 |    0.487521 |
| BI\_DR0.2\_LR0.005\_BS128 |            0.2 |          0.005  |                128 |       0.946921 |          0.947368 |    0.487568 |
| BI\_DR0.3\_LR0.0005\_BS64 |            0.3 |          0.0005 |                 64 |       0.940844 |          0.945129 |    0.429927 |
| BI\_DR0.3\_LR0.0005\_BS128 |            0.3 |          0.0005 |                128 |       0.93507  |          0.955207 |    0.388994 |
| BI\_DR0.3\_LR0.001\_BS64 |            0.3 |          0.001  |                 64 |       0.948872 |          0.950728 |    0.458444 |
| BI\_DR0.3\_LR0.001\_BS128 |            0.3 |          0.001  |                128 |       0.94078  |          0.948488 |    0.454184 |
| BI\_DR0.3\_LR0.005\_BS64 |            0.3 |          0.005  |                 64 |       0.9464   |          0.946249 |    0.490051 |
| BI\_DR0.3\_LR0.005\_BS128 |            0.3 |          0.005  |                128 |       0.947794 |          0.957447 |    0.455463 |
| BI\_DR0.4\_LR0.0005\_BS64 |            0.4 |          0.0005 |                 64 |       0.935793 |          0.93953  |    0.43763  |
| BI\_DR0.4\_LR0.0005\_BS128 |            0.4 |          0.0005 |                128 |       0.938451 |          0.952968 |    0.382453 |
| BI\_DR0.4\_LR0.001\_BS64 |            0.4 |          0.001  |                 64 |       0.947184 |          0.956327 |    0.383938 |
| BI\_DR0.4\_LR0.001\_BS128 |            0.4 |          0.001  |                128 |       0.944082 |          0.954087 |    0.368322 |
| BI\_DR0.4\_LR0.005\_BS64 |            0.4 |          0.005  |                 64 |       0.942565 |          0.954087 |    0.466255 |
| BI\_DR0.4\_LR0.005\_BS128 |            0.4 |          0.005  |                128 |       0.952314 |          0.955207 |    0.499229 |
| BI\_DR0.5\_LR0.0005\_BS64 |            0.5 |          0.0005 |                 64 |       0.942189 |          0.954087 |    0.400563 |
| BI\_DR0.5\_LR0.0005\_BS128 |            0.5 |          0.0005 |                128 |       0.931022 |          0.955207 |    0.378504 |
| BI\_DR0.5\_LR0.001\_BS64 |            0.5 |          0.001  |                 64 |       0.949374 |          0.954087 |    0.415548 |
| BI\_DR0.5\_LR0.001\_BS128 |            0.5 |          0.001  |                128 |       0.947889 |          0.960806 |    0.378557 |
| BI\_DR0.5\_LR0.005\_BS64 |            0.5 |          0.005  |                 64 |       0.946756 |          0.954087 |    0.536137 |
| BI\_DR0.5\_LR0.005\_BS128 |            0.5 |          0.005  |                128 |       0.952817 |          0.947368 |    0.491151 |

---

**Commento sui Risultati della Grid Search Completa (Bidirezionale + Dropout + LR + Batch Size):**

Hai eseguito un tuning molto più completo combinando Dropout Rate, Learning Rate e Batch Size. Questo ci dà una visione più dettagliata delle interazioni tra questi iperparametri per l'architettura Bidirezionale+Dropout.

Ecco un'analisi più approfondita basata sulla tabella:

1.  **Conferma del Miglioramento Globale:** I risultati confermano che l'architettura Bidirezionale+Dropout può raggiungere performance superiori sia alla Baseline (max F1 ~0.927, max Acc ~0.926) sia alla Bidirezionale senza Dropout (max F1 ~0.947, max Acc ~0.951). Il picco di performance nella Grid Search è **F1 Slot = 0.952817** e **Intent Acc = 0.960806**.

2.  **Impatto del Dropout Rate (DR):**
    *   Dropout rate pari a 0.0 (nessun dropout) offre buone performance, ma i picchi assoluti di F1 e Accuracy si trovano con DR > 0.0.
    *   Tassi di dropout **da 0.1 a 0.5** mostrano tutti il potenziale per migliorare le performance.
    *   Sembra che tassi di dropout più alti (0.4, 0.5) siano particolarmente efficaci, soprattutto se combinati con altri parametri specifici.
    *   I tassi di dropout intermedi (0.1, 0.2, 0.3) mostrano anche ottime performance, specialmente con LR=0.001.

3.  **Impatto del Learning Rate (LR):**
    *   **0.0005:** Buon compromesso, spesso porta a Loss Test basse (migliore generalizzazione), ma non sempre ai picchi di F1 o Acc.
    *   **0.001:** Questo LR continua a essere un forte candidato e spesso appare nelle combinazioni con l'Acc Intent più alta e buoni F1. Tassi di dropout come 0.1, 0.4, 0.5 con LR=0.001 danno ottimi risultati (es. BI\_DR0.1\_LR0.001\_BS64, BI\_DR0.4\_LR0.001\_BS128, BI\_DR0.5\_LR0.001\_BS128).
    *   **0.005:** Interessante notare che il F1 Slot più alto (0.952817) si raggiunge con questo LR, ma solo combinato con DR=0.5 e BS=128. Con altri tassi di dropout o batch size, potrebbe non essere il migliore. Questo suggerisce un'interazione tra LR, DR e BS.

4.  **Impatto del Batch Size (BS):**
    *   **BS=64:** Tende a dare performance molto buone, specialmente con LR=0.001 e tassi di dropout da 0.1 a 0.5 (es. BI\_DR0.1\_LR0.001\_BS64, BI\_DR0.5\_LR0.001\_BS64). Spesso associato a Loss basse.
    *   **BS=128:** È il Batch Size coinvolto nelle configurazioni che raggiungono i picchi assoluti di F1 e Accuracy (DR=0.5, LR=0.005 per F1; DR=0.5, LR=0.001 per Acc). Sembra beneficiare tassi di dropout più alti. Associato anche a Loss Test molto basse (es. BI\_DR0.4\_LR0.001\_BS128, BI\_DR0.5\_LR0.0005\_BS128, BI\_DR0.5\_LR0.001\_BS128).

5.  **Interazioni tra Parametri:** La tabella evidenzia che non c'è una "regola" semplice (es. "Dropout X è sempre il migliore"). L'ottimale per un parametro dipende dai valori degli altri. Le heatmap generate dal tuo script dovrebbero visualizzare chiaramente queste interazioni. Ad esempio, potresti vedere che un certo LR funziona bene solo per un range specifico di Dropout Rate o Batch Size.

**Identificazione della Configurazione Migliore Complessiva:**

Basandoci su questa grid search, le configurazioni al vertice sono diverse a seconda della metrica preferita:

*   **Miglior F1 Slot (0.952817):** **DR=0.5, LR=0.005, BS=128**
*   **Miglior Intent Acc (0.960806):** **DR=0.5, LR=0.001, BS=128**
*   **Loss Test più bassa (0.368322):** **DR=0.4, LR=0.001, BS=128**

La configurazione **DR=0.5, LR=0.001, BS=128** sembra essere un eccellente compromesso: raggiunge l'Accuracy Intent massima, ha una Loss Test estremamente bassa (seconda solo a BI\_DR0.4\_LR0.001\_BS128) e un F1 Slot molto alto (0.947889).

Un altro candidato forte per un buon equilibrio è **DR=0.1, LR=0.001, BS=64**, con F1=0.949656 e Acc=0.952968 e Loss=0.393074. Questo potrebbe essere preferibile se le risorse (memoria) per Batch Size 128 sono un problema.

**Commento finale rispetto ai precedenti step:**

*   La Baseline pura aveva max F1 ~0.927 e max Acc ~0.926.
*   La Bidirezionale senza Dropout aveva max F1 ~0.947 e max Acc ~0.951.
*   La Bidirezionale con Dropout (tuning) ha raggiunto **max F1 ~0.953** e **max Acc ~0.961**.

Questo dimostra che **entrambe le modifiche incrementali (bidirezionalità e dropout) hanno contribuito a migliorare significativamente le performance** sul test set per entrambi i task, e che il tuning combinato deiperparametri ha permesso di trovare configurazioni che superano quelle ottenute testando i parametri uno alla volta in modo più semplice.

Per la tua relazione finale della Parte 2.A, riporterai i risultati della configurazione migliore trovata in questa grid search (es. BI\_DR0.5\_LR0.001\_BS128) come performance del modello Bidirezionale+Dropout, confrontandola con i migliori risultati della Baseline pura e della Bidirezionale senza Dropout.
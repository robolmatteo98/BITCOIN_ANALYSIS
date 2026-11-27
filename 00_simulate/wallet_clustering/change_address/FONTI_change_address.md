# Fonti — change address e clustering Bitcoin

Questo documento raccoglie le fonti utilizzate per definire l’algoritmo di
rilevamento dei change address e per scegliere i segnali usati nel progetto.

## Paper principali

### Meiklejohn et al. — *A Fistful of Bitcoins* (IMC 2013)

- [Versione UCSD](https://cseweb.ucsd.edu/~savage/papers/Login13.pdf)
- [Versione ACM/IMC](https://conferences.sigcomm.org/imc/2013/papers/imc182-meiklejohnA.pdf)

È la fonte classica per il common-input-ownership heuristic e per il change
address heuristic. Il lavoro osserva che il change può essere identificato
quando un output nuovo viene successivamente speso secondo un pattern
compatibile; se più output soddisfano il pattern, il metodo preferisce non
selezionare alcun change per ridurre i falsi positivi.

### Möser e Narayanan — *Resurrecting Address Clustering in Bitcoin* (FC 2022)

- [Paper PDF](https://fc22.ifca.ai/preproceedings/87.pdf)
- [Versione arXiv](https://arxiv.org/abs/2107.05749)

È la fonte principale per la revisione dell’algoritmo. Il paper analizza 26
euristiche e mostra che le singole euristiche hanno spesso buona precisione,
ma copertura limitata. Per questo combina più segnali tramite voting e Random
Forest.

Le euristiche considerate includono:

- optimal change: il resto è più piccolo di ciascuno dei due o più input;
- address/script type: il tipo dell’output coincide con quello degli input;
- potenze di dieci e importi arrotondati;
- indirizzo fresco;
- fingerprint della transazione successiva;
- numero e ordine di input/output;
- versione, locktime, SegWit, RBF, fee e coin age.

Il lavoro impone inoltre che al massimo un output venga scelto come change e
introduce vincoli sul clustering per evitare il *cluster collapse*, cioè la
fusione incontrollata di entità diverse.

### Ghesmati et al. — *Studying Bitcoin privacy attacks and their impact on
Bitcoin-based identity methods* (IACR ePrint 2021/1088)

- [Paper PDF](https://eprint.iacr.org/2021/1088.pdf)

Riepiloga le euristiche comunemente usate:

- indirizzo fresco;
- stesso tipo di script;
- stesso indirizzo tra input e output;
- optimal change;
- importi arrotondati;
- fingerprint del wallet;
- peeling chain;
- esclusione delle CoinJoin dal common-input heuristic.

## Fonte aggiuntiva sull’aggregazione di più euristiche

### He et al. — *Bitcoin address clustering method based on multiple heuristic
conditions* (IET Blockchain, 2022)

- [Articolo Wiley/IET](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/blc2.12014)

Analizza l’uso combinato di più condizioni e sottolinea i limiti dei metodi
basati su una sola euristica. Discute anche l’uso della freschezza, del numero
di output e di condizioni diverse a seconda della struttura della
transazione.

## Riferimento implementativo

### BlockSci — change address heuristics

- [Implementazione BlockSci](https://github.com/citp/BlockSci/blob/master/src/heuristics/change_address.cpp)

È un riferimento software utile per confrontare le euristiche pratiche,
incluse address reuse, fresh address, optimal change e fingerprint del wallet.

## Decisioni implementative nel progetto

Nel nostro algoritmo attuale utilizziamo solo segnali disponibili nel database:

- coerenza del tipo di script tra input e output;
- indirizzo fresco rispetto allo storico osservato;
- output più piccolo del totale degli input;
- optimal change quando ci sono almeno due input;
- struttura della transazione;
- posizione dell’output;
- presenza di un unico output non-input;
- esclusione delle transazioni CoinJoin candidate.

Il tipo di script viene calcolato durante la decodifica dei file `.dat` e
salvato in `tx_output.fk_script_type_id`. Per gli input viene letto il tipo
dell’output precedente tramite `prev_transaction_id` e `prev_vout`.

## Limiti

Un change address non è marcato esplicitamente nella blockchain. Il risultato
è quindi una stima probabilistica e non una prova di proprietà comune.

In particolare:

- CoinJoin, batching e pagamenti collaborativi possono invalidare il
  common-input heuristic;
- wallet diversi possono usare lo stesso tipo di script;
- un output nuovo può essere sia un pagamento sia un change;
- un output unico che raccoglie tutti gli input può essere una consolidazione o
  un trasferimento interno;
- senza fee, locktime, RBF, script completo e comportamento futuro non si può
  implementare l’intero insieme di fingerprint della letteratura;
- un falso positivo può propagarsi transitivamente nel clustering.

Per questo il progetto usa una soglia conservativa e non seleziona alcun
change quando più output risultano ugualmente plausibili.

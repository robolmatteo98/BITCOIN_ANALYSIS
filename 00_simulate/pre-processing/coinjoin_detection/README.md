# CoinJoin detection

Questa cartella contiene una prima euristica di preprocessing per assegnare a
ogni transazione una probabilità di essere CoinJoin.

Lo score usa i dati già presenti nel database:

- numero di input e output;
- presenza di più input e più output;
- output positivi con lo stesso importo;
- numero e quota degli importi ripetuti.

La ripetizione degli importi è il segnale principale. Il risultato non è una
prova di CoinJoin: `coinjoin_score` è uno score compreso tra `0` e `1`.
La soglia e gli altri segnali possono essere calcolati al momento dell'analisi,
senza duplicarli nel database.

## Output

Lo script non modifica il database. Legge le transazioni e salva i risultati
in file intermedi:

```text
07_reports/pre-processing/coinjoin/coinjoin_scores.csv
07_reports/pre-processing/coinjoin/summary.json
```

Per eseguire il calcolo dalla root:

```bash
python3 00_simulate/pre-processing/coinjoin_detection/run.py --threshold 0.75
```

La soglia non modifica lo score: serve solo per valorizzare la colonna
`candidate` e contare le candidate nel riepilogo. È possibile modificare la
cartella di output con `--output-dir`.

## Motivazione metodologica

La rilevazione implementata è un'euristica pesata ispirata alla letteratura
sulla rilevazione delle transazioni CoinJoin. Non esiste una formula universale
valida per tutti i protocolli: JoinMarket, Wasabi e Whirlpool possono produrre
transazioni con strutture differenti.

Il segnale principale è la presenza di più output con lo stesso importo. In
una CoinJoin più partecipanti combinano i propri input e ricevono spesso
output di valore uguale. Questo criterio è utilizzato anche in studi di analisi
empirica della blockchain, pur con il rischio di falsi positivi.

I segnali usati nel progetto sono:

```text
has_multiple_inputs_outputs  → struttura con almeno 2 input e 2 output
has_participant_shape        → struttura con almeno 3 input e 3 output
has_repeated_output_amount   → presenza di importi di output ripetuti
repeated_output_share        → quota di output appartenenti a gruppi ripetuti
repeated_amount_groups       → numero di gruppi di importi ripetuti
largest_equal_group          → dimensione del gruppo con importo più frequente
```

I criteri relativi agli importi ripetuti sono ispirati al concetto di
`most common value` utilizzato negli algoritmi di identificazione CoinJoin.
Il criterio della quota degli output ripetuti è inoltre vicino all'idea di
verificare se il valore più frequente rappresenta una parte significativa
della transazione.

I pesi utilizzati nel codice:

```text
0.15  almeno 2 input e 2 output
0.20  almeno 3 input e 3 output
0.35  presenza di importi ripetuti
0.20  almeno il 50% degli output è ripetuto
0.10  almeno due gruppi di importi ripetuti
```

non sono presi direttamente da una pubblicazione. Sono una scelta progettuale
sperimentale, resa esplicita e modificabile, che combina segnali documentati
nella letteratura in un unico score compreso tra `0` e `1`.

Pertanto il risultato deve essere descritto come:

```text
euristica sperimentale ispirata alla letteratura
```

e non come classificatore certo o formula standard per tutte le CoinJoin.

## Riferimenti

1. Maksutov et al., *Unsupervised clustering of bitcoin transactions*,
   Financial Innovation, 2023. L'articolo descrive come potenziali CoinJoin le
   transazioni con almeno due output dello stesso valore e segnala la
   possibilità di falsi positivi:
   <https://link.springer.com/article/10.1186/s40854-023-00525-y>

2. Eldefrawy, Gehani e Matton, *Longitudinal Analysis of Misuse of Bitcoin*,
   2019. Il metodo utilizza il valore di output più frequente, il numero di
   partecipanti e la compatibilità tra input e output per identificare
   transazioni di mixing:
   <https://keldefrawy.github.io/pubs/2019/btc_acns2019.pdf>

3. Schnoering e Vazirgiannis, *Heuristics for Detecting CoinJoin Transactions
   on the Bitcoin Blockchain*, 2023. Il lavoro evidenzia che protocolli diversi
   producono strutture diverse e propone euristiche specifiche per JoinMarket,
   Wasabi e Whirlpool:
   <https://arxiv.org/abs/2311.12491>

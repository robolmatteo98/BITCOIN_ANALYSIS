# Utilizzo del PageRank

Il PageRank viene eseguito come fase di preprocessing e i punteggi calcolati vengono salvati nella tabella di appoggio `address_pagerank`.

Il PageRank misura l'importanza di un address come destinatario dei flussi nella rete: un valore elevato indica che riceve transazioni da molti address o da address a loro volta importanti. Nella `role_detection`, address con PageRank elevato possono rappresentare possibili hub, aggregatori o destinatari centrali; nella `anomaly_detection` il punteggio può fornire un contesto storico per valutare la rilevanza e i cambiamenti anomali dei nodi e delle relazioni.

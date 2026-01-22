Un possibile metodo di geolocalizzazione di un indirizzo Bitcoin tramite indirizzo IP si basa sull’analisi della propagazione di una transazione all’interno della rete. 
Quando una transazione viene inoltrata, ogni nodo della rete Bitcoin è connesso a un insieme limitato di altri nodi (tipicamente otto), dai quali riceve aggiornamenti per mantenersi sincronizzato con la rete globale e dei quali conosce l’indirizzo IP.

Monitorando l’ordine temporale con cui una transazione viene ricevuta dai diversi indirizzi IP, è possibile stimare il nodo di origine: gli indirizzi IP che ricevono per primi la transazione sono probabilmente più vicini alla sorgente rispetto a quelli che la ricevono successivamente. 
Più nodi si controllano all'interno della rete Bitcoin, più alta sarà la probabilità di ottenere una stima più precisa della gelolcalizzazione dell'indirizzo.

Nel corso degli anni sono state adottate delle contromisure per ridurre l’efficacia di questo metodo, tra cui:
- l’introduzione di ritardi casuali nella propagazione delle transazioni tra i nodi;
- il continuo cambiamento dei nodi di peer-to-peer da cui ciascun nodo riceve gli aggiornamenti;
- Tor, VPN.
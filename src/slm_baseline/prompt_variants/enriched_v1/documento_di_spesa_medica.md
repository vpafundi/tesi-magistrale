Ti sarà fornito un testo classificato come documento_di_spesa_medica.

Considera che:
* le sole sotto-categorie ammesse sono: fattura, ticket_ssn, scontrino, preventivo_di_spesa, ricevuta_di_pagamento, documento_di_spesa_altro;
* se all'interno della stessa pagina sono presenti contemporaneamente una ricevuta di pagamento e un'altra categoria di documento di spesa, non classificare quella pagina come ricevuta di pagamento ma dai priorità all'altra categoria di documento di spesa identificato.

Utilizza le seguenti regole di classificazione di dettaglio:

fattura
È un documento ufficiale emesso da un fornitore o prestatore di servizi che attesta l'avvenuto pagamento o richiede pagamento posticipato di una merce o di un servizio, e contiene dettagli come la descrizione dei beni o servizi, l'importo totale, le aliquote IVA applicate e i dati del fornitore e del cliente (come ad esempio la ricevuta fiscale).
Indizi: fattura, numero/data fattura, cedente/cessionario, P.IVA/C.F., imponibile, IVA, totale, descrizione di prestazioni o beni.

ticket_ssn
È il documento che attesta la prenotazione e/o il pagamento di una prestazione sanitaria tramite il Sistema Sanitario Nazionale (SSN). Questo documento può assumere diverse forme, come una ricevuta di prenotazione, una fattura o un modulo di pagamento, e indica dettagli come il tipo di prestazione, l'importo pagato, la data e i dati del paziente.
Indizi: SSN, Servizio Sanitario Nazionale, ticket, CUP, prenotazione sanitaria, impegnativa, ricetta, prestazione sanitaria, codice prestazione, dati del paziente.

scontrino
È un documento commerciale emesso al momento dell'acquisto di beni o servizi, generalmente da esercizi commerciali al dettaglio. Lo scontrino riporta i dettagli della transazione, come i prodotti acquistati, le quantità, i prezzi e l'IVA applicata.
Indizi: scontrino, documento commerciale, corrispettivo, prodotti acquistati, totale pagato.

preventivo_di_spesa
È un documento che stima i costi di un intervento o acquisto prima che venga effettuato. Non rappresenta ancora un impegno di pagamento, ma permette al cliente di conoscere l'importo previsto e di valutare l'opportunità di procedere.
Indizi: preventivo, offerta, quotazione, stima, proposta economica, validità dell'offerta/preventivo.

ricevuta_di_pagamento
È un documento che attesta l'avvenuto pagamento di una spesa o di un servizio e indica che la somma dovuta è stata effettivamente versata dal cliente (esempio: ricevuta di pagamento elettronico, contabile bonifico, pagopa, estratto conto…).
Indizi: pagamento effettuato/eseguito, pagato, quietanza, bonifico, transazione, PagoPA, contabile bancaria, ricevuta POS/carta.

documento_di_spesa_altro
Utilizza questa categoria esclusivamente per i documenti di spesa medica che non rientrano in nessuna delle sotto-categorie precedentemente elencate.

Il testo del segmento segue nel messaggio utente (Macro e documento).

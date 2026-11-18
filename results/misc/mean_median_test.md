this test is to determine whether document, vocab or corpus mean/median is preferred for masking.

Reuters 1 doc, 5 authors

placeholder       | AOPC top | AOPC random | AOPC bottom | gap (top-random)
delete            | +0.0384  | +0.0021     | -0.0098     | +0.0364
mask              | +0.0398  | +0.0026     | -0.0110     | +0.0372
vocab_mean        | +0.0339  | +0.0109     | -0.0074     | +0.0231
vocab_median      | +0.0350  | +0.0106     | -0.0070     | +0.0244
corpus_mean       | +0.0407  | +0.0045     | -0.0106     | +0.0362
corpus_median     | +0.0412  | +0.0043     | -0.0094     | +0.0369
doc_mean          | +0.0413  | +0.0040     | -0.0108     | +0.0373
doc_median        | +0.0403  | +0.0035     | -0.0098     | +0.0368

corpus and doc mean/median are very similar, defaulting to the established corpus mean

Reddit
placeholder       | AOPC top | AOPC random | AOPC bottom | gap (top-random)
delete            | +0.0463  | +0.0058     | -0.0312     | +0.0405
mask              | +0.0379  | -0.0018     | -0.0231     | +0.0397
vocab_mean        | +0.0384  | +0.0032     | -0.0282     | +0.0352
vocab_median      | +0.0369  | +0.0039     | -0.0273     | +0.0330
corpus_mean       | +0.0449  | +0.0047     | -0.0298     | +0.0402
corpus_median     | +0.0455  | +0.0036     | -0.0304     | +0.0419
doc_mean          | +0.0457  | +0.0043     | -0.0323     | +0.0414
doc_median        | +0.0461  | +0.0047     | -0.0325     | +0.0414
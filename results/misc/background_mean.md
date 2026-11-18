# Background mean: zero-filled vs available-case

// probably not accurate anymore

Date: 2026-11-16
Matching file (zero-filled, reference): matching_20261022-193008.json (scoring: scoring_20261022-192941.json, deltas: deltas_20261022-192941.pt, cache series 20261022-192344)
Available-case results: (to add)
Data: darkreddit, luar-mud, plain, doc_mean, units: word, n = 1; 10 authors, 50 sampled val comments per author (seed 0), 6817 train comments, 2276 test comments. Full LUAR (ceiling): 88.14% (2006/2276).

## Names

- cosine distance -> score 1
- Euclidean distance -> score 2
- relative frequency -> frequency
- zero-filled -> original background mean
- available-case -> new background mean

All scores damped: final = corrected * n / (n + d), d = 5 unless stated, n = |C_A(w)|.

## What was done

(to add)
## TF-SVM: top-1 accuracy on test

| signature                | mean           |            k = 1 |            k = 2 |            k = 5 |           k = 10 |           k = 20 |           k = 50 |
| ------------------------ | -------------- | ---------------: | ---------------: | ---------------: | ---------------: | ---------------: | ---------------: |
| cosine distance          | zero-filled    |             33.5 |             42.3 |             55.1 |             57.5 |             62.8 |             66.8 |
| cosine distance          | available-case |             40.8 |             49.0 |             55.0 |             60.8 |             64.2 |             67.2 |
| cosine distance (d = 20) | zero-filled    |             45.3 |             52.7 |             57.9 |             60.3 |             64.3 |             68.2 |
| cosine distance (d = 20) | available-case |             41.7 |             49.6 |             57.8 |             61.5 |             65.3 |             68.5 |
| Euclidean distance       | zero-filled    |             40.9 |             40.9 |             50.8 |             59.4 |             62.4 |             67.4 |
| Euclidean distance       | available-case |             40.7 |             40.7 |             52.5 |             58.3 |             62.8 |             66.2 |
| relative frequency       | zero-filled    |             45.4 |             51.2 |             60.1 |             64.9 |             67.3 |             70.6 |
| relative frequency       | available-case |             45.4 |             49.7 |             58.9 |             62.7 |             66.2 |             69.5 |
| random, mean (min–max)   | both           | 18.9 (14.1–27.7) | 20.5 (16.3–27.4) | 25.0 (19.7–35.3) | 30.9 (24.6–40.5) | 41.7 (32.4–51.6) | 55.9 (45.3–64.1) |

All units (k = all): 73.6% (both means).

## Δ-sum: top-1 accuracy on test

| signature                | mean           |            k = 1 |            k = 2 |            k = 5 |           k = 10 |           k = 20 |           k = 50 |
| ------------------------ | -------------- | ---------------: | ---------------: | ---------------: | ---------------: | ---------------: | ---------------: |
| cosine distance          | zero-filled    |             49.6 |             58.3 |             57.6 |             60.8 |             60.2 |             61.2 |
| cosine distance          | available-case |             51.4 |             58.4 |             57.8 |             61.2 |             62.2 |             56.9 |
| cosine distance (d = 20) | zero-filled    |             59.3 |             64.1 |             63.9 |             63.5 |             61.8 |             61.0 |
| cosine distance (d = 20) | available-case |             59.6 |             64.4 |             62.9 |             63.9 |             64.3 |             58.4 |
| Euclidean distance       | zero-filled    |             53.4 |             62.3 |             62.2 |             64.0 |             60.0 |             60.8 |
| Euclidean distance       | available-case |             58.0 |             64.2 |             66.2 |             63.8 |             66.0 |             64.7 |
| relative frequency       | zero-filled    |             58.0 |             61.7 |             65.5 |             66.0 |             65.4 |             63.8 |
| relative frequency       | available-case |             58.0 |             60.9 |             64.1 |             64.9 |             64.9 |             63.1 |
| random, mean (min–max)   | zero-filled    | 21.0 (10.8–34.3) | 25.0 (13.7–38.4) | 32.7 (22.4–44.2) | 38.4 (26.8–47.8) | 45.6 (36.6–55.3) | 53.1 (46.6–58.5) |
| random, mean (min–max)   | available-case |  13.3 (6.8–29.8) |  20.0 (7.2–37.2) | 31.6 (19.1–40.8) | 38.1 (26.1–49.6) | 44.1 (37.3–55.6) | 50.6 (44.6–58.8) |

All units (k = all): 59.8% (zero-filled), 60.1% (available-case).

## Paired test

(to add if computed: McNemar, available-case vs zero-filled per signature and k)

## Notes

- Units cached for only one author: with the available-case mean their corrected value is 0 (corrected Δ is the zero vector), so they drop out of the rankings and add nothing to Δ-sum.
- Relative frequency with the zero-filled mean picks many such units. Its Δ-sum result with available-case Δs is therefore not a fair comparison and is not reported.

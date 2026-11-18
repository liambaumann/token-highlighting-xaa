Fingerprint: 5 most important tokens per author

source: masking results, luar-mud, plain mode, delete perturbation, n=1 (single-token occlusion)
- darkreddit: results/masking/darkreddit/delete/luar-mud_plain/user1_0_n1.json
- reuters: results/masking/reuters/delete/luar-mud_plain/AaronPressman_2_n1.json

score = base similarity - similarity after deleting that token (cosine to the author's centroid). higher = token pulls the doc more towards its author. `Ġ` = leading space in the RoBERTa BPE token.

darkreddit, user1, doc 0 (97 tokens, base similarity 0.7106)
| rank | pos | token | score | context |
|---|---|---|---|---|
| 1 | 80 | `,` | 0.0303 | In other words **,** fakes are getting |
| 2 | 81 | `Ġf` | 0.0270 | In other words, **f**akes are getting pretty |
| 3 | 89 | `ĠIts` | 0.0250 | pretty good. **Its** a disappointing revelation. |
| 4 | 94 | `Ġ` | 0.0184 | a disappointing revelation. **(space)** `</s>` |
| 5 | 95 | `Ġ` | 0.0184 | disappointing revelation. **(space)** `</s>` |

reuters, AaronPressman, doc 2 (512 tokens, truncated; base similarity 0.8926)
| rank | pos | token | score | context |
|---|---|---|---|---|
| 1 | 473 | `Ġ"` | 0.0209 | said Luker. **"**But in the long |
| 2 | 474 | `But` | 0.0109 | Luker. " **But** in the long run |
| 3 | 338 | `ĠCongress` | 0.0070 | of the Library of **Congress** every four months. |
| 4 | 470 | `ĠLuk` | 0.0062 | adding hardware, said **Luk**er. "But |
| 5 | 232 | `Ġ"` | 0.0060 | , describes it as **"**the busiest public exchange |

the top tokens are mostly punctuation, whitespace and style rather than topic words: user1's comma after "In other words", "Its" without the apostrophe and the double space after the final full stop; for AaronPressman the opening quote marks and "But" starting a quote. "Congress" and "Luker" are the only content tokens. scores for user1 (short post) are higher than for AaronPressman (512 tokens), since each token carries more of the embedding in a short doc.

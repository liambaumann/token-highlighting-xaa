Dataset lengths (from drlen.py)

tokens = LUAR-MUD tokenizer, no special tokens. words = whitespace split (punctuation counts toward word length).

tokens per doc
| split | docs | mean | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|---|---|
| darkreddit train | 6817 | 141.7 | 37 | 54 | 66 | 97 | 160 | 267.4 | 3245 |
| darkreddit val | 2275 | 139.4 | 39 | 54 | 68 | 98 | 162 | 265 | 1831 |
| darkreddit test | 2276 | 141.3 | 40 | 54 | 67 | 99 | 167 | 268.5 | 2179 |
| reuters train | 2500 | 657.1 | 62 | 459 | 554 | 656 | 747 | 852 | 1976 |
| reuters test | 2500 | 667.1 | 75 | 466 | 559 | 666 | 762 | 869 | 1901 |

share of docs with at least n tokens
| split | >=16 | >=32 | >=64 | >=128 | >=256 | >=512 |
|---|---|---|---|---|---|---|
| darkreddit train | 100.0% | 100.0% | 78.4% | 35.2% | 10.9% | 2.7% |
| darkreddit val | 100.0% | 100.0% | 78.9% | 35.9% | 10.6% | 2.3% |
| darkreddit test | 100.0% | 100.0% | 79.3% | 37.0% | 11.2% | 2.1% |
| reuters train | 100.0% | 100.0% | 100.0% | 99.2% | 98.3% | 82.4% |
| reuters test | 100.0% | 100.0% | 100.0% | 99.5% | 98.7% | 83.2% |

words
| split | words/doc mean | words/doc median | chars/word mean | chars/word median | tokens/word |
|---|---|---|---|---|---|
| darkreddit train | 106.6 | 73 | 4.57 | 4 | 1.33 |
| darkreddit val | 104.3 | 74 | 4.59 | 4 | 1.34 |
| darkreddit test | 106.6 | 75 | 4.57 | 4 | 1.32 |
| reuters train | 501.8 | 506 | 5.13 | 5 | 1.31 |
| reuters test | 509.6 | 515 | 5.13 | 5 | 1.31 |

reuters docs are ~7x longer than darkreddit posts and ~82% exceed LUAR's 512-token limit (only 2-3% of darkreddit posts do). every doc in both datasets has at least 32 tokens, so 32-token windows always fit. both tokenize at ~1.3 tokens/word.

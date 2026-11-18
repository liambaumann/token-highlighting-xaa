AaronPressman, doc 2: plain vs chunked (32-token pieces)

base similarity: plain 0.8616, chunked 0.8723 (all n)

n=1
| k | pct | top plain | top chunked | random avg plain | random avg chunked |
|---|---|---|---|---|---|
| 5 | 1% | 0.7883 | 0.8605 | 0.8660 | 0.8739 |
| 15 | 3% | 0.7714 | 0.8394 | 0.8653 | 0.8760 |
| 26 | 5% | 0.7779 | 0.8281 | 0.8594 | 0.8741 |
| 51 | 10% | 0.7418 | 0.8147 | 0.8768 | 0.8771 |

n=2
| k | pct | top plain | top chunked | random avg plain | random avg chunked |
|---|---|---|---|---|---|
| 4 | 1% | 0.7928 | 0.8626 | 0.8662 | 0.8722 |
| 16 | 3% | 0.7910 | 0.8484 | 0.8507 | 0.8732 |
| 26 | 5% | 0.7789 | 0.8395 | 0.8699 | 0.8727 |
| 52 | 10% | 0.7729 | 0.8240 | 0.8690 | 0.8758 |

n=3
| k | pct | top plain | top chunked | random avg plain | random avg chunked |
|---|---|---|---|---|---|
| 6 | 1% | 0.7870 | 0.8631 | 0.8637 | 0.8739 |
| 15 | 3% | 0.7945 | 0.8535 | 0.8627 | 0.8743 |
| 27 | 5% | 0.7967 | 0.8438 | 0.8751 | 0.8758 |
| 51 | 10% | 0.7670 | 0.8323 | 0.8641 | 0.8760 |

chunked top-k drops are consistently smaller than plain; chunked random removal barely moves similarity at all.

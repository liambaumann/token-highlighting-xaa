Reuters validation

setting	        best        epoch    top-1	MAP
original        LUAR	    0	    63.20%	75.82%
top 2 layers,   lr 5e-6	    1 of 3	62.40%	75.35%
top 2 layers,   lr 2e-5	    1 of 3	61.40%	74.31%
top 1 layer,    lr 2e-5	    1 of 3	61.40%	74.63%

Reddit validation

setting	        best        epoch    top-1	MAP
original        LUAR	    0	    83.03%	89.26%
top 2 layers,   lr 2e-5	    7 of 10	87.47%	92.62%

Reddit retrieval test

	                top-1	soft top-5	soft    top-10	MAP
original, plain	    88.14%	99.30%	    100%	93.03%
fine-tuned, plain	88.97%	99.30%	    100%	93.45%
original, chunk32	82.73%	97.45%	    100%	89.15%
fine-tuned, chunk32	86.16%	99.03%	    100%	91.75%



LOSS

Reuters validation (50 authors, 500 held-back training articles)

setting                  epoch  train loss  top-1    MAP
original LUAR            0      -           63.20%   75.82%
top 2 layers, lr 2e-5    1      1.4324      61.40%   74.31%
top 2 layers, lr 2e-5    2      0.6083      60.00%   73.54%
top 2 layers, lr 2e-5    3      0.5555      60.80%   74.03%
top 2 layers, lr 5e-6    1      2.3893      62.40%   75.35%
top 2 layers, lr 5e-6    2      1.4786      61.80%   74.78%
top 2 layers, lr 5e-6    3      0.9704      62.00%   75.04%
top 1 layer, lr 2e-5     1      1.7049      61.40%   74.63%
top 1 layer, lr 2e-5     2      0.7027      61.00%   74.27%
top 1 layer, lr 2e-5     3      0.6530      61.20%   74.53%
all runs: early stopping after epoch 3, best epoch 0 (original)

Reddit validation (10 authors, 2,275 validation comments)

setting                  epoch  train loss  top-1    MAP
original LUAR            0      -           83.03%   89.26%
top 2 layers, lr 2e-5    1      0.8396      84.26%   90.01%
top 2 layers, lr 2e-5    2      0.0910      84.62%   90.83%
top 2 layers, lr 2e-5    3      0.0284      85.71%   91.51%
top 2 layers, lr 2e-5    4      0.0141      86.15%   91.78%
top 2 layers, lr 2e-5    5      0.0182      86.64%   92.14%
top 2 layers, lr 2e-5    6      0.0090      86.81%   92.27%
top 2 layers, lr 2e-5    7      0.0141      87.47%   92.62%   <- best
top 2 layers, lr 2e-5    8      0.0059      86.81%   92.31%
top 2 layers, lr 2e-5    9      0.0078      86.73%   92.31%
top 2 layers, lr 2e-5    10     0.0059      87.25%   92.54%
early stopping after epoch 10, best epoch 7

Reddit retrieval test (fine-tuned = top 2 layers, lr 2e-5, epoch 7)

setting                  top-1    soft top-5   soft top-10   MAP
original, plain          88.14%   99.30%       100%          93.03%
fine-tuned, plain        88.97%   99.30%       100%          93.45%
original, chunk32        82.73%   97.45%       100%          89.15%
fine-tuned, chunk32      86.16%   99.03%       100%          91.75%
# PyTorch CRF with N-best Decoding

Implementation of Conditional Random Fields (CRF) in PyTorch 1.0. It supports top-N most probable paths decoding.

The package is based on [pytorch-crf](https://github.com/kmkurn/pytorch-crf) with only the following differences

- Method `_viterbi_decode` that decodes the most probable path get optimized. Running time gets reduced to 50% or less with batch size 15+ and sequence length 20+
- The class now supports decoding top-N most probable paths through the implementation of the method `_viterbi_decode_nbest`

## Requirements

- Python 3 (>= 3.6)
- PyTorch (>= 1.0)

## Installation

```bash
pip install pytorchcrf
```

## Examples

```python
>>> import torch
>>> from pytorchcrf import CRF
>>> num_tags = 5                        # number of tags is 5
>>> model = CRF(num_tags)
>>> seq_length = 3                      # maximum sequence length in a batch
>>> batch_size = 2                      # number of samples in the batch
>>> emissions = torch.randn(seq_length, batch_size, num_tags)

# Computing log likelihood
>>> tags = torch.tensor([[2, 3], [1, 0], [3, 4]], dtype=torch.long)  # (seq_length, batch_size)
>>> model(emissions, tags)

# Decoding
>>> model.decode(emissions)             # decoding the best path
>>> model.decode(emissions, nbest=3)    # decoding the top 3 paths
```

# PyTorch CRF with N-best Decoding

Implementation of Conditional Random Fields (CRF) in PyTorch 1.0. It supports top-N most probable paths decoding.

The package is based on [pytorch-crf](https://github.com/kmkurn/pytorch-crf) with only the following differences

- Method `_viterbi_decode` that decodes the most probable path get optimized. Running time gets reduced to 50% or less with batch size 15+ and sequence length 20+
- The class now supports decoding top-N most probable paths through the implementation of the method `_viterbi_decode_nbest`

## Requirements

- Python 3 (>= 3.6)
- PyTorch 1.0

## Installation

```bash
pip install pytorchcrf
```

## Examples

```python
>>> import torch
>>> from pytorchcrf import CRF
>>> num_tags = 4  # number of tags is 4
>>> model = CRF(num_tags)
>>> seq_length = 3  # maximum sequence length in a batch
>>> batch_size = 2  # number of samples in the batch
>>> emissions = torch.randn(seq_length, batch_size, num_tags)

>>> model.decode(emissions)
>>> model.decode(emissions, nbest=3)
```

# Visual Attention mechanisms

Repo of visual attention mechanisms from various papers, including their
implementations, testing, and studies of them in various model architectures.

## Using and viewing

Implementations are done in [PyTorch](https://pytorch.org/) (v0.3 for now).
Models are implemented as extending `nn.Module`, and done in a way that lets
them be import and used in any PyTorch Module.

Mechanism tests and studies are done in Jupyter notebooks, which can be view
(but not run) without having PyTorch installed.

## References

* [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) (Mnih, Graves, Kavukcuoglu. 2014): recurrent \[visual\] attention model (RAM)


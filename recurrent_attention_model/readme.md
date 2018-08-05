# Recurrent Visual Attention Model

Implements the recurrent visual attention model as initially described in
[Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) (Mnih,
Graves, Kavukcuoglu. 2014), along with some of the improvements described in
[Multiple Object Recognition with Visual Attention](https://arxiv.org/abs/1412.7755)
(Ba, Mnih, Kavukcuoglu. 2014) as optional model and training arguments. I've also
tested some of my own ideas to improve upon the model and training; the ideas
tested are listed below, and the results of those tests can be found in the
accompanying Jupyter notebooks. More specific details of the ideas can be found
in the comments within the code, or just send me a message.

## Ideas tested

* Action network output as location network input
* Recurrent action and/or location networks
* Convoluted glimpse extractions
* Use the classification loss from every step
* Give a positive reward for each step where there's a correct classification
* End a training episode early if there's a correct classification

## References

* [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) (Mnih, Graves, Kavukcuoglu. 2014): recurrent \[visual\] attention model (RAM)
* [Multiple Object Recognition with Visual Attention](https://arxiv.org/abs/1412.7755) (Ba, Mnih, Kavukcuoglu. 2014): improvements to the RAM architecture, plus an algorithm for learning multiple sequential targets
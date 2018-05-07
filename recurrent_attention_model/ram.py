'''
Implements the Recurrent Attention Model (RAM), as described in "Recurrent
Models of Visual Attention" (Mnih, et al.)
'''

import torch
import torch.nn as nn

'''
The RecurrentAttentionModel is overall an RNN. The core network of the model
takes a glimpse representation as input and combines it with the internal
representation from the previous time step, `h_t1`, to produce the new internal
state of the model `h_t`. The location and action networks use the internal
state representation `h_t` of the model to produce the next location to attend
to (`loc_t`) and the action/classification (`a_t`). This basic RNN iteration is
repeated for a variable number of steps.
'''
class RecurrentAttentionModel(nn.Module):
    def __init__(self):
        super(RecurrentAttentionModel, self).__init__()


'''
Given the location `loc_t1` and input image `x_t`, the GlimpseNetwork uses the
GlimpseSensor to extract retina representation `rho(x_t, loc_t1)`. The retina
representation and glimpse location are then mapped into a hidden space using
independent linear layers, followed by a ReLU activation and another linear
layer that combines the information from both components. The GlimpseNetwork
defines a trainable bandwidth-limited sensor for the attention network producing
the glimpse representation.
'''
class GlimpseNetwork(nn.Module):
    def __init__(self):
        super(GlimpseNetwork, self).__init__()

'''
Given the coordinates of the glimpse and an input image, the GlimpseSensor
extracts a retina-like representation `rho(x_t, loc_t1)`, centered at `loc_t1`,
that contains multiple resolution patches.
'''
class GlimpseSensor(nn.Module):
    def __init__(self):
        super(GlimpseSensor, self).__init__()
'''
Implements the Recurrent Attention Model (RAM), as described in "Recurrent
Models of Visual Attention" (Mnih, et al.)
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.autograd import Variable

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
    '''
    Inputs:
    `image_size`: the size of the input image as an `int`. For now, assumes the
            image is square.
    `in_channels`: number of channels in the input images (typically 1 for
        grayscale, 3 for RGB, 4 for RGBa)
    `glimpse_h_size`: size of the hidden layer for the independent glimpse linear
        layer representation
    `loc_h_size`: size of the hidden layer for the independent location linear
        layer representation (for each dimension!)
    `glimpse_network_size`: size of the hidden layer for the combined
        glimpse/location linear layer representation
    `rnn_state_size`: size of the RNN state vectors
    `action_state_size`: size of action state vectors
    `rnn_type`: the type of RNN cell to use. Should be one of: `'RNN'`, `'LSTM'`
        (default), or `'GRU'`
    `num_rnn_layers`: number of recurrent layers to use in the recurrent network
    `nonlinearity`: the nonlinear activation function to use when `rnn_type` is
        `'RNN'`; ignored when using an `'LSTM'` or `'GRU'`. Should be either
        `'tanh'` or `'relu'`
    `glimpse_sizes`: a list of either `int`s or `float`s representing the
        (square) size of each glimpse to extract from the input image. The sizes
        should be in increasing order, but this will still make sure they're
        ordered correctly. The length of `glimpse_sizes` will also be the number
        of glimpse patches to extract. If `glimpse_sizes` is a list of `int`s,
        they'll be treated as exact sizes; if it's a list of `float`s, they'll
        be treated as percentages of `image_size`, and should thus all be <= 1.
        Padding will be added to the inputs to allow glimpses near the edges.
        The actual sizes of the resulting glimpses will be the same size as the
        first glimpse.
    `pad_imgs`: whether or not the input images need padding applied. Defaults
        to `True`, but set to `False` if the input images already have padding
        applied; make sure it's of the correct size!
    `learn_kernels`: if `True`, the kernels used to create the glimpses will
        have learnable parameters. If `False` (the default), the glimpses will
        be created using bicubic interpolation, and the kernels used will have
        fixed weights
    `a`: the scaling factor used for bicubic interpolation during resizing
    `preserve_out_channels`: if `True`, the number of out channels is kept in
        each of the glimpses. If `False` (default), the number of out channels
        in the glimpses is 1
    `action_network`: a PyTorch module for this RAM module to use to select the
        next action. Should take as input a tensor of size `(batch_size, rnn_state_size)`
        and return a tensor of size `(batch_size, action_state_size)` 
        representing probabilities for each action. If `None` (default),
        then a linear layer is created and used
    `location_network`: a PyTorch module for this RAM module to use to select
        the next location to center glimpses around. Should take as input a
        tensor of size `(batch_size, rnn_state_size)` and return a 2-tuple of
        tensors of size `(batch_size, image_size)` in a tuple, each representing
        probabilities for the location along each axis. If `None` (default),
        then two linear layers are created and used
    `dropout`: dropout drop probability
    '''
    def __init__(self, image_size, in_channels, glimpse_h_size, loc_h_size,
                glimpse_network_size, rnn_state_size, action_state_size,
                rnn_type = 'LSTM', num_rnn_layers = 1, nonlinearity = 'tanh',
                glimpse_sizes = [0.05, 0.1, 0.3, 0.6], pad_imgs = True,
                learn_kernels = False, a = -0.5, preserve_out_channels = False,
                action_network = None, location_network = None,
                dropout = 0.1):
        super(RecurrentAttentionModel, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = in_channels if preserve_out_channels else 1
        self.glimpse_h_size = glimpse_h_size
        self.loc_h_size = loc_h_size
        self.glimpse_network_size = glimpse_network_size
        self.rnn_state_size = rnn_state_size
        self.action_state_size = action_state_size
        rnn_type = rnn_type.upper()
        if rnn_type not in ['RNN', 'LSTM', 'GRU']:
            raise ValueError(
                """Incorrect RNN type (expected one of 'RNN', 'LSTM', or 'GRU',
                but found '%s')""" % rnn_type
            )
        else:
            self.rnn_type = rnn_type
        self.num_rnn_layers = num_rnn_layers

        # Basic layers
        self.drop = nn.Dropout(dropout)

        # GlimpseNetwork
        self.glimpse_network = GlimpseNetwork(
            image_size = image_size, in_channels = in_channels,
            glimpse_h_size = glimpse_h_size, loc_h_size = loc_h_size,
            network_size = glimpse_network_size, glimpse_sizes = glimpse_sizes,
            pad_imgs = pad_imgs, learn_kernels = learn_kernels, a = a,
            preserve_out_channels = preserve_out_channels, dropout = dropout,
        )
        # Core recurrent network
        if rnn_type == 'RNN':
            nonlinearity = nonlinearity.lower()
            if nonlinearity not in ['tanh', 'relu']:
                raise ValueError(
                    """Incorrect RNN nonlinearity function (expected either
                    'tanh' or 'relu', but found '%s')""" % nonlinearity
                )
            else:
                self.rnn_type = '_'.join([self.rnn_type, nonlinearity])
            self.rnn_stack = nn.ModuleList([nn.RNNCell(
                input_size = glimpse_network_size if i == 0 else rnn_state_size,
                hidden_size = rnn_state_size, nonlinearity = nonlinearity
            ) for i in range(num_rnn_layers)])
        else:
            rnn_cell = getattr(nn, rnn_type + 'Cell')
            self.rnn_stack = nn.ModuleList([rnn_cell(
                input_size = glimpse_network_size if i == 0 else rnn_state_size,
                hidden_size = rnn_state_size
            ) for i in range(num_rnn_layers)])
        # Action network
        if action_network is None:
            action_network = nn.Sequential(
                nn.Linear(rnn_state_size, action_state_size),
                nn.Softmax(dim = -1)
            )
        self.action_network = action_network
        # Location network
        if location_network is None:
            class LocationNetwork(nn.Module):
                def __init__(self, in_size, out_size):
                    super(LocationNetwork, self).__init__()
                    self.in_size = in_size
                    self.out_size = out_size
                    self.x_loc = nn.Sequential(
                        nn.Linear(in_size, out_size),
                        nn.Softmax(dim = -1)
                    )
                    self.y_loc = nn.Sequential(
                        nn.Linear(in_size, out_size),
                        nn.Softmax(dim = -1)
                    )

                def forward(self, h):
                    x = self.x_loc(h)
                    y = self.y_loc(h)
                    return (x, y)
            # Initialize a LocationNetwork
            location_network = LocationNetwork(rnn_state_size, image_size)
        self.location_network = location_network

        # Initialize weights
        self.init()

    def init(self):
        self.glimpse_network.init()
        for layer in self.rnn_stack:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal(p)
                else:
                    p.data.fill_(0)
        for layer in [self.action_network, self.location_network]:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal(p)
                else:
                    p.data.fill_(0)

    def init_rnn_states(self, batch_size):
        if self.rnn_type == 'LSTM':
            states = [(
                Variable(torch.zeros(batch_size, self.rnn_state_size)),
                Variable(torch.zeros(batch_size, self.rnn_state_size))
            ) for _ in range(self.num_rnn_layers)]
        else:
            states = [
                Variable(torch.zeros(batch_size, self.rnn_state_size))
                for _ in range(self.num_rnn_layers)
            ]
        return states

    '''
    Inputs:
    `imgs`: input tensor of size `(batch_size, channels, image_height, image_width)`
    `locs`: a list of 2-tuples representing the `(x_pos, y_pos)` for each image
        in the batch to center the extracted glimpses around
    `rnn_states`: the current hidden states for the layer(s) of the RNN network

    Outputs:
    `actions`: a tensor of size `(batch_size, action_state_size)` representing
        probabilities for each possible action
    `locs`: a 2-tuple of tensors of size `(batch_size, image_size)`, each
        representing probabilties for the x- and y-coordinates of the image,
        respectively, where the next glimpse should be centered around. Note
        that this output should be argmax'd and converted to a list of 2-tuples
        before being passed back to the network
    '''
    def forward(self, imgs, locs, rnn_states):
        # Glimpse network
        glimpse_net_out = self.glimpse_network(imgs, locs)
        rnn_in = self.drop(glimpse_net_out)
        # Core RNN network
        new_rnn_states = []
        for rnn, state in zip(self.rnn_stack, rnn_states):
            rnn_out = rnn(rnn_in, state)
            new_rnn_states.append(rnn_out)
            if isinstance(rnn_out, tuple):
                rnn_in = self.drop(rnn_out[0])
            else:
                rnn_in = self.drop(rnn_out)
        net_in = rnn_in
        # Action and Location networks
        actions = self.action_network(net_in)
        locs = self.location_network(net_in)

        return actions, locs


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
    '''
    Inputs:
    `image_size`: the size of the input image as an `int`. For now, assumes the
            image is square.
    `in_channels`: number of channels in the input images (typically 1 for
        grayscale, 3 for RGB, 4 for RGBa)
    `glimpse_h_size`: size of the hidden layer for the independent glimpse linear
        layer representation
    `loc_h_size`: size of the hidden layer for the independent location linear
        layer representation (for each dimension!)
    `network_size`: size of the hidden layer for the combined glimpse/location
        linear layer representation
    `glimpse_sizes`: a list of either `int`s or `float`s representing the
        (square) size of each glimpse to extract from the input image. The sizes
        should be in increasing order, but this will still make sure they're
        ordered correctly. The length of `glimpse_sizes` will also be the number
        of glimpse patches to extract. If `glimpse_sizes` is a list of `int`s,
        they'll be treated as exact sizes; if it's a list of `float`s, they'll
        be treated as percentages of `image_size`, and should thus all be <= 1.
        Padding will be added to the inputs to allow glimpses near the edges.
        The actual sizes of the resulting glimpses will be the same size as the
        first glimpse.
    `pad_imgs`: whether or not the input images need padding applied. Defaults
        to `True`, but set to `False` if the input images already have padding
        applied; make sure it's of the correct size!
    `learn_kernels`: if `True`, the kernels used to create the glimpses will
        have learnable parameters. If `False` (the default), the glimpses will
        be created using bicubic interpolation, and the kernels used will have
        fixed weights
    `a`: the scaling factor used for bicubic interpolation during resizing
    `preserve_out_channels`: if `True`, the number of out channels is kept in
        each of the glimpses. If `False` (default), the number of out channels
        in the glimpses is 1
    `dropout`: dropout drop probability
    '''
    def __init__(self, image_size, in_channels,
                glimpse_h_size, loc_h_size, network_size,
                glimpse_sizes = [0.05, 0.1, 0.3, 0.6],
                pad_imgs = True, learn_kernels = False, a = -0.5,
                preserve_out_channels = False, dropout = 0.1):
        super(GlimpseNetwork, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = in_channels if preserve_out_channels else 1
        self.glimpse_h_size = glimpse_h_size
        self.loc_h_size = loc_h_size
        self.network_size = network_size
        # Basic layers
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # Initialize the GlimpseSensor
        self.glimpse_sensor = GlimpseSensor(
            image_size = image_size, in_channels = in_channels, a = a,
            glimpse_sizes = glimpse_sizes, learn_kernels = learn_kernels,
            preserve_out_channels = preserve_out_channels, pad_imgs = pad_imgs
        )

        # Initialize the network
        self.glimpse_layer = nn.Linear(
            self.glimpse_sensor.unrolled_glimpse_size, glimpse_h_size
        )
        self.loc_embeds = nn.ModuleList([
            nn.Embedding(image_size, loc_h_size),
            nn.Embedding(image_size, loc_h_size)
        ])
        self.out_layer = nn.Linear(glimpse_h_size + 2*loc_h_size, network_size)

        # Initialize weights
        self.init()

    def init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
            else:
                p.data.fill_(0)

    '''
    Inputs:
    `imgs`: input tensor of size `(batch_size, channels, image_height, image_width)`
    `locs`: a list of 2-tuples representing the `(x_pos, y_pos)` for each image
        in the batch to center the extracted glimpses around

    Outputs:
    `glimpse_net`: a tensor of size `(batch_size, network_size)`
    '''
    def forward(self, imgs, locs):
        nbatches = imgs.size(0)
        # Get glimpse representations for the images
        glimpses = self.glimpse_sensor(imgs, locs).view(nbatches, -1)           # (nbatches, unrolled_glimpse_size)
        # Pass the glimpse representations through the linear layer
        glimpse_rep = self.drop(self.relu(self.glimpse_layer(                   # (nbatches, glimpse_h_size)
            self.relu(glimpses).view(nbatches, -1)
        )))
        # Embed the glimpse locations
        x_loc = Variable(                                                       # (nbatches, 1)
            torch.LongTensor([loc[0] for loc in locs]).unsqueeze(1)
        )
        y_loc = Variable(                                                       # (nbatches, 1)
            torch.LongTensor([loc[1] for loc in locs]).unsqueeze(1)
        )

        x_embeds = self.drop(self.relu(self.loc_embeds[0](x_loc)))              # (nbatches, loc_h_size)
        y_embeds = self.drop(self.relu(self.loc_embeds[1](y_loc)))              # (nbatches, loc_h_size)

        # Concatenate
        loc_embed = torch.cat([x_embeds, y_embeds], -1)                         # (nbatches, 2*loc_h_size)
        glimpse_net_in = torch.cat([glimpse_rep, loc_embed], -1)                # (nbatches, glimpse_h_size + 2*loc_h_size)

        # Pass through final linear layer
        glimpse_net = self.relu(self.out_layer(glimpse_net_in))                 # (nbatches, network_size)

        return glimpse_net
        

'''
Given the coordinates of the glimpse and an input image, the GlimpseSensor
extracts a retina-like representation `rho(x_t, loc_t1)`, centered at `loc_t1`,
that contains multiple resolution patches.
'''
class GlimpseSensor(nn.Module):
    '''
    Inputs:
    `image_size`: the size of the input image as an `int`. For now, assumes the
        image is square.
    `in_channels`: number of channels in the input images (typically 1 for
        grayscale, 3 for RGB, 4 for RGBa)
    `glimpse_sizes`: a list of either `int`s or `float`s representing the
        (square) size of each glimpse to extract from the input image. The sizes
        should be in increasing order, but this will still make sure they're
        ordered correctly. The length of `glimpse_sizes` will also be the number
        of glimpse patches to extract. If `glimpse_sizes` is a list of `int`s,
        they'll be treated as exact sizes; if it's a list of `float`s, they'll
        be treated as percentages of `image_size`, and should thus all be <= 1.
        Padding will be added to the inputs to allow glimpses near the edges.
        The actual sizes of the resulting glimpses will be the same size as the
        first glimpse.
    `pad_imgs`: whether or not the input images need padding applied. Defaults
        to `True`, but set to `False` if the input images already have padding
        applied; make sure it's of the correct size!
    `learn_kernels`: if `True`, the kernels used to create the glimpses will
        have learnable parameters. If `False` (the default), the glimpses will
        be created using bicubic interpolation, and the kernels used will have
        fixed weights
    `a`: the scaling factor used for bicubic interpolation during resizing
    `preserve_out_channels`: if `True`, the number of out channels is kept in
        each of the glimpses. If `False` (default), the number of out channels
        in the glimpses is 1
    '''
    def __init__(self, image_size, in_channels,
                glimpse_sizes = [0.05, 0.1, 0.3, 0.6],
                pad_imgs = True, learn_kernels = False, a = -0.5,
                preserve_out_channels = False):
        super(GlimpseSensor, self).__init__()
        glimpse_sizes.sort()

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = in_channels if preserve_out_channels else 1
        self.num_glimpses = len(glimpse_sizes)
        if isinstance(glimpse_sizes[0], float):
            glimpse_sizes = [int(np.ceil(image_size * sz)) for sz in glimpse_sizes]
        self.glimpse_sizes = glimpse_sizes
        self.padding = int((glimpse_sizes[-1] - 1) // 2)
        self.learn_kernels = learn_kernels
        self.a = a
        self.pad_imgs = pad_imgs
        self.preserve_out_channels = preserve_out_channels
        self.unrolled_glimpse_size = glimpse_sizes[0]**2 * self.out_channels
        self.kernels = None
        self.init(a, in_channels, preserve_out_channels)

    '''
    Initializes a set of convolutional kernels to use during resizing. When
    `learn_kernels` is `True`, a set of 2D convolutional layers with trainable
    kernel weights and biases are used. When `learn_kernels` is `False`, a set
    of windowed filter kernels with fixed weights are used; these filters
    perform bicubic interpolation, so the kernel function is given as:

            | (a+2)|x|^3 - (a+3)|x|^2 + 1       ; |x| <= 1
    k(x) =  | a|x|^3 - 5a|x|^2 + 8a|x| - 4a     ; 1 < |x| < 2
            | 0                                 ; otherwise

    `a`: kernel coefficient used when calculating kernel weights
    `in_channels`: number of channels in the input images (typically 1 for
        grayscale, 3 for RGB, 4 for RGBa)
    `preserve_out_channels`: if `True`, the number of out channels is kept in
        each of the glimpses. If `False` (default), the number of out channels
        in the glimpses is 1
    '''
    def init(self, a = -0.5, in_channels = 3, preserve_out_channels = False):
        g_sz = self.glimpse_sizes[0]
        kernel_sizes = [
            sz - g_sz + 1 for sz in self.glimpse_sizes
        ]
        out_channels = in_channels if preserve_out_channels else 1
        if self.learn_kernels:
            # Initialize a 2D convolutional layer for resizing each glimpse
            # that will be extracted from the input images. Convolutional
            # kernel weights and biases will be trainable parameters
            self.kernels = nn.ModuleList([  # Convolution kernels
                nn.Conv2d(in_channels, out_channels, sz) for sz in kernel_sizes
            ])
        else:
            # Glimpses will be resized using a set of windowed filters with
            # fixed weights based on the bicubic interpolation function
            kernels = [     # x-values to apply the filter function over
                np.arange(-sz/2.+0.5, sz/2.+0.5)[..., np.newaxis] * 2./sz * (
                    2. if sz > 2. else 1.  # 2x2 filters get special treatment
                ) for sz in kernel_sizes
            ]
            # Kernel values in 1D
            kernels = [
                np.apply_along_axis(lambda x:
                    0. if x > 2. else (
                        a*x**3. - 5.*a*x**2. + 8.*a*x - 4.*a if x > 1. else
                        (a+2.)*x**3. - (a+3.)*x**2. + 1.
                    ), 1, np.abs(k)
                ) for k in kernels
            ]
            # Kernel values in 2D
            kernels = [np.matmul(k, k.T) for k in kernels]
            # Normalize and turn into torch tensors
            self.kernels = [
                Variable(
                    torch.from_numpy(
                            # Scale pixel amounts by the number of input and
                            # output channels so that the glimpses have pixel
                            # values in the same range as the original images
                            (k * out_channels)/(in_channels * k.sum())
                        ).expand(out_channels, in_channels, -1, -1)
                         .type(torch.FloatTensor),
                    requires_grad = False
                ) for k in kernels
            ]

    '''
    Inputs:
    `imgs`: input tensor of size `(batch_size, channels, image_height, image_width)`
    `locs`: a list of 2-tuples representing the `(x_pos, y_pos)` for each image
        in the batch to center the extracted glimpses around

    Outputs:
    `glimpses`: a tensor of size
        `(batch_size, out_channels*num_glimpses, glimpse_height, gilmpse_width)`
    '''
    def forward(self, imgs, locs):
        batch_size, channels, height, width = imgs.size()
        p = self.padding  # Convenience
        # Create a padded version of the images
        if self.pad_imgs:
            padded_imgs = Variable(torch.zeros(
                batch_size, channels, width + 2*p, height + 2*p
            ))
            padded_imgs[:, :, p:-p, p:-p] = imgs
        else:
            padded_imgs = imgs

        # Extract glimpses from the padded images
        unsized_glimpses = [
            torch.cat([
                padded_imgs[
                    b, :, p+x-g//2:p+x+(g+1)//2, p+y-g//2:p+y+(g+1)//2
                ].unsqueeze(0) for b, (x, y) in enumerate(locs)
            ], 0) for g in self.glimpse_sizes
        ]
        # Resize the glimpses and concatenate over the channels dimension
        glimpses = torch.cat(
            [self.resize(g, k) for g, k in zip(unsized_glimpses, self.kernels)], 1
        )
        return glimpses

    '''
    Performs 2D convolution on the input images. `kernel`s need to be either `None`
    or a tensor of size `(out_channels, in_channels, kernel_height, kernel_width)`
    '''
    def resize(self, img, kernel = None):
        # Convolute the kernel over the image and return the downsampled glimpse
        if self.learn_kernels:  # Kernel is actually a Conv2d layer
            resized_img = kernel(img)
        else:                   # Kernel is a tensor
            resized_img = F.conv2d(img, kernel)
        return resized_img

'''
Implements the Recurrent Attention Model (RAM), as described in "Recurrent
Models of Visual Attention" (Mnih, et al.), with some small changes to the
reinforcement learning algorithms used (namely having the network calculate the
advantage and state value separate before adding them together to get the Q-
values for the actions). This implementation also applies a 2D batch norm over
the channels dimension to the glimpses before passing them unrolled into the
GlimpseNetwork.
'''
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable

class RecurrentAttentionModel(nn.Module):
    '''
    The RecurrentAttentionModel is overall an RNN. The core network of the model
    takes a glimpse representation as input and combines it with the internal
    representation from the previous time step, `h_t1`, to produce the new
    internal state of the model `h_t`. The location and action networks use the 
    internal state representation `h_t` of the model to produce the next
    location to attend to (`loc_t`) and the action/classification (`a_t`). This
    basic RNN iteration is repeated for a variable number of steps.

    NOTE: this implementation differs slightly from the one in the paper by
    having the network estimate both the action advantages and state values
    before adding them together to get the Q-values over all of the actions.
    This implementation also applies a 2D batch norm over the channels dimension
    to the glimpses before passing them unrolled into the GlimpseNetwork.
    '''
    def __init__(self, image_size, in_channels, glimpse_h_size,
                 loc_h_size, glimpse_network_size, rnn_state_size,
                 action_state_size, rnn_type = 'LSTM', num_rnn_layers = 1,
                 nonlinearity = 'tanh', glimpse_sizes = [0.05, 0.1, 0.3, 0.6],
                 pad_imgs = True, learn_kernels = False, action_network = None,
                 location_network = None, continuous_location = True,
                 interpolation = PIL.Image.BICUBIC, dropout = 0.1):
        '''
            Inputs:  
        `image_size`: the size of the input image as an `int` (not including any
        padding, if that's already been applied). For now, assumes the image is
        square.  
        `in_channels`: number of channels in each input image  
        `glimpse_h_size`: size of the hidden layer for the independent glimpse
        linear layer representation  
        `loc_h_size`: size of the hidden layer for the independent location
        linear layer representation (total, for both dimensiones)  
        `glimpse_network_size`: size of the hidden layer for the combined
        glimpse/location linear layer representation  
        `rnn_state_size`: size of the RNN state vectors  
        `action_state_size`: size of action state vectors  
        `rnn_type`: the type of RNN cell to use. Should be one of: `'RNN'`,
        `'LSTM'` (default), or `'GRU'`  
        `num_rnn_layers`: number of recurrent layers to use in the recurrent
        network  
        `nonlinearity`: the nonlinear activation function to use when `rnn_type`
        is `'RNN'`; ignored when using an `'LSTM'` or `'GRU'`. Should be either
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
        `pad_imgs`: whether or not the input images need padding applied.  
        Defaults to `True`, but set to `False` if the input images already have
        padding applied; make sure it's of the correct size!  
        `learn_kernels`: if `True`, the kernels used to create the glimpses will
        have learnable parameters. If `False` (the default), the glimpses will
        be created by resizing the crops from the image using the specified
        interpolation type  
        `action_network`: a PyTorch module for this RAM module to use to select
        the next action. Should take as input a tensor of size `(batch_size,
        rnn_state_size)` and return a tensor of size `(batch_size, action_state_size)`
        representing predicted Q-values for each action. If `None` (default),
        then a dueling Q-network is created and used  
        `location_network`: a PyTorch module for this RAM module to use to
        select the next location to center glimpses around. Should take as input
        a tensor of size `(batch_size, rnn_state_size)` and return a 2-tuple of
        tensors of size `(batch_size, image_size)` in a tuple, each representing
        predicted Q-values for the location along each axis. If `None` (default),
        then two dueling Q-networks are created and used  
        `continuous_location`: if `True`, the input locations should be `float`s
        in the range of `(0, 1)` representing how far along each axis the next
        glimpse will be taken; the actual pixel locations will be found using
        `np.round(loc*image_size)`. If `False`, the input locations should be
        `int`s in the range of `(0, image_size-1)` representing the exact pixel
        locations the next glimpse will be taken; the values will be used as-is  
        `interpolation`: desired interpolation type, specified as an `int` (aka
        one of those constants from the PIL.Image package). Defaults to 3,
        aka `PIL.Image.BICUBIC`. Ignored if `learn_kernels == True`  
        `dropout`: dropout drop probability  

        NOTE: The paper has the location network output two real-valued x- and
        y- coordinates in the range of (-1, 1)m with (0, 0) being the center of
        the image and (-1, -1) being the top left corner, rather than discrete
        coordinates across the dimensions of each side of the image. They found
        it difficult for the agent to learn policies with discrete coordinates
        over more than 25 possible locations
        '''
        super(RecurrentAttentionModel, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_glimpses = len(glimpse_sizes)
        self.glimpse_h_size = glimpse_h_size
        self.loc_h_size = loc_h_size
        self.glimpse_network_size = glimpse_network_size
        self.rnn_state_size = rnn_state_size
        self.action_state_size = action_state_size
        self.continuous_location = continuous_location
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
            pad_imgs = pad_imgs, learn_kernels = learn_kernels, dropout = dropout,
            interpolation = interpolation, continuous_location = continuous_location
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
            action_network = DuelingQNetwork(rnn_state_size, action_state_size)
        self.action_network = action_network
        # Location network
        if location_network is None:
            class LocationNetwork(nn.Module):
                def __init__(self, in_size, img_size, continuous):
                    super(LocationNetwork, self).__init__()
                    self.in_size = in_size
                    self.img_size = img_size
                    self.continuous = continuous
                    if continuous:
                        self.loc_layer = nn.Sequential(
                            nn.Linear(in_size, 2),
                            nn.Sigmoid()
                        )
                    else:
                        self.x_loc = DuelingQNetwork(in_size, img_size)
                        self.y_loc = DuelingQNetwork(in_size, img_size)

                def forward(self, h):
                    if self.continuous:
                        loc = self.loc_layer(h)
                    else:
                        x = self.x_loc(h)
                        y = self.y_loc(h)
                        loc = (x, y)
                    return loc
            # Initialize a LocationNetwork
            location_network = LocationNetwork(
                rnn_state_size, image_size, continuous_location
            )
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
            rnn_states = [(
                Variable(torch.zeros(batch_size, self.rnn_state_size)),
                Variable(torch.zeros(batch_size, self.rnn_state_size))
            ) for _ in range(self.num_rnn_layers)]
        else:
            rnn_states = [
                Variable(torch.zeros(batch_size, self.rnn_state_size))
                for _ in range(self.num_rnn_layers)
            ]
        return rnn_states

    def forward(self, imgs, locs, rnn_states):
        '''
            Inputs:  
        `imgs`: input tensor of size `(batch_size, channels, image_height,
        image_width)`  
        `locs`: if `continuous_location` is `True`, this should be a tensor of
        size `(batch_size, 2)` with each value in the range of `(0, 1)`
        representing how far along each axis the next glimpse will be taken;
        these values will be converted into a list of 2-tuples to be used with
        the GlimpseSensor. If `continuous_location` is `False`, the this should
        be a list of 2-tuples representing the `(x_pos, y_pos)` for each image
        in the batch to center the extracted glimpses around  

            Outputs:  
        `new_rnn_states`: the new RNN hidden states for each RNN cell in the
        stack. The last state vector in this list (or the first state vector in
        the last tuple in this list, if the RNN cells are LSTMs) is the actual
        state representation used to get the next actions and locations  
        `actions`: a tensor of size `(batch_size, action_state_size)`
        representing predicted Q-values for each possible action  
        `locs`: if `continuous_location` is `True`, this will be a tensor of
        size `(batch_size, 2)` with values just like what was passed in as an
        input. If `continuous_location` is `False`, this will be a 2-tuple of
        tensors of size `(batch_size, image_size)`, each representing predicted
        Q-values for the x- and y-coordinates of the image, respectively, where
        the next glimpse should be centered around. Note that this output should
        be argmax'd and converted to a list of 2-tuples before being passed back
        into the network.  
        '''
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
        cur_state = rnn_in
        # Action and Location networks
        actions = self.action_network(cur_state)
        locs = self.location_network(cur_state)

        return new_rnn_states, actions, locs


class GlimpseNetwork(nn.Module):
    '''
    Given the location `loc_t1` and input image `x_t`, the GlimpseNetwork uses
    the GlimpseSensor to extract retina representation `rho(x_t, loc_t1)`. The
    retina representation and glimpse location are then mapped into a hidden
    space using independent linear layers, followed by a ReLU activation and
    another linear layer that combines the information from both components. The
    GlimpseNetwork defines a trainable bandwidth-limited sensor for the
    attention network producing the glimpse representation.
    '''
    
    def __init__(self, image_size, in_channels, glimpse_h_size, loc_h_size,
                 network_size, pad_imgs = True, learn_kernels = False,
                 interpolation = PIL.Image.BICUBIC,
                 glimpse_sizes = [0.05, 0.1, 0.3, 0.6],
                 continuous_location = True, dropout = 0.1):
        '''
            Inputs:  
        `image_size`: the size of the input image as an `int` (not including any
        padding, if that's already been applied). For now, assumes the image is
        square.  
        `in_channels`: number of channels in each input image  
        `glimpse_h_size`: size of the hidden layer for the independent glimpse
        linear layer representation  
        `loc_h_size`: size of the hidden layer for the independent location
        linear layer representation (total, for both dimensiones)  
        `network_size`: size of the hidden layer for the combined glimpse/location
        linear layer representation  
        `pad_imgs`: whether or not the input images need padding applied.  
        Defaults to `True`, but set to `False` if the input images already have
        padding applied; make sure it's of the correct size!  
        `learn_kernels`: if `True`, the kernels used to create the glimpses will
        have learnable parameters. If `False` (the default), the glimpses will
        be created by resizing the crops from the image using the specified
        interpolation type  
        `interpolation`: desired interpolation type, specified as an `int` (aka
        one of those constants from the PIL.Image package). Defaults to 3,
        aka `PIL.Image.BICUBIC`. Ignored if `learn_kernels == True`  
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
        `continuous_location`: if `True`, the input locations should be `float`s
        in the range of `(0, 1)` representing how far along each axis the next
        glimpse will be taken; the actual pixel locations will be found using
        `np.round(loc*image_size)`. If `False`, the input locations should be
        `int`s in the range of `(0, image_size-1)` representing the exact pixel
        locations the next glimpse will be taken; the values will be used as-is  
        `dropout`: dropout drop probability  
        '''
        super(GlimpseNetwork, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_glimpses = len(glimpse_sizes)
        self.glimpse_h_size = glimpse_h_size
        self.loc_h_size = loc_h_size
        self.network_size = network_size
        self.continuous_location = continuous_location
        # Basic layers
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # Initialize the GlimpseSensor
        self.glimpse_sensor = GlimpseSensor(
            image_size = image_size, in_channels = in_channels,
            learn_kernels = learn_kernels, pad_imgs = pad_imgs,
            interpolation = interpolation, glimpse_sizes = glimpse_sizes
        )

        # 2D Batch norm
        self.batchnorm2d = nn.BatchNorm2d(self.in_channels*self.num_glimpses)

        # Initialize the network components
        if continuous_location:
            self.loc_layer = nn.Linear(2, loc_h_size)
        else:
            self.loc_embeds = nn.ModuleList([
                nn.Embedding(image_size, loc_h_size//2),
                nn.Embedding(image_size, loc_h_size//2)
            ])
        glimpse_size = self.glimpse_sensor.glimpse_sizes[0]
        unrolled_glimpse_size = glimpse_size**2 * in_channels * self.num_glimpses
        self.glimpse_layer = nn.Linear(
            unrolled_glimpse_size, glimpse_h_size
        )
        # NOTE: the paper just concatenated the glimpse and location
        # representations and passed that through a ReLU activation to get the
        # output of the GlimpseNetwork rather than passing the concatenation
        # through another linear layer first
        self.out_layer = nn.Linear(glimpse_h_size + loc_h_size, network_size)

        # Initialize weights
        self.init()

    def init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
            else:
                p.data.fill_(0)

    def forward(self, imgs, locs):
        '''
            Inputs:  
        `imgs`: input tensor of size `(batch_size, channels, image_height,
        image_width)`  
        `locs`: if `continuous_location` is `True`, this should be a tensor of
        size `(batch_size, 2)` with each value in the range of `(0, 1)`
        representing how far along each axis the next glimpse will be taken;
        these values will be converted into a list of 2-tuples to be used with
        the GlimpseSensor. If `continuous_location` is `False`, the this should
        be a list of 2-tuples representing the `(x_pos, y_pos)` for each image
        in the batch to center the extracted glimpses around  

            Outputs:  
        `net_out`: a tensor of size `(batch_size, network_size)`  
        '''
        nbatches = imgs.size(0)

        ### Location representation
        if self.continuous_location:
            # Pass the (nbatches, 2) tensor into the linear layer
            loc_rep = self.relu(self.drop(self.loc_layer(locs)))                # (nbatches, loc_h_size)
            # Convert locs into a list of 2-tuples for the GlimpseSensor
            locs = [
                (
                    int(np.round(locs.data[i, 0]*self.image_size)),
                    int(np.round(locs.data[i, 1]*self.image_size))
                ) for i in range(locs.size(0))  # Iterate over batches
            ]
        else:
            # Embed the glimpse locations
            x_loc = Variable(                                                   # (nbatches, 1)
                torch.LongTensor([loc[0] for loc in locs]).unsqueeze(1),
            )
            y_loc = Variable(                                                   # (nbatches, 1)
                torch.LongTensor([loc[1] for loc in locs]).unsqueeze(1),
            )
            x_embeds = self.relu(self.drop(self.loc_embeds[0](x_loc)))          # (nbatches, loc_h_size/2)
            y_embeds = self.relu(self.drop(self.loc_embeds[1](y_loc)))          # (nbatches, loc_h_size/2)
            # Concatenate
            loc_rep = torch.cat([x_embeds, y_embeds], -1)                       # (nbatches, loc_h_size)

        ### Glimpse representation
        # Get glimpse representations for the images
        glimpses = self.glimpse_sensor(imgs, locs)                              # (nbatches, channels, glimpse_height, glimpse_width)
        # Apply batch norm over the channels, then apply a ReLU activation
        normed_glimpses = self.relu(self.batchnorm2d(glimpses))                 # (nbatches, channels, glimpse_height, glimpse_width)

        # Unroll the glimpses and pass them through the linear layer
        glimpse_rep = self.relu(self.drop(self.glimpse_layer(
            normed_glimpses.view(nbatches, -1)                                  # (nbatches, unrolled_glimpse_size)
        )))                                                                     # (nbatches, glimpse_h_size)

        ### Full network output
        # Concatenate the glimpse and location representations
        glimpse_net_in = torch.cat([glimpse_rep, loc_rep], -1)                  # (nbatches, glimpse_h_size + loc_h_size)
        # Pass through final linear layer
        net_out = self.relu(self.drop(self.out_layer(glimpse_net_in)))          # (nbatches, network_size)
        return net_out
        

class GlimpseSensor(nn.Module):
    '''
    Given the coordinates of the glimpse and an input image, the GlimpseSensor
    extracts a retina-like representation `rho(x_t, loc_t1)`, centered at
    `loc_t1`, that contains multiple resolution patches.
    '''
    def __init__(self, image_size, in_channels,
                 pad_imgs = True, learn_kernels = False,
                 interpolation = PIL.Image.BICUBIC,
                 glimpse_sizes = [0.05, 0.1, 0.3, 0.6]):
        '''
            Inputs:  
        `image_size`: the size of the input image as an `int` (not including any
        padding, if that's already been applied). For now, assumes the image is
        square.  
        `in_channels`: number of channels in each input image  
        `pad_imgs`: whether or not the input images need padding applied.  
        Defaults to `True`, but set to `False` if the input images already have
        padding applied; make sure it's of the correct size!  
        `learn_kernels`: if `True`, the kernels used to create the glimpses will
        have learnable parameters. If `False` (the default), the glimpses will
        be created by resizing the crops from the image using the specified
        interpolation type  
        `interpolation`: desired interpolation type, specified as an `int` (aka
        one of those constants from the PIL.Image package). Defaults to 3,
        aka `PIL.Image.BICUBIC`. Ignored if `learn_kernels == True`  
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
        '''
        super(GlimpseSensor, self).__init__()
        glimpse_sizes.sort()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.pad_imgs = pad_imgs
        self.learn_kernels = learn_kernels
        self.interpolation = interpolation
        if isinstance(glimpse_sizes[0], float):
            glimpse_sizes = [int(np.ceil(image_size * sz)) for sz in glimpse_sizes]
        self.glimpse_sizes = glimpse_sizes
        self.num_glimpses = len(glimpse_sizes)
        self.padding = int((glimpse_sizes[-1] - 1) // 2)

        # If we're using learnable kernels, then we need to initialize the
        # 2D convolutional layers we'll be using to extract the glimpses
        if self.learn_kernels:
            glimpse_size = glimpse_sizes[0]
            kernel_sizes = [
                sz - glimpse_size + 1 for sz in glimpse_sizes
            ]
            self.kernels = nn.ModuleList([
                nn.Conv2d(in_channels, in_channels, sz) for sz in kernel_sizes
            ])
        # If we aren't using learnable kernels, then we don't need to do
        # anything because the PIL/torchvision packages will do everything

    def forward(self, imgs, locs):
        '''
            Inputs:  
        `imgs`: input tensor of size `(batch_size, channels, image_height,
        image_width)`  
        `locs`: a list of 2-tuples representing the `(x_pos, y_pos)` for each
        image in the batch to center the extracted glimpses around  

            Outputs:  
        `glimpses`: a tensor of size `(batch_size, out_channels*num_glimpses,
        glimpse_height, gilmpse_width)`  
        '''
        batch_size, channels, height, width = imgs.size()
        # Convenience
        p = self.padding
        gs = self.glimpse_sizes[0]

        if self.pad_imgs:
            # Pad the input images
            padded_imgs = Variable(torch.zeros(
                batch_size, channels, width + 2*p, height + 2*p
            ))
            padded_imgs[:, :, p:-p, p:-p] = imgs
        else:
            # Images already padded
            padded_imgs = imgs

        # Create the resized glimpses and return them as a single tensor
        if self.learn_kernels:
            # Extract the glimpsed region from each padded input image
            unsized_glimpses = [
                torch.cat([
                    padded_imgs[
                        b, :, p+x-g//2:p+x+(g+1)//2, p+y-g//2:p+y+(g+1)//2
                    ].unsqueeze(0) for b, (x, y) in enumerate(locs)
                ], 0) for g in self.glimpse_sizes
            ]
            # Use the 2D convolutional kernels to resize them, then concatenate
            # each of the resized glimpses together along the channels dimension
            glimpses = torch.cat([
                k(g) for g, k in zip(unsized_glimpses, self.kernels)
            ], 1)
        else:
            # Torchvision transforms composer that converts each image
            # (initially a Tensor) to a PIL Image and resizes it, then converts
            # it back to a Tensor
            composer = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(gs, self.interpolation),
                transforms.ToTensor(),
            ])
            # Extract the glimpsed region from each padded input image
            unsized_glimpses = [
                [
                    padded_imgs[
                        b, :, p+x-g//2:p+x+(g+1)//2, p+y-g//2:p+y+(g+1)//2
                    ] for b, (x, y) in enumerate(locs)
                ] for g in self.glimpse_sizes
            ]
            # Use the Torchvision transforms composer to resize each unsized
            # glimpse, then concatenate each resized glimpse of the same
            # resolution together along the batch dimension, then concatenate
            # all of the resized across all resolutions together along the
            # channels dimension
            glimpses = torch.cat([
                # Have to work over each item in the batch one at a time
                torch.cat([
                    composer(i).unsqueeze(0) for i in glist
                ], 0) for glist in unsized_glimpses
            ], 1)
        return Variable(glimpses)  # Wrap in a Variable?


class DuelingQNetwork(nn.Module):
    '''
    A simple dueling Q-Network consisting of a single hidden layer that splits
    the input in half and uses them to estimate the action advantages and
    state values independently, then adds them together to get the Q-values
    over the entire action space (of size `out_size`).
    '''
    def __init__(self, in_size, num_actions):
        super(DuelingQNetwork, self).__init__()
        self.in_size = in_size
        self.out_size = num_actions
        self.advantage_layer = nn.Linear(in_size//2, num_actions)
        self.value_layer = nn.Linear(in_size//2, 1)

    def forward(self, state):
        # Split the input in half along the last dimension
        adv_stream, val_stream = state.chunk(chunks = 2, dim = -1)
        # Pass each through their respective layers
        adv_out = self.advantage_layer(adv_stream)
        val_out = self.value_layer(val_stream)
        # Add the estimated values to the de-meaned estimated advantages
        adv_mean = adv_out.mean(dim = -1, keepdim = True)
        q_out = val_out + (adv_out - adv_mean)
        return q_out

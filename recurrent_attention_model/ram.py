'''
Implements the Recurrent Attention Model (RAM), as described in "Recurrent
Models of Visual Attention" (Mnih, et al.).

Features to try:
    - Smoothed action network output (pass the output of the action network
    through a sigmoid function before using to reduce the effect of large
    logits) => DONE
    - Different recurrent networks as the core network (base is a single LSTM)
            => IMPLEMENTED; need to test
    - Different glimpse representation subnetworks within the GlimpseNetwork,
    especially ones using convolutions
            => IMPLEMENTED; need to test
'''
import copy
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
    '''
    def __init__(self, image_size, in_channels, glimpse_g_size, loc_g_size,
                glimpse_net_size, rnn_state_size, num_actions, sensor_size,
                glimpse_sizes, conv_glimpses = False, out_channels = None,
                glimpse_rep_net = None, rnn_type = 'LSTM', double_core = False,
                nonlinearity = 'relu', core_rnn_net = None, action_net = None,
                loc_net = None, smooth_predictions = False, pad_imgs = True,
                interpolation = PIL.Image.BICUBIC, g_rep_op = 'add',
                padding = None, dropout = 0.1, convoluted_glimpses = False,
                input_action_to_loc_net = False, recurrent_action_net = False,
                recurrent_loc_net = False, context_net = None):
        '''
        Inputs
        ---
        `image_size`: the size of the input image as an `int` (not including any
        padding, if that's already been applied). Assumes input images are square.

        `in_channels`: number of channels in each input image.

        `glimpse_g_size`: size of the output of the glimpse representaion. If
        `g_rep_op` is `'mul'`, this is set to be equal to `glimpse_net_size`.

        `loc_g_size`: size of the output of the location representation (total,
        with both dimensions as the input). If `g_rep_op` is `'mul'`, this is
        set to be equal to `glimpse_net_size`.

        `glimpse_net_size`: size of the output of the GlimpseNetwork.

        `rnn_state_size`: size of the RNN state vectors.

        `num_actions`: size of the action state space.

        `sensor_size`: the bandwidth/resolution of the Glimpse sensor, i.e. what
        size each of the glimpses will be resized to, as an `int`. Usually,
        `sensor_size >= 2`.
        
        `glimpse_sizes`: a list of `int`s representing the (square) size of each
        glimpse to extract from the input images. The length of `glimpse_sizes`
        will also be the number of glimpse patches to extract. The actual sizes
        of the resulting glimpse will be scaled to `sensor_size` pixels.

        `conv_glimpses`: if `False` (default), a single Linear layer will be
        used to get the glimpse representation. If `True`, the captured glimpses
        will be passed into a classical 2-layer convolutional network (with each
        layer using a 3x3 filter, and with the output of the second layer passed
        through a 2x2 max-pooling layer, then flattened) first. This is ignored
        if `glimpse_rep_net` is not `None`.

        `out_channels`: if `convoluted_glimpses` is `True`, this should be an
        integer indicating the number of output channels after each convolutional
        layer while extracting and producing the glimpses. If `out_channels` is
        left as `None`, it'll default to the same value as `in_channels`.

        `glimpse_rep_net`: if `None` (default), the glimpse representation
        network will be initialized according to the value of `conv_glimpses`.
        If a PyTorch nn.Module is passed in instead, it will be used as the
        glimpse representation network. This network should take inputs of size
        `(batch_size, channels*num_glimpses, glimpse_height, gilmpse_width)` and
        return outputs of size `(batch_size, glimpse_g_size)`.

        `rnn_type`: the type of RNN to use for all recurrent networks that need
        to be initialized. Should be one of `'RNN'`, `'LSTM'` (default), or
        `'GRU'`. Ignored if `core_rnn_net` is not `None`, `action_net` is not
        `None`, `action_net` is `None` but `recurrent_action_net` is `False`,
        `loc_net` is not `None`, or `loc_net` is `None` but `recurrent_loc_net`
        is `False`.

        `double_core`: if `True`, the core recurrent network is duplicated so
        that the first takes in the output of the glimpse network as its input
        and passes its output to the action network, while the second takes in
        the output of the first core recurrent network as its input and passes
        its output to the location network. If this is `True`, `glimpse_net_size`
        and `rnn_state_size` should be equal.

        `num_rnn_layers`: number of recurrent layers to use in the core recurrent
        network, each having the same size hidden states. Ignored if
        `core_rnn_net` is not `None`.

        `nonlinearity`: the nonlinear activation function to use when `rnn_type`
        is `RNN`. Should be either `'tanh'` or `'relu'`. Ignored if `rnn_type`
        is either `'LSTM'` or `'GRU'`, or if `core_rnn_net` is not `None`.

        `core_rnn_net`: if `None` (default), the core recurrent network will be
        initialized according to the values of `rnn_type` and `nonlinearity`. If
        a PyTorch nn.Module is passed in instead, it will be used as the core
        recurrent network. This network should take a single-step input of size
        `(batch_size, glimpse_net_size)` and its hidden states, then return a
        single-step output of the new hidden state(s); if there is only one
        hidden state, it'll also be used as the output and should be of size
        `(batch_size, rnn_state_size)`, otherwise the first hidden state will be
        used as the output. The network should also implement a method with the
        signature `init_rnn_states(batch_size)` for initializing hidden states.

        `action_net`: if `None` (default), the action network will be
        initialized as a Linear layer. If a PyTorch nn.Module is passed in
        instead, it will be used as the action network. This network should take
        inputs of size `(batch_size, rnn_state_size)` and return outputs of size
        `(batch_size, num_actions)`. If `recurrent_action_net` is `True`, then
        this should be a recurrent Module (or `None`) that implements an
        `init_rnn_states(batch_size)` method.

        `loc_net`: if `None` (default), the location network will be initialized
        as a Linear layer. If a PyTorch nn.Module is passed in instead, it will
        be used as the location network. This network should take inputs of size
        `(batch_size, rnn_state_size)` and return outputs of size `(batch_size,
        2)`. If `recurrent_loc_net` is `True`, then this should be a recurrent
        Module (or `None`) that implements an `init_rnn_states(batch_size)`
        method. If `input_action_to_loc_net` is `True`, then the output of the
        action network is concatenated to the other inputs into the location
        network each step, so that should be taken into account.

        `smooth_predictions`: when `True`, applies a sigmoid activation to the
        output of the action network, reducing the effect of very large values.

        `interpolation`: desired interpolation type, specified as an `int` (aka
        one of those constants from the PIL.Image package). Defaults to 3,
        aka `PIL.Image.BICUBIC`.

        `g_rep_op`: a string representing how the hidden location and glimpse
        representations within the GlimpseNetwork should be combined together.
        Choices: `'cat'` (default) to have them concatenated and passed through
        a Linear layer with ReLU activation, `'add'` to have them passed through
        separate Linear layers with ReLU activations before being elementwise
        added, or `'mul'` to have them elementwise multiplied (also sets
        `loc_g_size` and `glimpse_g_size` to be equal to `glimpse_net_size`).

        `pad_imgs`: whether or not the input images need padding applied.
        Defaults to `True`, but set to `False` if the input images already have
        padding applied; make sure it's of the correct size!

        `padding`: if `None`, the amount of padding is calculated, otherwise an
        `int` should be passed in.

        `dropout`: dropout drop probability.

        `convoluted_glimpses`: if `True`, the GlimpseSensor uses a convolutional
        network with learnable parameters to extract and resize the glimpses.
        Defaults to `False`.

        `input_action_to_loc_net`: if `True`, the output of the action network
        is concatenated to the other inputs into the location network each step.
        Defaults to `False`.

        `recurrent_action_net`: if `True`, indicates that the action network is
        (or should be, if we're to initialize one) recurrent. If an action
        network is given (i.e. if `action_net is not None`), then it should also
        implement an `init_rnn_states(batch_size)` method.

        `recurrent_loc_net`: if `True`, indicates that the location network is
        (or should be, if we're to initialize one) recurrent. If a location
        network is given (i.e. if `loc_net is not None`), then it should also
        implement an `init_rnn_states(batch_size)` method.
        '''
        super(RecurrentAttentionModel, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_glimpses = len(glimpse_sizes)
        self.glimpse_g_size = glimpse_g_size
        self.loc_g_size = loc_g_size
        self.glimpse_net_size = glimpse_net_size
        self.rnn_state_size = rnn_state_size
        if double_core:
            assert glimpse_net_size == rnn_state_size
        self.double_core = double_core
        self.num_actions = num_actions
        self.sensor_size = sensor_size
        self.smooth_predictions = smooth_predictions
        self.input_action_to_loc_net = input_action_to_loc_net
        self.recurrent_action_net = recurrent_action_net
        self.recurrent_loc_net = recurrent_loc_net
        self.context_net = context_net

        ### Layer initializations ###
        self.drop = nn.Dropout(dropout)

        ## GlimpseNetwork
        self.glimpse_net = GlimpseNetwork(
            image_size = image_size, in_channels = in_channels, padding = padding,
            glimpse_g_size = glimpse_g_size, loc_g_size = loc_g_size,
            glimpse_net_size = glimpse_net_size, g_rep_op = g_rep_op,
            sensor_size = sensor_size, glimpse_sizes = glimpse_sizes,
            conv_glimpses = conv_glimpses, glimpse_rep_net = glimpse_rep_net,
            pad_imgs = pad_imgs, convoluted_glimpses = convoluted_glimpses,
            out_channels = out_channels, interpolation = interpolation
        )

        # Helper function
        def get_rnn_cell(rnn_type, nonlinearity, input_size, hidden_size):
            rnn_type = rnn_type.upper()
            if rnn_type not in ['RNN', 'LSTM', 'GRU']:
                raise ValueError(
                    """Incorrect RNN type (expected one of 'RNN', 'LSTM', or 'GRU',
                    but found '%s')""" % rnn_type
                )
            if rnn_type == 'RNN':
                nonlinearity = nonlinearity.lower()
                if nonlinearity not in ['tanh', 'relu']:
                    raise ValueError(
                        """Incorrect RNN nonlinearity function (expected either
                        'tanh' or 'relu', but found '%s')""" % nonlinearity
                    )
                else:
                    rnn_type = '_'.join([rnn_type, nonlinearity])
                core_rnn_net = nn.RNNCell(
                    input_size = input_size, hidden_size = hidden_size,
                    nonlinearity = nonlinearity
                )
            else:
                rnn_cell = getattr(nn, rnn_type + 'Cell')
                core_rnn_net = rnn_cell(
                    input_size = input_size, hidden_size = hidden_size
                )
            return core_rnn_net, rnn_type

        # Keep a dict of our dropout masks to apply to our recurrent networks
        # Using the same dropout mask on each different input/output of the
        # network for the entire sequence implements variational dropout and
        # helps regularize the recurrent networks
        dp_masks = ['input_core', 'hidden_core', 'output_core']

        ## Core recurrent network
        if core_rnn_net is None:
            core_rnn_net, self.core_rnn_type = get_rnn_cell(
                rnn_type, nonlinearity, glimpse_net_size, rnn_state_size
            )
        else:
            assert hasattr(core_rnn_net, 'init_rnn_states')
            self.core_rnn_type = 'CUSTOM'
        if double_core:
            self.core_rnn_net = nn.ModuleList([
                core_rnn_net,                # Action
                copy.deepcopy(core_rnn_net)  # Location
            ])
            dp_masks.extend(['input_core2', 'hidden_core2', 'output_core2'])
        else:
            self.core_rnn_net = core_rnn_net

        ## Action network
        if recurrent_action_net:
            if action_net is None:
                action_net, self.action_rnn_type = get_rnn_cell(
                    rnn_type, nonlinearity, rnn_state_size, rnn_state_size
                )
                action_net = nn.ModuleList([
                    action_net, nn.Linear(rnn_state_size, num_actions)
                ])
            else:
                assert hasattr(action_net, 'init_rnn_states')
            dp_masks.append('hidden_act')
        elif action_net is None:
            action_net = nn.Linear(rnn_state_size, num_actions)
        self.action_net = action_net
        if smooth_predictions:
            self.sigmoid = nn.Sigmoid()

        ## Location network
        loc_net_in_size = rnn_state_size + (num_actions if input_action_to_loc_net else 0)
        if recurrent_loc_net:
            if loc_net is None:
                loc_net, self.loc_rnn_type = get_rnn_cell(
                    rnn_type, nonlinearity, loc_net_in_size, rnn_state_size
                )
                loc_net = nn.ModuleList([
                    loc_net, nn.Linear(rnn_state_size, 2)
                ])
            else:
                assert hasattr(loc_net, 'init_rnn_states')
            dp_masks.append('hidden_loc')
        elif loc_net is None:
            loc_net = nn.Linear(loc_net_in_size, 2)
        self.loc_net = loc_net

        ## Baseline reward
        self.baseline = nn.Linear(rnn_state_size, 1)

        ### Initialize parameters and dropout masks ###
        # This is for training the context net with a separate set of hyperparameters
        self.base_net = nn.ModuleList([
            self.glimpse_net, self.core_rnn_net, self.loc_net, self.action_net, self.baseline
        ])
        self.rnn_dp_masks = {name: None for name in dp_masks}
        self.init()

    def get_context_loc(self, imgs, batch_size):
        '''
        Inputs
        ---
        `imgs`: input tensor of size `(batch_size, channels, image_height,
        image_width)`.


        Outputs
        ---
        `context`: a FloatTensor of size `(batch_size, core_rnn_state_size)` to
        be used as the initial hidden state of the core RNN network in the
        `RecurrentAttentionModel`.
        
        `loc`: a FloatTensor of size `(batch_size, core_rnn_state_size)` to
        be used as the initial hidden state of the core RNN network in the
        `RecurrentAttentionModel`.
        '''
        context = self.context_net(imgs)
        loc = self.loc_net(context).unsqueeze(0)
        states = self.init_rnn_states(batch_size, context)
        return states, loc

    def init(self):
        self.glimpse_net.init()

        rec_layers = [self.core_rnn_net]
        lin_layers = [self.baseline]
        if self.recurrent_action_net:
            rec_layers.append(self.action_net)
        else:
            lin_layers.append(self.action_net)
        if self.recurrent_loc_net:
            rec_layers.append(self.loc_net)
        else:
            lin_layers.append(self.loc_net)

        for layer in rec_layers:
            if hasattr(layer, 'init'):
                layer.init()
            else:
                for p in layer.parameters():
                    if p.dim() > 1:
                        nn.init.orthogonal(p)
                    else:
                        p.data.fill_(0)

        for layer in lin_layers:
            if hasattr(layer, 'init'):
                layer.init()
            else:
                for p in layer.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_normal(p)
                    else:
                        p.data.fill_(0)

    def reset_dp_masks(self):
        self.rnn_dp_masks = {name: None for name in self.rnn_dp_masks.keys()}

    def init_rnn_states(self, batch_size, init_state = None):
        state_size = self.rnn_state_size

        def state_init(rnn_type, state_size):
            if rnn_type == 'CUSTOM':
                rnn_states = self.core_rnn_net.init_rnn_states(batch_size)
            elif rnn_type == 'LSTM':
                rnn_states = (
                    Variable(torch.zeros(batch_size, state_size)),
                    Variable(torch.zeros(batch_size, state_size))
                )
            else:
                rnn_states = Variable(torch.zeros(batch_size, state_size))
            return rnn_states

        # Core network states
        core_states = state_init(self.core_rnn_type, state_size)
        if init_state is not None:
            if isinstance(core_states, tuple):
                core_states = list(core_states)
                core_states[0] = init_state
                core_states = tuple(core_states)
            else:
                core_states = init_state
        # Need to duplicate the core network if we're using a double core
        if self.double_core:
            core_states = (
                state_init(self.core_rnn_type, state_size), core_states
            )
        
        state_package = [core_states]

        if self.recurrent_action_net:
            if hasattr(self.action_net, 'init_rnn_states'):
                action_states = self.action_net.init_rnn_states(batch_size)
            else:
                action_states = state_init(self.action_rnn_type, state_size)
            state_package.append(action_states)

        if self.recurrent_loc_net:
            if hasattr(self.loc_net, 'init_rnn_states'):
                loc_states = self.loc_net.init_rnn_states(batch_size)
            else:
                loc_states = state_init(self.loc_rnn_type, state_size)
            state_package.append(loc_states)

        return tuple(state_package)

    def forward(self, locs, state_package, imgs = None, glimpses = None):
        '''
        Inputs
        ---
        `locs`: a FloatTensor of size `(seq_len, batch_size, 2)`, each row
        representing the x- and y-coordinates of the location to center the
        glimpses around at each step, with values in the range of [-1, 1];
        location (-1, -1) is the top left corner of the image, while (0, 0) is
        the center.

        `state_package`: the initial RNN states to use. If `double_core` is `True`,
        this should be a 2-tuple holding the states for each core network. If
        `recurrent_action_net` and/or `recurrent_loc_net` are `True`, then the
        states for those two networks should be included (in that order), too.

        `imgs`: input tensor of size `(batch_size, channels, image_height,
        image_width)`. Can be `None` if `glimpses` is not `None`, and ignored if
        it isn't.

        `glimpses`: instead of passing in `imgs`, you can pass the glimpses
        directly in as a tensor of size `(batch_size, channels*num_glimpses,
        glimpse_height, gilmpse_width)`. If this is `None`, then `imgs` can't be
        `None`.


        Outputs
        ---
        `actions_seq`: a tensor of size `(seq_len, batch_size, num_actions)`
        
        `locs_seq`: a FloatTensor of size `(seq_len, batch_size, 2)` with values
        just like what was passed in as an input.

        `rnn_outputs`: the sequence of outputs from the core recurrent network
        as a tensor of size `(seq_len, batch_size, rnn_state_size)`. If
        `double_core` is `True`, then the outputs are from the second core
        recurrent network.

        `state_package`: the new RNN hidden states for each of the recurrent
        network(s) after the very last step in the sequence.

        `glimpses_seq`: the glimpses used this step. If `glimpses` was passed in as
        an input, it'll just be returned back.

        `baseline_rewards`: the baseline reward for each step as a tensor of
        size `(seq_len, batch_size, 1)`.
        '''
        assert imgs is not None or glimpses is not None
        seq_len = locs.size(0)
        # Unpack the hidden states
        state_package = list(state_package)
        if self.recurrent_loc_net:
            loc_states = state_package[-1]
            state_package = state_package[:-1]
        if self.recurrent_action_net:
            action_states = state_package[-1]
            state_package = state_package[:-1]
        core_states = state_package[0]
        state_package = []

        actions_seq = []
        locs_seq = []
        glimpses_seq =[]
        rnn_outputs = []
        rnn_dp_masks = self.rnn_dp_masks

        # Helper functions for dropout:
        def get_dp_mask(vars):
            if isinstance(vars, tuple):
                return tuple(get_dp_mask(var) for var in vars)
            else:
                return self.drop(torch.ones_like(vars))

        def apply_state_masks(states, masks):
            if isinstance(states, tuple):
                return tuple(
                    apply_state_masks(s, m) for s, m in zip(states, masks)
                )
            else:
                return states.mul(masks)

        for t in range(seq_len):
            l_t = locs[t]
            g_t = None if glimpses is None else glimpses[t]

            # Glimpse network
            rnn_in, g_t = self.glimpse_net(l_t, imgs, g_t)

            core_rnn = self.core_rnn_net[0] if self.double_core else self.core_rnn_net
            states = core_states[0] if self.double_core else core_states

            if rnn_dp_masks['input_core'] is None:
                rnn_dp_masks['input_core'] = get_dp_mask(rnn_in)
            rnn_in = rnn_in.mul(rnn_dp_masks['input_core'])

            if rnn_dp_masks['hidden_core'] is None:
                rnn_dp_masks['hidden_core'] = get_dp_mask(states)
            states = apply_state_masks(states, rnn_dp_masks['hidden_core'])

            new_states = core_rnn(rnn_in, states)
            if isinstance(new_states, tuple):
                rnn_out = new_states[0]
            else:
                rnn_out = new_states

            if rnn_dp_masks['output_core'] is None:
                rnn_dp_masks['output_core'] = get_dp_mask(rnn_out)

            action_h = rnn_out.mul(rnn_dp_masks['output_core'])

            if self.double_core:
                core_rnn2 = self.core_rnn_net[1]
                states2 = core_states[1]

                if rnn_dp_masks['input_core2'] is None:
                    rnn_dp_masks['input_core2'] = get_dp_mask(rnn_out)
                # Detach the output of the core action RNN before using it as
                # the next input to the core location RNN
                rnn2_in =  rnn_out.detach().mul(rnn_dp_masks['input_core2'])

                if rnn_dp_masks['hidden_core2'] is None:
                    rnn_dp_masks['hidden_core2'] = get_dp_mask(states2)
                states2 = apply_state_masks(states2, rnn_dp_masks['hidden_core2'])

                new_states2 = core_rnn2(rnn2_in, states2)
                if isinstance(new_states2, tuple):
                    rnn2_out = new_states2[0]
                else:
                    rnn2_out = new_states2

                if rnn_dp_masks['output_core2'] is None:
                    rnn_dp_masks['output_core2'] = get_dp_mask(rnn2_out)

                loc_h = rnn2_out.mul(rnn_dp_masks['output_core2'])
                rnn_outputs.append(rnn2_out)
                core_states = (new_states, new_states2)
            else:
                # Detach the state of the core network before using it as the
                # next input to the location network
                loc_h = rnn_out.detach().mul(rnn_dp_masks['output_core'])
                rnn_outputs.append(rnn_out)
                core_states = new_states

            state_package = [core_states]

            # Action network
            if self.recurrent_action_net:
                if rnn_dp_masks['hidden_act'] is None:
                    rnn_dp_masks['hidden_act'] = get_dp_mask(action_states)
                action_states = apply_state_masks(
                    action_states, rnn_dp_masks['hidden_act']
                )
                action_states = self.action_net[0](action_h, action_states)
                if isinstance(action_states, tuple):
                    a_t = action_states[0]
                else:
                    a_t = action_states
                
                state_package.append(action_states)
            else:
                a_t = self.action_net(action_h)

            # Concatenate the action network output to the location network input
            if self.input_action_to_loc_net:
                loc_h = torch.cat([loc_h, a_t.detach()], -1)

            # Location network. Disconnect input from computation graph first
            if self.recurrent_loc_net:
                if rnn_dp_masks['hidden_loc'] is None:
                    rnn_dp_masks['hidden_loc'] = get_dp_mask(loc_states)
                loc_states = apply_state_masks(
                    loc_states, rnn_dp_masks['hidden_loc']
                )
                loc_states = self.loc_net(loc_h, loc_states)
                if isinstance(loc_states, tuple):
                    l_t1 = loc_states[0]
                else:
                    l_t1 = loc_states

                state_package.append(loc_states)
            else:
                l_t1 = self.loc_net(loc_h)

            # Append
            actions_seq.append(a_t)
            locs_seq.append(l_t1)
            glimpses_seq.append(g_t)

        # Assemble outputs
        actions_seq = torch.stack(actions_seq, 0)
        if self.smooth_predictions:
            actions_seq = self.sigmoid(actions_seq)
        locs_seq = torch.stack(locs_seq, 0)
        glimpses_seq = torch.stack(glimpses_seq, 0)
        rnn_outputs = torch.stack(rnn_outputs, 0)
        # Don't propagate predicted baseline errors back through the core network
        baseline_rewards = self.baseline(rnn_outputs.detach())

        return actions_seq, locs_seq, rnn_outputs, tuple(state_package), glimpses_seq, baseline_rewards


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
    def __init__(self, image_size, in_channels, glimpse_g_size, loc_g_size,
                glimpse_net_size, sensor_size, glimpse_sizes, g_rep_op = 'cat',
                conv_glimpses = False, glimpse_rep_net = None, pad_imgs = True,
                padding = None, interpolation = PIL.Image.BICUBIC,
                convoluted_glimpses = False, out_channels = None):
        '''
        Inputs
        ---
        `image_size`: the size of the input image as an `int` (not including any
        padding, if that's already been applied). Assumes input images are square.

        `in_channels`: number of channels in each input image.

        `glimpse_g_size`: size of the output of the glimpse representaion. If
        `g_rep_op` is `'mul'`, this is set to be equal to `glimpse_net_size`.

        `loc_g_size`: size of the output of the location representation (total,
        with both dimensions as the input). If `g_rep_op` is `'mul'`, this is
        set to be equal to `glimpse_net_size`.

        `glimpse_net_size`: size of the output of the GlimpseNetwork.

        `sensor_size`: the bandwidth/resolution of the Glimpse sensor, i.e. what
        size each of the glimpses will be resized to, as an `int`. Usually,
        `sensor_size >= 2`.
        
        `glimpse_sizes`: a list of `int`s representing the (square) size of each
        glimpse to extract from the input images. The length of `glimpse_sizes`
        will also be the number of glimpse patches to extract. The actual sizes
        of the resulting glimpse will be scaled to `sensor_size` pixels.

        `g_rep_op`: a string representing how the hidden location and glimpse
        representations within the GlimpseNetwork should be combined together.
        Choices: `'cat'` (default) to have them concatenated and passed through
        a Linear layer with ReLU activation, `'add'` to have them passed through
        separate Linear layers with ReLU activations before being elementwise
        added, or `'mul'` to have them elementwise multiplied (also sets
        `loc_g_size` and `glimpse_g_size` to be equal to `glimpse_net_size`).

        `conv_glimpses`: if `False` (default), a single Linear layer will be
        used to get the glimpse representation. If `True`, the captured glimpses
        will be passed into a classical 2-layer convolutional network (with each
        layer using a 3x3 filter, and with the output of the second layer passed
        through a 2x2 max-pooling layer, then flattened) first. This is ignored
        if `glimpse_rep_net` is not `None`.

        `glimpse_rep_net`: if `None` (default), the glimpse representation
        network will be initialized according to the value of `conv_glimpses`.
        If a PyTorch nn.Module is passed in instead, it will be used as the
        glimpse representation network. This network should take inputs of size
        `(batch_size, channels*num_glimpses, glimpse_height, gilmpse_width)` and
        return outputs of size `(batch_size, glimpse_g_size)`.

        `pad_imgs`: whether or not the input images need padding applied.
        Defaults to `True`, but set to `False` if the input images already have
        padding applied; make sure it's of the correct size!

        `padding`: if `None`, the amount of padding is calculated, otherwise an
        `int` should be passed in.

        `interpolation`: desired interpolation type, specified as an `int` (aka
        one of those constants from the PIL.Image package). Defaults to 3,
        aka `PIL.Image.BICUBIC`.

        `convoluted_glimpses`: if `True`, the GlimpseSensor uses a convolutional
        network with learnable parameters to extract and resize the glimpses.
        Defaults to `False`.

        `out_channels`: if `convoluted_glimpses` is `True`, this should be an
        integer indicating the number of output channels after each convolutional
        layer while extracting and producing the glimpses. If `out_channels` is
        left as `None`, it'll default to the same value as `in_channels`.
        '''
        super(GlimpseNetwork, self).__init__()
        g_rep_op = g_rep_op.lower()
        assert g_rep_op in ['cat', 'add', 'mul']
        if g_rep_op == 'mul':
            loc_g_size = glimpse_net_size
            glimpse_g_size = glimpse_net_size

        self.image_size = image_size
        self.in_channels = in_channels
        self.glimpse_g_size = glimpse_g_size
        self.loc_g_size = loc_g_size
        self.glimpse_net_size = glimpse_net_size
        self.g_rep_op = g_rep_op
        self.sensor_size = sensor_size
        self.num_glimpses = len(glimpse_sizes)
        self.conv_glimpses = conv_glimpses

        ### Layer initializations ###
        # Basic
        self.relu = nn.ReLU()

        # GlimpseSensor
        self.glimpse_sensor = GlimpseSensor(
            image_size = image_size, in_channels = in_channels,
            sensor_size = sensor_size, glimpse_sizes = glimpse_sizes,
            pad_imgs = pad_imgs, interpolation = interpolation, padding = padding,
            convoluted_glimpses = convoluted_glimpses, out_channels = out_channels
        )

        # Location representation subnetwork
        self.loc_layer = nn.Linear(2, loc_g_size)

        # Glimpse representation subnetwork. If glimpse_rep_net is None, we need
        # to create and initialize a subnetwork to use
        if glimpse_rep_net is None:
            # Initialize the desired network type
            channel_depth = in_channels * self.num_glimpses
            if conv_glimpses:
                out_channels = out_channels or in_channels
                unrolled_size = sensor_size**2 * out_channels
                glimpse_rep_net = nn.Sequential(
                    # Convolutional layers 1
                    nn.Conv2d(channel_depth, out_channels, 3, padding = 1),
                    nn.ReLU(),
                    # Convolutional layers 2
                    nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                    nn.ReLU(),
                    # Convolutional layers 3
                    nn.Conv2d(out_channels,out_channels, 3, padding = 1),
                    nn.ReLU(),
                    # Flatten and output
                    Flatten(),
                    nn.Linear(unrolled_size, glimpse_g_size)
                )
            else:
                unrolled_size = sensor_size**2 * channel_depth
                glimpse_rep_net = nn.Sequential(
                    Flatten(),
                    nn.Linear(unrolled_size, glimpse_g_size)
                )
        self.glimpse_layer = glimpse_rep_net

        # GlimpseNetwork output
        if g_rep_op == 'cat':
            self.output_layer = nn.Sequential(
                nn.Linear(glimpse_g_size + loc_g_size, glimpse_net_size),
                nn.ReLU(),
            )
        elif g_rep_op == 'add':
            self.loc_out_layer = nn.Linear(loc_g_size, glimpse_net_size, bias = False)
            self.glimpse_out_layer = nn.Linear(glimpse_g_size, glimpse_net_size)
        # No extra layers needed if g_rep_op == 'mul'

        ### Initialize parameters ###
        self.init()
    
    def init(self):
        layers = [self.loc_layer]
        if self.g_rep_op == 'cat':
            layers.append(self.output_layer)
        elif self.g_rep_op == 'add':
            layers.append(self.loc_out_layer)
            layers.append(self.glimpse_out_layer)
            
        if hasattr(self.glimpse_layer, 'init'):
            self.glimpse_layer.init()
        else:
            if self.conv_glimpses:
                for p in self.glimpse_layer.parameters():
                    if p.dim() > 2:
                        nn.init.kaiming_normal(p)
                    elif p.dim() > 1:
                        if self.g_rep_op == 'mul':
                            nn.init.xavier_normal(p)
                        else:
                            nn.init.kaiming_normal(p)
                    else:
                        p.data.fill_(0)
            else:
                layers.append(self.glimpse_layer)

        for layer in layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    if self.g_rep_op == 'mul':
                        nn.init.xavier_normal(p)
                    else:
                        nn.init.kaiming_normal(p)
                else:
                    p.data.fill_(0)

    def forward(self, locs, imgs = None, glimpses = None):
        '''
        Inputs
        ---
        `locs`: a FloatTensor of size `(batch_size, 2)`, each row representing
        the x- and y-coordinates of the location to center the glimpse around,
        with values in the range of [-1, 1]; location (-1, -1) is the top left
        corner of the image, while (0, 0) is the center.

        `imgs`: input tensor of size `(batch_size, channels, image_height,
        image_width)`. Can be `None` if `glimpses` is not `None`, and ignored if
        it isn't.

        `glimpses`: instead of passing in `imgs`, you can pass the glimpses
        directly in as a tensor of size `(batch_size, channels*num_glimpses,
        glimpse_height, gilmpse_width)`. If this is `None`, then `imgs` can't be
        `None`.

        Outputs
        ---
        `output`: a tensor of size `(batch_size, network_size)`
        
        `glimpses`: the glimpses used this step. If `glimpses` was passed in as
        an input, it'll just be returned back.
        '''
        assert imgs is not None or glimpses is not None

        # Location representation
        loc_rep = self.loc_layer(locs)

        # Glimpse representation
        if glimpses is None and imgs is not None:
            glimpses = self.glimpse_sensor(imgs, locs)
        glimpse_rep = self.glimpse_layer(glimpses)

        # Output
        if self.g_rep_op in ['cat', 'add']:
            # ReLU activations
            loc_rep = self.relu(loc_rep)
            glimpse_rep = self.relu(glimpse_rep)
            # Combine the two representations to get the GlimpseNetwork output
            if self.g_rep_op == 'cat':
                output = self.output_layer(torch.cat([loc_rep, glimpse_rep], -1))
            elif self.g_rep_op == 'add':
                loc_out = self.loc_out_layer(loc_rep)
                glimpse_out = self.glimpse_out_layer(glimpse_rep)
                output = self.relu(loc_out + glimpse_out)
        else:  # Multiply the two representations to get the GlimpseNetwork output
            output = torch.mul(loc_rep, glimpse_rep)

        return output, glimpses


class GlimpseSensor(nn.Module):
    '''
    Given the coordinates of the glimpse and an input image, the GlimpseSensor
    extracts a retina-like representation `rho(x_t, loc_t1)`, centered at
    `loc_t1`, that contains multiple resolution patches.
    '''
    def __init__(self, image_size, in_channels, sensor_size, glimpse_sizes,
                pad_imgs = True, padding = None, convoluted_glimpses = False,
                out_channels = None, interpolation = PIL.Image.BICUBIC):
        '''
        Inputs
        ---
        `image_size`: the size of the input image as an `int` (not including any
        padding, if that's already been applied). Assumes input images are square.

        `in_channels`: number of channels in each input image.

        `sensor_size`: the bandwidth/resolution of the Glimpse sensor, i.e. what
        size each of the glimpses will be resized to, as an `int`. Usually,
        `sensor_size >= 2`.
        
        `glimpse_sizes`: a list of `int`s representing the (square) size of each
        glimpse to extract from the input images. The length of `glimpse_sizes`
        will also be the number of glimpse patches to extract. The actual sizes
        of the resulting glimpse will be scaled to `sensor_size` pixels.
        
        `pad_imgs`: whether or not the input images need padding applied.
        Defaults to `True`, but set to `False` if the input images already have
        padding applied; make sure it's of the correct size!

        `padding`: if `None`, the amount of padding is calculated, otherwise an
        `int` should be passed in.

        `convoluted_glimpses`: if `True`, the GlimpseSensor uses a convolutional
        network with learnable parameters to extract and resize the glimpses.
        Defaults to `False`.

        `out_channels`: if `convoluted_glimpses` is `True`, this should be an
        integer indicating the number of output channels after each convolutional
        layer while extracting and producing the glimpses. If `out_channels` is
        left as `None`, it'll default to the same value as `in_channels`.

        `interpolation`: desired interpolation type, specified as an `int` (aka
        one of those constants from the PIL.Image package). Defaults to 3,
        aka `PIL.Image.BICUBIC`.
        '''
        super(GlimpseSensor, self).__init__()
        glimpse_sizes.sort()
        out_channels = out_channels or in_channels

        self.image_size = image_size
        self.in_channels = in_channels
        self.sensor_size = sensor_size
        self.glimpse_sizes = glimpse_sizes
        self.pad_imgs = pad_imgs
        self.out_channels = out_channels
        self.interpolation = interpolation
        self.num_glimpses = len(glimpse_sizes)
        self.padding = int(np.ceil(glimpse_sizes[-1] / 2)) or padding
        self.convoluted_glimpses = convoluted_glimpses
        if convoluted_glimpses:
            composers = [None for _ in glimpse_sizes]
            output_size = sensor_size**2 * in_channels
            for i, g in enumerate(glimpse_sizes):
                unrolled_size = g**2 * out_channels
                composers[i] = nn.Sequential(
                    # Convolutional layers 1
                    nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                    nn.ReLU(),
                    # Convolutional layers 2
                    nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                    nn.ReLU(),
                    # Convolutional layers 3
                    nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                    nn.ReLU(),
                    # Flatten, affine transform, and unflatten
                    Flatten(),
                    nn.Linear(unrolled_size, output_size),
                    Unflatten(sensor_size, in_channels),
                    # Should this be a ReLU instead? Or no activation? Or a [0, 1] clipping?
                    nn.Sigmoid(),
                )
            self.composer = nn.ModuleList(composers)

            # Initialize weights
            for p in self.composer.parameters():
                if p.dim() > 1:
                    nn.init.kaiming_normal(p)
                else:
                    p.data.fill_(0)
        else:
            # The Torchvision transforms composer converts each image (initially a
            # Tensor) to a PIL image, resizes it, then converts it back to a Tensor
            self.composer = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((sensor_size, sensor_size), interpolation),
                transforms.ToTensor(),
            ])

    def forward(self, imgs, locs):
        '''
        Inputs
        ---
        `imgs`: input tensor of size `(batch_size, channels, image_height,
        image_width)`.

        `locs`: a FloatTensor of size `(batch_size, 2)`, each row representing
        the x- and y-coordinates of the location to center the glimpse around,
        with values in the range of [-1, 1]; location (-1, -1) is the top left
        corner of the image, while (0, 0) is the center.

        Outputs
        ---
        `glimpses`: a tensor of size `(batch_size, channels*num_glimpses,
        glimpse_height, gilmpse_width)`
        '''
        # Get some constant values
        batch_size, channels, height, width = imgs.size()
        p = self.padding

        if self.pad_imgs:
            # Pad the input images
            padded_imgs = torch.zeros(
                batch_size, channels, width + 2*p, height + 2*p
            ).type(torch.FloatTensor)
            padded_imgs[:, :, p:-p, p:-p] = imgs
        else:
            # Input images are already padded
            padded_imgs = imgs

        # Scale locs to be pixel values, including padding
        locs = (locs + 1.) / 2.
        locs = [
            (
                int(p + np.round(locs.data[i, 0] * self.image_size)),
                int(p + np.round(locs.data[i, 1] * self.image_size))
            ) for i in range(batch_size)
        ]
        # Extract the glimpsed regions from each padded image
        unsized_glimpses = [
            [
                padded_imgs[                                                    # [channels, g, g]
                    b, :, x-g//2 : x+(g+1)//2, y-g//2 : y+(g+1)//2
                ] for b, (x, y) in enumerate(locs)
            ] for g in self.glimpse_sizes
        ]
        # Resize the glimpses
        if self.convoluted_glimpses:
            glimpses = torch.cat([
                torch.cat([
                    c(g.unsqueeze(0)) for c, g in zip(self.composers, g_list)
                ], 0) for g_list in unsized_glimpses
            ], 1)
        else:
            glimpses = torch.cat([
                # Have to resize each item in the batch one at a time
                torch.stack([                                                   # [batch_size, channels, g, g]
                    self.composer(g) for g in g_list
                ], 0) for g_list in unsized_glimpses
            ], 1)                                                               # [batch_size, num_glimpses*channels, g, g]
        return Variable(glimpses)


# Flatten module that unrolls an input from size (batch_size, *, ..., *) to size
# (batch_size, *)
class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)


# Unflatten module that rolls an input image up from size (batch_size, *) to
# size (bsz, ch, dim, dim)
class Unflatten(nn.Module):
    def __init__(self, dim, channels):
        super(Unflatten, self).__init__()
        self.dim = dim
        self.channels = channels

    def forward(self, inputs):
        return inputs.view(inputs.size(0), self.channels, self.dim, self.dim)

class Repeater(nn.Module):
    def __init__(self, num_times):
        super(Repeater, self).__init__()
        self.num_times = num_times

    def forward(self, inputs):
        '''
        `inputs`: of size `(B, C, H, W)`
        '''
        return inputs.repeat(1, self.num_times, 1, 1)

class GroupedBatchNorm2d(nn.Module):
    def __init__(self, groups, num_features):
        '''
        `num_features`: number of filters in each filter group, not the total
        number of filters across all filter groups.

        NOTE: `num_features * groups` should equal the size of the channels
        dimension in the input (dimension 1).
        '''
        super(GroupedBatchNorm2d, self).__init__()
        self.groups = groups
        self.num_features = num_features
        self.bn_groups = nn.ModuleList([
            nn.BatchNorm2d(num_features) for _ in range(groups)
        ])

    def forward(self, inputs):
        # torch.split() takes the resulting SIZE of each chunk as input, and not
        # the number of chunks...
        groups = inputs.split(self.num_features, dim = 1)
        bn_out = [bn(g) for bn, g in zip(self.bn_groups, groups)]
        return torch.cat(bn_out, dim = 1)


class RAMContextNetwork(nn.Module):
    def __init__(self, in_channels, img_downsample_size, output_size,
                out_channels = None, interpolation = PIL.Image.BICUBIC,
                kernel_size = 3, filter_groups = 2, unpad = None):
        '''
        Inputs
        ---
        `out_channels`: an integer indicating the number of output channels
        after each convolutional layer while obtaining the context vector. If
        `out_channels` is left as `None`, it'll default to the same value as
        `in_channels`. This value is the number of channels in each filter group,
        not the total number of output channels across all filter groups.

        `kernel_size`: height and width of the convolutional filter kernels. If
        this isn't odd, it'll be made odd by adding 1 (the padding is just
        easier to deal with that way).

        `unpad`: the amount of padding to "unpad" from each image passed in to
        the network. Defaults to `None`, meaning no padding needs to be removed
        from the imput images (so they'll be used as-is), otherwise an integer
        should be used. Currently assumes the amount of padding on each side of
        the input images is the same, because that's how the padding is applied
        in the GlimpseSensor.
        '''
        assert unpad is None or isinstance(unpad, int)
        out_channels = out_channels or in_channels
        kernel_size += 1 if kernel_size % 2 == 0 else 0

        super(RAMContextNetwork, self).__init__()
        self.in_channels = in_channels
        self.img_downsample_size = img_downsample_size
        self.out_channels = out_channels
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.filter_groups = filter_groups
        self.unpad = unpad

        self.composer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_downsample_size, img_downsample_size), interpolation),
            transforms.ToTensor(),
        ])

        # The general architecture for this is loosely inspired by VGGNet/ResNet
        all_out_channels = filter_groups * out_channels
        unrolled_size = img_downsample_size**2 * out_channels  # of each group
        preout_size = output_size * filter_groups
        padding = (kernel_size - 1) // 2
        self.conv_net = nn.Sequential(
            # Repeat the input so that each filter group gets their own copy
            Repeater(filter_groups),
            
            ### Convolutional layers 1
            nn.Conv2d(
                # in_channels, out_channels, kernel_size
                in_channels*filter_groups, all_out_channels, kernel_size,
                padding = padding, groups = filter_groups, bias = False
            ),
            GroupedBatchNorm2d(filter_groups, out_channels),
            nn.ReLU(),
            
            ### Convolutional layers 2
            #GroupedBatchNorm2d(filter_groups, out_channels),
            nn.Conv2d(
                all_out_channels, all_out_channels, kernel_size,
                padding = padding, groups = filter_groups, bias = False
            ),
            GroupedBatchNorm2d(filter_groups, out_channels),
            nn.ReLU(),
            
            ### Convolutional layers 3
            #GroupedBatchNorm2d(filter_groups, out_channels),
            nn.Conv2d(
                all_out_channels, all_out_channels, kernel_size,
                padding = padding, groups = filter_groups, bias = False
            ),
            GroupedBatchNorm2d(filter_groups, out_channels),
            nn.ReLU(),
        )
        self.fc_bn_groups = nn.ModuleList([
            nn.Sequential(
                nn.Linear(unrolled_size, output_size, bias = False),
                nn.BatchNorm1d(output_size),
                nn.ReLU()
            ) for _ in range(filter_groups)
        ])

        self.output = nn.Linear(preout_size, output_size, bias = False)
        self.output_bn = nn.BatchNorm1d(output_size)
        self.tanh = nn.Tanh()  # try with and without nonlinearity

        # For convenience
        self.bn_layers = [
            layer for layer in self.modules()
                if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d))
        ]

        self.init()
        # For visualizations
        self.coarse_imgs = None
        self.context = None

    def init(self, bn_mean = 0, bn_var = 1):
        for p in self.parameters():
            if p.dim() > 1:  # Targets the convolutional and linear layers
                nn.init.kaiming_normal(p)
            else:
                p.data.fill_(0)
        for bn in self.bn_layers:
            bn.weight.data.fill_(bn_var)
            bn.bias.data.fill_(bn_mean)

    def forward(self, imgs):
        '''
        Inputs
        ---
        `imgs`: input tensor of size `(batch_size, channels, image_height,
        image_width)`.


        Outputs
        ---
        `output`: a FloatTensor of size `(batch_size, core_rnn_state_size)` to
        be used as the initial hidden state of the core RNN network in the
        `RecurrentAttentionModel`.
        '''
        batch_size = imgs.size(0)

        coarse_imgs = []
        # Remove padding around the input images if needed
        if self.unpad is not None:
            p = self.unpad
            imgs = imgs[:, :, p:-p, p:-p]
        # Transform composer only works on one image at a time
        for b in range(imgs.size(0)):
            i = self.composer(imgs[b])
            coarse_imgs.append(i)
        # Now, images are of size `(batch_size, channels, downsample_size, downsample_size)`
        coarse_imgs = Variable(torch.stack(coarse_imgs, 0))
        self.coarse_imgs = coarse_imgs.data
        
        # Pass the coarse images through the CNN groups
        conv = self.conv_net(coarse_imgs)
        
        # Split the output of the CNN subnetwork along the channels dimension
        conv_groups = conv.split(self.out_channels, dim = 1)
        # Unroll each group output, pass through affine transformation, batch
        # normalize, then pass through ReLU activation
        fc_bn_out = [
            fc_bn(
                cg.contiguous().view(batch_size, -1)
            ) for fc_bn, cg in zip(self.fc_bn_groups, conv_groups)
        ]
        # Concatenate and pass through output layer
        context = self.output_bn(  #self.tanh(self.output_bn(
            self.output(torch.cat(fc_bn_out, -1))
        )  #))
        self.context = context.data
        return context

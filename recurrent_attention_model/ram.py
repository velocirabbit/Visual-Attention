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

Inputs:
`image_size`: the size of the input image as an `int`. For now, assumes the
        image is square.
`in_channels`: number of channels in the input images (typically 1 for
    grayscale, 3 for RGB, 4 for RGBa)
`glimpse_h_size`: size of the hidden layer for the independent glimpse linear
    layer representation
`loc_h_size`: size of the hidden layer for the independent location linear
    layer representation (for each dimension!)
`h_size`: size of the hidden layer for the combined glimpse/location linear
    layer representation
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
class GlimpseNetwork(nn.Module):
    def __init__(self, image_size, in_channels,
                glimpse_h_size, loc_h_size, h_size,
                glimpse_sizes = [0.05, 0.1, 0.3, 0.6],
                pad_imgs = True, learn_kernels = False, a = -0.5,
                preserve_out_channels = False):
        super(GlimpseNetwork, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = in_channels if preserve_out_channels else 1
        self.glimpse_h_size = glimpse_h_size
        self.loc_h_size = loc_h_size
        self.h_size = h_size
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
        self.h_layer = nn.Linear(glimpse_h_size + 2*loc_h_size, h_size)

        # Initialize weights
        self.init()

    def init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
            else:
                p.data.fill_(0)

    def forward(self, imgs, locs):
        nbatches = imgs.size(0)
        # Get glimpse representations for the images
        glimpses = self.glimpse_sensor(imgs, locs)                              # (nbatches, out_channels, gH, gW)
        # Embed the glimpse locations
        x_loc = Variable(                                                       # (nbatches, 1)
            torch.LongTensor([loc[0] for loc in locs]).unsqueeze(1)
        )
        y_loc = Variable(                                                       # (nbatches, 1)
            torch.LongTensor([loc[1] for loc in locs]).unsqueeze(1)
        )

        x_embeds = self.loc_embeds[0](x_loc)                                    # (nbatches, loc_h_size)
        y_embeds = self.loc_embeds[1](y_loc)                                    # (nbatches, loc_h_size)

        # Concatenate
        loc_embed = torch.cat([x_embeds, y_embeds], -1)                         # (nbatches, 2*loc_h_size)
        # Pass through final linear layer
        glimpse_net = self.h_layer(                                             # (nbatches, h_size)
            # (nbatches, unrolled_glimpse_size + 2*loc_h_size)
            torch.cat([
                glimpses.view(nbatches, -1), loc_embed
            ], -1)
        )

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
        self.init(a, in_channels, preserve_out_channels)
        self.unrolled_glimpse_size = glimpse_sizes[0]**2 * self.out_channels

    '''
    Initializes a set of convolutional kernels to use during resizing. This uses
    bicubic interpolation, so the kernel function is given as:

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
    def init(self, a, in_channels = 3, preserve_out_channels = False):
        g_sz = self.glimpse_sizes[0]
        kernel_sizes = [
            sz - g_sz + 1 for sz in self.glimpse_sizes
        ]
        out_channels = in_channels if preserve_out_channels else 1
        if not self.learn_kernels:
            self.a = a
            # x-values to apply the filter function over
            kernels = [
                np.arange(-sz/2.+0.5, sz/2.+0.5)[..., np.newaxis]*4./sz
                for sz in kernel_sizes
            ]
            # Kernel values in 1D
            kernels = [np.apply_along_axis(lambda x:
                [0.] if x > 2 else (
                    a*x**3. - 5.*a*x**2. + 8.*a*x - 4.*a if x > 1 else
                    (a+2.)*x**3. - (a+3.)*x**2. + 1.
                ), 1, np.abs(k)
            ) for k in kernels]
            # Kernel values in 2D
            kernels = [np.matmul(k, k.T) for k in kernels]
            # Normalize and turn into torch tensors
            kernels = [
                Variable(
                    torch.from_numpy((k * out_channels)/(in_channels * k.sum()))
                        .expand(out_channels, in_channels, -1, -1)
                        .type(torch.FloatTensor),
                    requires_grad = False
                ) for k in kernels
            ]
            self.kernels = kernels
        else:
            self.kernels = nn.ModuleList([  # Convolution kernels
                nn.init.xavier_normal(
                    Variable(torch.zeros((out_channels, in_channels, sz, sz)))
                ) for sz in kernel_sizes
            ])

    '''
    Inputs:
    `imgs`: input tensor of size `(batch_size, channels, image_height, image_width)`
    `locs`: a list of 2-tuples representing the `(x_pos, y_pos)` for each image
        in the batch to center the extracted glimpses around
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
        return unsized_glimpses, glimpses

    '''
    Performs 2D convolution on the input images. `kernel`s need to be either `None`
    or a tensor of size `(out_channels, in_channels, kernel_height, kernel_width)`
    '''
    def resize(self, img, kernel = None):
        if kernel is None:
            return img
        # Convolute the kernel over the image and return the downsampled glimpse
        resized_img = F.conv2d(img, kernel)
        return resized_img
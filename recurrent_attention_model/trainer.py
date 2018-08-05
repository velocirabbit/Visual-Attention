'''
Helper functions for training and evaluating a RAM.

Training techniques to try:
    - Rather than using a random or fixed initial glimpse location, have the
    agent learn where to start by feeding the entire image through a simple
    network (e.g. a basic 2-layer convnet) that's trained via REINFORCE
    - Learn a baseline reward for each time step
    - Have either a fixed number of glimpses, or allow the model to take as many
    glimpses as it needs (up to some max) until it makes a correct
    classification  => probably won't work on this one since it's currently
    training well without this
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as distributions
import torch.optim as optim
import torch.nn as nn

from matplotlib import gridspec
from torch.autograd import Variable

import time

CMAP = 'cubehelix'

def discount_rewards(rewards, gamma):
    '''
    Discounts a sequence of rewards using the decay rate gamma.

    Inputs
    ---
    `rewards`: `list` of numbers representing the rewards given at each step
    
    `gamma`: discount factor as a `float` in the range of [0, 1]

    Outputs
    ---
    `discounted_rewards`: `list` of `float`s representing the discounted rewards
    given at each step
    '''
    seq_len = rewards.size(0)
    batch_size = rewards.size(1)
    discounted_rewards = torch.zeros_like(rewards)
    future_rewards = torch.zeros(batch_size)
    for t in reversed(range(seq_len)):
        future_rewards = future_rewards*gamma + rewards[t]
        discounted_rewards[t] = future_rewards
    return discounted_rewards

def get_lr_scheduler(size, warmup_steps, decay_factor, optimizer, min_lr = 0):
    lrate = lambda e: max(min_lr, size**(-0.5) * min(
        (e+1)**(-decay_factor), (e+1) * warmup_steps**(-(decay_factor+1))
    ))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lrate)

def visualize_context_net(context_net):
    coarse_img = context_net.coarse_imgs[0]
    num_fg = context_net.filter_groups
    # Convolutional layer weights
    conv_wts = [
        layer.weight.data for layer in context_net.conv_net.modules()
            if isinstance(layer, nn.Conv2d)
    ]
    # Fully connected layer weights
    fc_wts = [
        layer.weight.data for layer in context_net.fc_bn_groups.modules()
            if isinstance(layer, nn.Linear)
    ]
    fc_wts.append(context_net.output.weight.data)
    # Batchnorm weights
    cv_bn_wts = [
        (layer.weight.data, layer.bias.data)
            for layer in context_net.conv_net.modules()
            if isinstance(layer, nn.BatchNorm2d)
    ]
    fc_bn_wts = [
        (layer.weight.data, layer.bias.data)
            for layer in context_net.fc_bn_groups.modules()
            if isinstance(layer, nn.BatchNorm1d)
    ] + [(context_net.output_bn.weight.data, context_net.output_bn.bias.data)]

    out_channels = conv_wts[-1].size(0)

    fig = plt.figure(figsize = (14, 10))
    ds_img_w = 3
    n_conv_layers = len(conv_wts)
    fig_w = ds_img_w + out_channels*(out_channels // num_fg)
    fig_h = max(ds_img_w, n_conv_layers)
    tot_fig_h = 2*fig_h+8
    grids = gridspec.GridSpec(tot_fig_h, fig_w)

    # Plot the downsampled image
    img_ax = fig.add_subplot(grids[:fig_h, :ds_img_w])
    img_ax.imshow(
        coarse_img.squeeze(), aspect = 'equal',
        cmap = CMAP, vmin = 0, vmax = 1
    )
    img_ax.set_title('Downsampled image')

    # Plot the 2D and 1D batchnorm weights and biases
    bn_wts = [cv_bn_wts, fc_bn_wts]
    for i in range(2):  # 0 := weights, 1 := biases
        vmin = min(bn[i].min() for bn in cv_bn_wts + fc_bn_wts)
        vmax = max(bn[i].max() for bn in cv_bn_wts + fc_bn_wts)
        for d in range(2):  # 0 := 2D, 1 := 1D
            num = len(bn_wts[d])
            ax_w = fig_w // num
            ar = 'equal' if bn_wts[d][0][0].numel() < 16 else (2*bn_wts[d][0][0].numel())/32
            for g in range(num):
                vs = 2*(2*i + d)
                ax = fig.add_subplot(grids[2*fig_h+vs:2*fig_h+vs+2, g*ax_w:(g+1)*ax_w])
                bn_img = ax.imshow(
                    bn_wts[d][g][i].unsqueeze(0), aspect = ar, cmap = CMAP,
                    vmin = vmin, vmax = vmax
                )
                if d == 1:  # for the 1D parameters...
                    ax.axes.get_yaxis().set_visible(False)
                else:  # d == 1, so for the 2D parameters...
                    ax.tick_params(
                        left = 'off', labelleft = 'off',
                        bottom = 'off', labelbottom = 'off',
                    )
                if g == 0:
                    ax.set_title('%dD BN %s' % (
                        ((d+1)%2)+1, 'weights (gamma)' if i == 0 else 'biases (beta)'
                    ))
                elif d == 1 and g == num - 1:
                    ax.set_title('output 1D BN %s' % (
                        'weights (gamma)' if i == 0 else 'biases (beta)'
                    ))
        c_ax = fig.add_axes([0.1, ((i+1)%2)*4/tot_fig_h, 0.8, 0.005])  # [left, bottom, width, height]
        fig.colorbar(bn_img, cax = c_ax, orientation = 'horizontal')

    # Plot the linear layer weights. Get the min and max across the linear layers
    vmin = min(wts.min() for wts in fc_wts)
    vmax = max(wts.max() for wts in fc_wts)
    ax_w = fig_w // (num_fg + 1)
    aspect = fc_wts[0].size(0) / fc_wts[0].size(1)
    aspect = 'equal' if aspect > 0.125 else 16*aspect
    for g in range(num_fg):
        ax = fig.add_subplot(grids[fig_h:2*fig_h, g*ax_w:(g+1)*ax_w])
        ax.imshow(
            fc_wts[g], aspect = aspect, cmap = CMAP, vmin = vmin, vmax = vmax
        )
        ax.set_title('FC #%d' % (g+1))
        if g % 2 != 0:
            ax.yaxis.tick_right()
    ax = fig.add_subplot(grids[fig_h:2*fig_h, -ax_w:])
    fc_img = ax.imshow(
        fc_wts[-1], aspect = 'equal', cmap = CMAP, vmin = vmin, vmax = vmax
    )
    ax.set_title("Output FC layer")
    c_ax = fig.add_axes([0.1, 8/tot_fig_h, 0.8, 0.005])
    fig.colorbar(fc_img, cax = c_ax, orientation = 'horizontal')

    # Visualize some of the convolutional filters. Get the filter ranges
    vmin = min(wts.min() for wts in conv_wts)
    vmax = max(wts.max() for wts in conv_wts)
    # First conv layer only has one set for each group
    axs = []
    for c in range(out_channels):
        ax = fig.add_subplot(grids[0, ds_img_w + c])
        f_img = ax.imshow(
            conv_wts[0][c, 0], aspect = 'equal',
            cmap = CMAP, vmin = vmin, vmax = vmax
        )
        ax.axis('off')
        axs.append(ax)
    axs[-1].set_title('Conv. filter visualizations')
    # Visualize the other two convolutional layers' filters
    for w in range(1, n_conv_layers):
        for g in range(out_channels // num_fg):
            for c in range(out_channels):
                ax = fig.add_subplot(grids[w, ds_img_w + c + g*out_channels])
                ax.imshow(
                    conv_wts[w][c, g], aspect = 'equal',
                    cmap = CMAP, vmin = vmin, vmax = vmax
                )
                ax.axis('off')
    left = ds_img_w / fig_w + 0.1
    c_ax = fig.add_axes([left, (8+fig_h)/tot_fig_h, 0.9 - left, 0.005])
    fig.colorbar(f_img, cax = c_ax, orientation = 'horizontal')
    #fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 1)
    grids.update(left = 0, right = 1, bottom = 0, top = 1, hspace = 1)
    plt.show()

def glimpse_visualization(image, glimpse_seq, loc_seq, padding, image_size,
                          target_class, predicted_classes, predicted_class_probs,
                          states = None, mean_locs = None, sampled_locs = None,
                          explored = None):
    num_glimpses = len(predicted_classes)
    # Convert the location sequence into pixel indices in the image,
    # split into two lists for the x- and y- coordinates. First, scale
    # loc_seq so each value is between [0, 1] instead of [-1, 1]
    loc_seq = (loc_seq + 1.) / 2.
    x_locs = [padding + int(np.round(x * image_size)) for x in loc_seq[:, 1]]
    y_locs = [padding + int(np.round(y * image_size)) for y in loc_seq[:, 0]]

    glimpse_depth = glimpse_seq.size(1) // image.size(-1)
    glimpse_size = glimpse_seq.size(-1)

    fig = plt.figure(figsize = (10 if glimpse_depth < 3 else 12, 8.5))
    img_sz = max(1, num_glimpses - (0 if states is None else 1))
    grids = gridspec.GridSpec(img_sz + (0 if states is None else 1), img_sz+glimpse_depth)

    # Plot the main, full image
    img_ax = fig.add_subplot(grids[:img_sz, :img_sz])
    img_ax.imshow(image.squeeze(), aspect = 'equal', cmap = CMAP, vmin = 0, vmax = 1)
    img_ax.scatter(x = x_locs, y = y_locs, c = 'magenta')
    for i, (x, y, pc) in enumerate(zip(x_locs, y_locs, predicted_classes)):
        img_ax.text(
            x-1, y-0.65, '%d) %d' % (i+1, pc),
            color = 'white', size = 'large'
        )

    if states is not None:
        states_min = states.min()
        states_max = states.max()
        st_ax = fig.add_subplot(grids[img_sz:, :img_sz])
        st_im = st_ax.imshow(states, aspect = 'auto', cmap = CMAP)
        st_ax.set_yticks(range(0, num_glimpses, 2))
        st_ax.set_yticklabels(range(1, num_glimpses+1, 2))
        st_ax.set_xlabel('Core state after glimpse (min: %.3e , max: %.3e)' % (states_min, states_max))
        st_ax.set_ylabel('Step')
        c_ax = fig.add_axes([-.005, 0.1, 0.005, 0.9/(img_sz+1)])
        fig.colorbar(st_im, cax = c_ax)

    # To visualize the glimpses properly, the channels dimension must be last
    glimpse_seq = glimpse_seq.permute(0, 2, 3, 1)
    for t, pc in enumerate(predicted_classes):
        this_glimpse = glimpse_seq[t]
        glimpse_layers = this_glimpse.chunk(n_chunks = glimpse_depth, dim = -1)
        axs = []
        for d, g in enumerate(glimpse_layers):
            ax = fig.add_subplot(grids[t, img_sz+d])
            ax.imshow(g.squeeze(), aspect = 'equal', cmap = CMAP, vmin = 0, vmax = 1)
            ax.axis('off')
            axs.append(ax)
        g_txt_vert = glimpse_size // 2
        g_txt_horz = glimpse_size + 1
        if mean_locs is not None and sampled_locs is not None:
            g_txt_vert = glimpse_size // 3
            g_txt = '({:.2f}, {:.2f}) ~> ({:.2f}, {:.2f})'.format(
                mean_locs[t, 1], mean_locs[t, 0],
                sampled_locs[t, 1], sampled_locs[t, 0]
            )
            if explored is not None and explored[t]:
                g_txt_col = 'blue'
            elif any(np.abs(loc) == 1.0 for loc in mean_locs[t]):
                g_txt_col = 'red'
            else:
                g_txt_col = 'black'
            axs[-1].text(
                g_txt_horz, 2*g_txt_vert, g_txt, color = g_txt_col
            )
        axs[-1].text(
            g_txt_horz, g_txt_vert, '{:2d}) predicted {:d} (p = {:.2%})'.format(
                t+1, pc, predicted_class_probs[t]
            )
        )
    #grids.tight_layout(fig, pad = 0, h_pad = 0.5)
    grids.update(wspace = 0, hspace = 0.1)
    plt.show()

def visualize_model_parameters(model):
    params = [p for p in model.base_net.parameters() if p.grad is not None]
    fig = plt.figure(figsize = (12, 3 if model.context_net is None else 9))

    w_ax = plt.subplot(321)
    weights = np.array(torch.cat([w.data.view(-1) for w in params if w.dim() > 1]))
    w_ax.hist(weights, bins = 'auto', density = True, stacked = True)
    w_ax.set_title('Base net weights (mean: %.3e, var: %.3e)' % (
        np.mean(weights), np.var(weights)
    ))

    b_ax = plt.subplot(322)
    biases = np.array(torch.cat([b.data for b in params if b.dim() == 1]))
    b_ax.hist(biases, bins = 'auto', density = True, stacked = True)
    b_ax.set_title('Base net biases (mean: %.3e, var: %.3e)' % (
        np.mean(biases), np.var(biases)
    ))

    if model.context_net is not None:
        params = [p for p in model.context_net.parameters() if p.grad is not None]
        
        weights = np.array(torch.cat([w.data.view(-1) for w in params if w.dim() > 1]))
        cn_ax = plt.subplot(312)
        cn_ax.hist(weights, bins = 'auto', density = True, stacked = True)
        cn_ax.set_title('Context net parameters (mean: %.3e, var: %.3e)' % (
            np.mean(weights), np.var(weights)
        ))

        bn_wts = np.array(torch.cat([
            layer.weight.data for layer in model.context_net.bn_layers
        ]))
        bn_w_ax = plt.subplot(325)
        bn_w_ax.hist(bn_wts, bins = 'auto', density = True, stacked = True)
        bn_w_ax.set_title('Context net BN weights (gamma) (mean: %.3e, var: %.3e)' % (
            np.mean(bn_wts), np.var(bn_wts)
        ))

        bn_bias = np.array(torch.cat([
            layer.bias.data for layer in model.context_net.bn_layers
        ]))
        bn_b_ax = plt.subplot(326)
        bn_b_ax.hist(bn_bias, bins = 'auto', density = True, stacked = True)
        bn_b_ax.set_title('Context net BN biases (beta) (mean: %.3e, var: %.3e)' % (
            np.mean(bn_bias), np.var(bn_bias)
        ))

    fig.tight_layout()
    plt.show()

def train(model, train_data, image_size, padding, optimizer, lr_scheduler,
        class_criterion, use_full_class_seq = False,loc_std = 1., epsilon = 0.,
        epsilon_min = 0., reward_all_correct = False, error_penalty = 0.,
        context_net = None, random_start_loc = True, gamma = 0.99, grad_clip = 1.,
        max_ep_len = 30, end_train_ep_early = False, steps_per_epoch = 1,
        log_freq = 1000, viz_freq = 1000):
    '''
    Trains the model once over the training data.

    Inputs
    ---
    `model`: PyTorch `nn.Module` object
    
    `train_data`: PyTorch DataIterator object, where each example is a tuple of
    `(image, target_class)` for this task
    
    `image_size`: original image size before any padding is applied. Used for
    visualizations
    
    `padding`: amount of padding applied to the original images. Used for
    visualizations
    
    `optimizer`: PyTorch `nn.optim` optimizer object
    
    `lr_scheduler`: PyTorch `nn.optim.lr_scheduler` LR scheduler object
    
    `class_criterion`: loss criteria to use for the classification task (should
    usually be CrossEntropyLoss)

    `use_full_class_seq`: if `True`, calculates the classification error using
    the classification output from each step, meaning we accumulate gradients
    from each step, too. Defaults to `False`.
    
    `loc_std`: standard deviation to use when sampling for the next glimpse
    locations from a normal distribution where the outputs of the location
    network are the distribution means

    `epsilon`: a float (in the range [0, 1]) that determines the chance of
    randomly exploring a new glimpse location rather than using the one output
    by the model. `epsilon` will be used as the starting value, and will be
    annealed to `epsilon_min` over time.

    `epsilon_min`: minimum epsilon value to anneal to, starting from `epsilon`,
    as training progresses.

    `reward_all_correct`: if `True`, awards a point for each step where the
    agent classifies the image correctly. Defaults to `False`.
    
    `error_penalty`: penalty given after each step in an episode to discourage
    the model from taking too many glimpses

    `context_net`: if not `None`, the agent uses the context network to learn
    an initial glimpse location (rather than picking one randomly or always
    starting at the origin), and `random_start_loc` is ignored
    
    `random_start_loc`: if `True` (default), uses a random starting location,
    otherwise starts the glimpse sequence at (0, 0) every episode
    
    `gamma`: discount factor used to discount the reward sequences
    
    `grad_clip`: gradient clipping threshold
    
    `max_ep_len`: maximum number of glimpses the model is allowed to take each
    episode

    `end_train_ep_early`: if `True`, each episode during training is ended early
    if the agent makes a correct classification; training must be done with a
    batch size of 1. Defaults to `False`.
    
    `steps_per_epoch`: number of times to step the LR each epoch
    
    `log_freq`: number of episodes between training log metrics print-outs
    
    `viz_freq`: number of episodes between glimpse location visualizations

    Outputs
    ---
    `accuracy`: epoch classification accuracy, as a ratio between [0, 1]
    
    `avg_steps`: average number of steps per episode
    
    `avg_reward`: average reward per episode
    
    `avg_class_loss`: average classification loss per episode
    
    `avg_grad_inf_norm`: average infinity norm of the gradients per episode
    
    `avg_grad_2norm`: average 2-norm of the gradients per episode
    '''
    num_train_data = len(train_data)
    # For metrics
    steps_hist         = []  # Number of steps to the first correct classification in the sequence
    reward_hist        = []
    class_loss_hist    = []  # As a batch average
    success_hist       = []
    grad_inf_norm_hist = []
    grad_2norm_hist    = []
    wts_max_hist       = []
    wts_mean_hist      = []
    wts_var_hist       = []
    bias_max_hist      = []
    bias_mean_hist     = []
    bias_var_hist      = []
    wt_udr_max_hist    = []
    wt_udr_mean_hist   = []
    wt_udr_var_hist    = []
    bias_udr_max_hist  = []
    bias_udr_mean_hist = []
    bias_udr_var_hist  = []

    baseline_criterion = nn.MSELoss(reduce = False)

    model.train()
    #if context_net is not None:
    #    context_net.train()

    last_ep = 0
    last_batch = 0
    eps = epsilon
    d_eps = (epsilon - epsilon_min) / (num_train_data * max_ep_len)
    start_time = time.time()
    for batch_num, (image, target_class) in enumerate(train_data):
        ########## OVERHEAD ##########
        if batch_num > 0 and batch_num % (num_train_data // steps_per_epoch) == 0:
            lr_scheduler.step()

        batch_size = image.size(0)
        assert (end_train_ep_early and batch_size == 1) or not end_train_ep_early

        glimpse_seq           = []
        loc_seq               = []
        mean_loc_seq          = []
        sampled_loc_seq       = []
        class_probs_seq       = []
        predicted_classes     = []
        reward_seq            = []
        baseline_seq          = []
        states_seq            = []
        explore_seq           = []

        # Initialize RNN states and initial glimpse location
        model.zero_grad()
        model.reset_dp_masks()
        if model.context_net is None:
            rnn_states = model.init_rnn_states(batch_size)
            loc_means = torch.from_numpy(
                np.random.uniform(-1, 1, size = (1, batch_size, 2))
            ).type(torch.FloatTensor) if random_start_loc else torch.zeros(1, batch_size, 2)
            loc_means = Variable(loc_means)
        else:
            # Pass the image to the context_net to get the initial context,
            # which we use as the initial hidden state of the recurrent location
            # network and to get the initial glimpse location from the location
            # (emission) network
            rnn_states, loc_means = model.get_context_loc(image, batch_size)
        loc_means = torch.clamp(loc_means, -1, 1)
        mean_loc_seq.append(loc_means)
        # Add some noise to this initial location mean to get the sampled location
        loc_noise = torch.normal(torch.zeros_like(loc_means), loc_std)
        loc = torch.clamp(loc_means + loc_noise, -1, 1)
        sampled_loc_seq.append(loc)
            
        ########## EXPLORATION/EVALUATION WHILE LOOP ##########
        # Number of steps to the first successful classification or max_ep_len
        steps = [None for _ in range(batch_size)]
        # During training, we'll run for the entire episode length
        for t in range(max_ep_len):
            loc_seq.append(loc)

            # Step forward through the network
            class_probs, loc_means, _, rnn_states, glimpses, baseline = model(
                loc, rnn_states, imgs = image
            )
            loc_means = torch.clamp(loc_means, -1, 1)
            class_probs_seq.append(class_probs)
            mean_loc_seq.append(loc_means)
            # With probability eps, we sample the next glimpse location from
            # anywhere in the image uniformly at random. Otherwise, we'll sample
            # the next glimpse location from a normal distribution using loc as
            # the mean and loc_std as the standard deviation, then clamp the
            # sampled locations between [-1, 1]
            if np.random.rand(1) < eps:
                loc = Variable(torch.from_numpy(
                    np.random.uniform(-1, 1, size = (1, batch_size, 2))
                ).type(torch.FloatTensor))
                explore_seq.append(True)
            else:
                loc_noise = torch.normal(torch.zeros_like(loc_means), loc_std)
                sampled_loc = loc_means + loc_noise
                loc = torch.clamp(sampled_loc, -1, 1)
                explore_seq.append(False)
            sampled_loc_seq.append(loc)

            # Check to see if the model made a correct classification
            # predicted_class_probs and predicted_class are of size:
            # [seq_len = 1, batch_size]
            _, predicted_class = torch.max(class_probs, dim = -1)
            predicted_class = predicted_class.data
            successful = predicted_class == target_class

            # Give a reward or penalty for this step
            reward = successful.type(torch.FloatTensor)
            if end_train_ep_early and successful.squeeze(0)[0]:
                reward.fill_(1.)
            elif reward_all_correct or t == max_ep_len - 1:
                reward[predicted_class != target_class] = error_penalty
            else:
                reward.fill_(error_penalty)
            successful = successful.squeeze(0)

            # Append to the episode histories
            if isinstance(rnn_states[0], tuple):
                if isinstance(rnn_states[0][0], tuple):
                    states_seq.append(rnn_states[0][0][0])
                else:
                    states_seq.append(rnn_states[0][0])
            else:
                states_seq.append(rnn_states[0])
            glimpse_seq.append(glimpses)
            reward_seq.append(reward)
            baseline_seq.append(baseline)
            predicted_classes.append(predicted_class)

            # Overhead stuff if we have a successful classification
            for b in range(batch_size):
                if successful[b]:
                    steps[b] = t + 1 if steps[b] is None else steps[b]
            if end_train_ep_early and successful[0]:
                break

            # Anneal epsilon
            eps -= d_eps

        ### End of while loop ###
        reward_seq = torch.cat(reward_seq, 0)
        total_reward = reward_seq.sum(0)
        steps = [max_ep_len if s is None else s for s in steps]
        ########## LEARNING/BACKPROPAGATION ##########
        # Concatenate each of the relevant sequences. Don't propagate gradients
        # from the sampled glimpse locations
        mean_loc_seq = torch.cat(mean_loc_seq[:-1], 0)                          # [seq_len, batch_size, 2]
        sampled_loc_seq = torch.cat(sampled_loc_seq[:-1], 0).detach()           # [seq_len, batch_size, 2]
        baseline_seq = torch.cat(baseline_seq, 0).squeeze(-1)                   # [seq_len, batch_size, 1] -> [seq_len, batch_size]

        if use_full_class_seq:
            '''
            Need to test out how well this works when an error penalty is used.
            When there isn't one, obtaining a classification loss for each step
            seems to result in the location network learning to output the same
            corner of the image every time (also need to investigate why it
            would do that).
            '''
            _class_probs_seq_ = torch.cat(class_probs_seq, 0)
            num_classes = _class_probs_seq_.size(-1)

            _class_probs_seq_ = _class_probs_seq_.view(-1, num_classes)
            target_classes = target_class.expand(max_ep_len, -1).contiguous().view(-1)
            # MAYBE multiply the class loss by the length of the sequence, which
            # basically just "undoes" the averaging over the sequence
            class_loss = class_criterion(_class_probs_seq_, target_classes)
        else:
            # Get the classification loss for only the very last step
            class_loss = class_criterion(
                class_probs_seq[-1].squeeze(0),Variable(target_class.view(-1))
            )

        discounted_rewards = discount_rewards(reward_seq, gamma)
        #if end_train_ep_early: #and steps > 1: #and not successful:  # ??
            # If we're ending the episode early, demean the rewards sequence
        #    discounted_rewards -= discounted_rewards.mean(0)
        # Wrap the discounted rewards sequence in a PyTorch Variable
        reward_seq = Variable(discounted_rewards).detach()                      # [seq_len, batch_size]

        # Use the mean and sampled locs to get each step's log-likelihood
        loc_distributions = distributions.Normal(mean_loc_seq, loc_std)
        loc_loss = loc_distributions.log_prob(sampled_loc_seq)                  # [seq_len, batch_size, 2]
        # Sum the losses for the predicted x- and y-coordinates
        loc_loss = loc_loss.sum(dim = -1)                                       # [seq_len, batch_size]
        # For each step, scale the location loss by the discounted reward
        # minus that step's baseline reward, then take the mean of the entire
        # tensor and multiply by the length of the sequence
        loc_loss = loc_loss.mul(reward_seq - baseline_seq.detach())             # [seq_len, batch_size]
        # Get the baseline loss. x2 since we have x- and y-coordinates
        baseline_loss = baseline_criterion(baseline_seq, reward_seq).mul(2)     # [seq_len, batch_size]
        # Add the class loss, location loss, and baseline loss. loc_loss is
        # negative since it's a log-likelihood loss we're trying to minimize,
        # not maximize. Start by element-wise adding -loc_loss and baseline_loss
        loc_baseline_loss = baseline_loss - loc_loss                            # [seq_len, batch_size]
        # Sum the loss over the entire sequence, then average over the batches
        # sizes: [seq_len, batch_size] -> [batch_size] -> scalar
        loc_baseline_loss = loc_baseline_loss.sum(0).mean()
        # Get the total loss
        loss = class_loss + loc_baseline_loss

        # Propagate the loss gradient back
        loss.backward()
        # Clip the gradients using either the given gradient clipping threshold,
        # or the infinity norm of the gradients, whichever is larger. This lets
        # the clipping threshold increase as training progresses, allowing for
        # larger updates late in the training when the gradients are larger. We
        # also use the infinity norm as a metric.
        params = [p for p in model.base_net.parameters() if p.grad is not None]
        grad_inf_norm = max(p.grad.data.abs().max() for p in params)
        grad_2norm = float(nn.utils.clip_grad_norm(
            model.base_net.parameters(), grad_clip
        ))
        if model.context_net is not None:
            cn_grad_inf_norm = max(
                p.grad.data.abs().max() for p in model.context_net.parameters()
                    if p.grad is not None
            )
            cn_grad_2norm = float(nn.utils.clip_grad_norm(
                model.context_net.parameters(), grad_clip
            ))

        # Update the parameters
        optimizer.step()

        ########## POST-EPISODE STUFF ##########
        weights = np.array(torch.cat([w.data.view(-1) for w in params if w.dim() > 1]))
        biases = np.array(torch.cat([b.data.view(-1) for b in params if b.dim() == 1]))
        wt_udr = np.array(torch.cat([
            (w.grad/w).data.abs().view(-1) for w in params if w.dim() > 1
        ]))
        b_udr = np.array(torch.cat([
            (b.grad/b).data.abs().view(-1) for b in params if b.dim() == 1
        ]))

        wts_max = max(np.abs(weights))
        wts_mean = np.mean(weights)
        wts_var = np.var(weights)
        bias_max = max(np.abs(biases))
        bias_mean = np.mean(biases)
        bias_var = np.var(biases)
        wt_udr_max = max(wt_udr)
        wt_udr_mean = np.nanmean(wt_udr)
        wt_udr_var = np.nanvar(wt_udr)
        b_udr_max = max(b_udr)
        b_udr_mean = np.nanmean(b_udr)
        b_udr_var = np.nanvar(b_udr)

        # Metrics. These track each example...
        steps_hist.extend(steps)
        reward_hist.extend(list(total_reward))
        success_hist.extend(list(successful.type(torch.FloatTensor)))
        # ... these track each batch
        class_loss_hist.append(float(class_loss))
        grad_inf_norm_hist.append(grad_inf_norm)
        grad_2norm_hist.append(grad_2norm)
        wts_max_hist.append(wts_max)
        wts_mean_hist.append(wts_mean)
        wts_var_hist.append(wts_var)
        bias_max_hist.append(bias_max)
        bias_mean_hist.append(bias_mean)
        bias_var_hist.append(bias_var)
        wt_udr_max_hist.append(wt_udr_max)
        wt_udr_mean_hist.append(wt_udr_mean)
        wt_udr_var_hist.append(wt_udr_var)
        bias_udr_max_hist.append(b_udr_max)
        bias_udr_mean_hist.append(b_udr_mean)
        bias_udr_var_hist.append(b_udr_var)

        # Print out training metrics
        if (batch_num+1) % log_freq == 0:
            runtime = 1000.*(time.time() - start_time) / log_freq

            # Get averages
            this_lr           = np.mean(lr_scheduler.get_lr())
            avg_steps         = np.mean(steps_hist[last_ep:])
            avg_reward        = np.mean(reward_hist[last_ep:])
            accuracy          = np.mean(success_hist[last_ep:])
            avg_class_loss    = np.mean(class_loss_hist[last_batch:])
            avg_grad_2norm    = np.mean(grad_2norm_hist[last_batch:])
            avg_grad_inf_norm = np.mean(grad_inf_norm_hist[last_batch:])
            avg_wts_max       = np.mean(wts_max_hist[last_batch:])
            avg_wts_mean      = np.mean(wts_mean_hist[last_batch:])
            avg_wts_var       = np.mean(wts_var_hist[last_batch:])
            avg_bias_max      = np.mean(bias_max_hist[last_batch:])
            avg_bias_mean     = np.mean(bias_mean_hist[last_batch:])
            avg_bias_var      = np.mean(bias_var_hist[last_batch:])
            avg_wt_udr_max    = np.mean(wt_udr_max_hist[last_batch:])
            avg_wt_udr_mean   = np.mean(wt_udr_mean_hist[last_batch:])
            avg_wt_udr_var    = np.mean(wt_udr_var_hist[last_batch:])
            avg_b_udr_max     = np.mean(bias_udr_max_hist[last_batch:])
            avg_b_udr_mean    = np.mean(bias_udr_mean_hist[last_batch:])
            avg_b_udr_var     = np.mean(bias_udr_var_hist[last_batch:])

            print(
                ' b {:6d}/{:6d} >> {:5.1f} ms/b | lr: {:10.4e} | grad 2norm: {:6.3f} | grad inf norm: {:6.3f}'.format(
                    batch_num+1, num_train_data, runtime, this_lr, avg_grad_2norm, avg_grad_inf_norm
                    
            ))
            print(' '*9 + 'Weights ||',
                'abs max: {:6.3f} ({:9.3f}) | mean: {:10.3e} ({:9.3f}) | var: {:9.3e} ({:9.3f})'.format(
                    avg_wts_max, avg_wt_udr_max,
                    avg_wts_mean, avg_wt_udr_mean,
                    avg_wts_var, avg_wt_udr_var
            ))
            print(' '*10 + 'Biases ||',
                'abs max: {:6.3f} ({:9.3f}) | mean: {:10.3e} ({:9.3f}) | var: {:9.3e} ({:9.3f})'.format(
                    avg_bias_max, avg_b_udr_max,
                    avg_bias_mean, avg_b_udr_mean,
                    avg_bias_var, avg_b_udr_var
            ))
            print(' '*19, 'class loss: {:5.3f} | steps: {:5.3f} | reward: {: 5.3f} | acc.: {:.2%}'.format(
                avg_class_loss, avg_steps, avg_reward, accuracy
            ))

            #'''
            # Check that the gradients of the context network's parameters aren't
            # immediately dying, and are just generally learning well
            if model.context_net is not None:
                cv_p = [p for p in model.context_net.parameters() if p.dim() == 4]
                fc_p = [p for p in model.context_net.parameters() if p.dim() == 2]
                nw_p = [layer.weight for layer in model.context_net.bn_layers]
                nb_p = [layer.bias for layer in model.context_net.bn_layers]
                cv_wts = np.array(torch.cat([p.data.view(-1) for p in cv_p]))
                fc_wts = np.array(torch.cat([p.data.view(-1) for p in fc_p]))
                nw_wts = np.array(torch.cat([p.data.view(-1) for p in nw_p]))
                nb_wts = np.array(torch.cat([p.data.view(-1) for p in nb_p]))
                cv_grad = np.array(torch.cat([p.grad.data.view(-1) for p in cv_p]))
                fc_grad = np.array(torch.cat([p.grad.data.view(-1) for p in fc_p]))
                nw_grad = np.array(torch.cat([p.grad.data.view(-1) for p in nw_p]))
                nb_grad = np.array(torch.cat([p.grad.data.view(-1) for p in nb_p]))
                print('\t| num zero params & grads   ', [
                    (
                        p.size(),
                        int((p == torch.zeros_like(p)).sum().data),
                        int((p.grad == torch.zeros_like(p.grad)).sum().data)
                    ) for p in model.context_net.parameters()
                ])
                num_dead = sum(
                    int(
                        (p == torch.zeros_like(p)).sum().data
                    ) for p in model.context_net.parameters()
                )
                num_grad_dead = sum(
                    int(
                        (p.grad == torch.zeros_like(p.grad)).sum().data
                    ) for p in model.context_net.parameters()
                )
                print(
                    '\t| %s' % ('NUM ZERO: %d' % num_dead if num_dead > 0 else 'no zero params'), ' '*4,
                    '\t\t\t%s' % ('NUM ZERO GRAD: %d' % num_grad_dead if num_grad_dead > 0 else 'no zero grad')
                )
                print(
                    '\t| context net gradient 2norm: %13.6e' % cn_grad_2norm,
                    '\t\t\tinf norm: %13.6e' % cn_grad_inf_norm
                )
                print('\t| =====')
                print(
                    '\t| cv wts avg: %13.6e' % np.mean(cv_wts), '\t\tvar: %13.6e' % np.var(cv_wts),
                    '\tinf norm: %13.6e' % max(p.data.abs().max() for p in cv_p)
                )
                print(
                    '\t| cv grad avg: %13.6e' % np.mean(cv_grad), '\t\tvar: %13.6e' % np.var(cv_grad),
                    '\tinf norm: %13.6e' % max(p.grad.data.abs().max() for p in cv_p)
                )
                print('\t| -----')
                print(
                    '\t| fc wts avg: %13.6e' % np.mean(fc_wts), '\t\tvar: %13.6e' % np.var(fc_wts),
                    '\tinf norm:', max(p.data.abs().max() for p in fc_p)
                )
                print(
                    '\t| fc grad avg: %13.6e' % np.mean(fc_grad), '\t\tvar: %13.6e' % np.var(fc_grad),
                    '\tinf norm: %13.6e' % max(p.grad.data.abs().max() for p in fc_p)
                )
                print('\t| -----')
                print(
                    '\t| bn wts avg: %13.6e' % np.mean(nw_wts), '\t\tvar: %13.6e' % np.var(nw_wts),
                    '\tinf norm: %13.6e' % max(p.data.abs().max() for p in nw_p)
                )
                print(
                    '\t| bn wt grad avg:', np.mean(nw_grad), '\tvar: %13.6e' % np.var(nw_grad),
                    '\tinf norm: %13.6e' % max(p.grad.data.abs().max() for p in nw_p)
                )
                print('\t| -----')
                print(
                    '\t| bn bias avg: %13.6e' % np.mean(nb_wts), '\t\tvar: %13.6e' % np.var(nb_wts),
                    '\tinf norm: %13.6e' % max(p.data.abs().max() for p in nb_p)
                )
                print(
                    '\t| bn bias grad avg: %13.6e' % np.mean(nb_grad), '\tvar: %13.6e' % np.var(nb_grad),
                    '\tinf norm: %13.6e' % max(p.grad.data.abs().max() for p in nb_p)
                )
                print('')
            #'''

            last_ep = len(steps_hist)
            last_batch = batch_num
            start_time = time.time()

        # Visualize the glimpse location sequence for the first batch item only
        if (batch_num+1) % viz_freq == 0 :
            softmax = nn.Softmax(-1)

            # If a context network was used, visualize the downsampled image
            if context_net is not None:
                visualize_context_net(context_net)

            glimpse_seq = torch.cat(glimpse_seq, 0).data
            loc_seq = torch.cat(loc_seq, 0).data
            states_seq = torch.stack(states_seq, 0).data[:, 0]
            predicted_classes = torch.cat(predicted_classes, 0)[:, 0]
            all_pred_probs = softmax(torch.cat(class_probs_seq, 0)[:, 0]).data
            predicted_probs = all_pred_probs.gather(
                -1, predicted_classes.unsqueeze(-1)
            ).squeeze(-1)
            print('\n\tTarget class: %d | Discounted rewards: [%s]' % (
                target_class[0], ', '.join(
                    '%.3f' % r for r in list(discounted_rewards[:, 0])
            )))
            print('\t                | Baseline rewards:   [%s]' % (', '.join(
                '%.3f' % b for b in list(baseline_seq[:, 0].data)
            )))
            img = image[0].permute(1, 2, 0)
            glimpse_visualization(
                img, glimpse_seq[:, 0], loc_seq[:, 0], padding, image_size,
                target_class[0], predicted_classes, predicted_probs,
                states = states_seq, mean_locs = mean_loc_seq[:,0].data,
                sampled_locs = sampled_loc_seq[:,0].data, explored = explore_seq
            )
        if (batch_num+1) % log_freq == 0:
            print(' '*20 + '-'*80, '\n')
    ### End of episode for loop ###

    ########## EPOCH METRICS ##########
    # Calculate some metrics to return
    averages = {
        'accuracy'     : np.mean(success_hist),
        'steps'        : np.mean(steps_hist),
        'reward'       : np.mean(reward_hist),
        'class_loss'   : np.mean(class_loss_hist),
        'grad_inf_norm': np.mean(grad_inf_norm_hist),
        'grad_2norm'   : np.mean(grad_2norm_hist),
        #'weights_max'  : np.mean(wts_max_hist),
        #'weights_mean' : np.mean(wts_mean_hist),
        #'weights_var'  : np.mean(wts_var_hist),
        #'bias_max'     : np.mean(bias_max_hist),
        #'bias_mean'    : np.mean(bias_mean_hist),
        #'bias_var'     : np.mean(bias_var_hist),
        #'weight_update_max': np.mean(wt_udr_max_hist),
        #'weight_update_mean': np.mean(wt_udr_mean_hist),
        #'weight_update_var': np.mean(wt_udr_var_hist),
        #'bias_update_max': np.mean(bias_udr_max_hist),
        #'bias_update_mean': np.mean(bias_udr_mean_hist),
        #'bias_update_var': np.mean(bias_udr_var_hist),
    }
    histories = {
        'accuracy'     : success_hist,
        'steps'        : steps_hist,
        'reward'       : reward_hist,
        'class_loss'   : class_loss_hist,
        'grad_inf_norm': grad_inf_norm_hist,
        'grad_2norm'   : grad_2norm_hist,
        #'weights_max'  : wts_max_hist,
        #'weights_mean' : wts_mean_hist,
        #'weights_var'  : wts_var_hist,
        #'bias_max'     : bias_max_hist,
        #'bias_mean'    : bias_mean_hist,
        #'bias_var'     : bias_var_hist,
        #'weight_update_max': wt_udr_max_hist,
        #'weight_update_mean': wt_udr_mean_hist,
        #'weight_update_var': wt_udr_var_hist,
        #'bias_update_max': bias_udr_max_hist,
        #'bias_update_mean': bias_udr_mean_hist,
        #'bias_update_var': bias_udr_var_hist,
    }
    return averages, histories

def eval(model, eval_data, image_size, padding, class_criterion,
        use_full_class_seq = False, error_penalty = 0., context_net = None,
        reward_all_correct = False, random_start_loc = True, max_ep_len = 30,
        gamma = 0.99, end_train_ep_early = False, viz_freq = 1000):
    # For metrics
    steps_hist = []
    reward_hist = []
    class_loss_hist = []
    success_hist = []

    model.eval()
    #if context_net is not None:
    #    context_net.eval()

    for batch_num, (image, target_class) in enumerate(eval_data):
        batch_size = image.size(0)
        assert (end_train_ep_early and batch_size == 1) or not end_train_ep_early

        glimpse_seq           = []
        loc_seq               = []
        reward_seq            = []
        class_probs_seq       = []
        predicted_classes     = []
        states_seq            = []

        model.reset_dp_masks()
        if model.context_net is None:
            rnn_states = model.init_rnn_states(batch_size)
            loc_means = torch.from_numpy(
                np.random.uniform(-1, 1, size = (1, batch_size, 2))
            ).type(torch.FloatTensor) if random_start_loc else torch.zeros(1, batch_size, 2)
            loc_means = Variable(loc_means)
        else:
            rnn_states, loc_means = model.get_context_loc(image, batch_size)
        loc = loc_means.clamp(-1, 1)
        # Number of steps to the first successful classification or max_ep_len
        steps = [None for _ in range(batch_size)]
        for t in range(max_ep_len):
            loc_seq.append(loc)

            # Step forward
            class_probs, loc_means, _, rnn_states, glimpses, _ = model(
                loc, rnn_states, imgs = image
            )
            class_probs_seq.append(class_probs)

            # When evaluating, take the glimpse location coordinates as-is, i.e.
            # without sampling. Still clamp them to be between (-1, 1), though
            loc = loc_means.clamp(-1, 1)

            # Predicted class
            _, predicted_class = torch.max(class_probs, dim = -1)
            predicted_class = predicted_class.data
            successful = predicted_class == target_class
            
            # Give a reward or penalty for this step
            reward = successful.type(torch.FloatTensor)
            if end_train_ep_early and successful.squeeze(0)[0]:
                reward.fill_(1.)
            elif reward_all_correct or t == max_ep_len - 1:
                reward[predicted_class != target_class] = error_penalty
            else:
                reward.fill_(error_penalty)
            successful = successful.squeeze(0)

            # Append to the episode histories
            if isinstance(rnn_states[0], tuple):
                if isinstance(rnn_states[0][0], tuple):
                    states_seq.append(rnn_states[0][0][0])
                else:
                    states_seq.append(rnn_states[0][0])
            else:
                states_seq.append(rnn_states[0])
            glimpse_seq.append(glimpses)
            reward_seq.append(reward)
            predicted_classes.append(predicted_class)

            for b in range(batch_size):
                if successful[b]:
                    steps[b] = t + 1 if steps[b] is None else steps[b]
            if end_train_ep_early and successful[0]:
                break

        ### End of while loop ###
        reward_seq = torch.cat(reward_seq, 0)
        total_reward = reward_seq.sum(0)
        steps = [max_ep_len if s is None else s for s in steps]

        # Get the classification losses
        if use_full_class_seq:
            _class_probs_seq_ = torch.cat(class_probs_seq, 0)
            num_classes = _class_probs_seq_.size(-1)

            _class_probs_seq_ = _class_probs_seq_.view(-1, num_classes)
            target_classes = target_class.expand(max_ep_len, -1).contiguous().view(-1)
            class_loss = class_criterion(_class_probs_seq_, target_classes)
        else:
            class_loss = class_criterion(
                class_probs_seq[-1].squeeze(0),
                Variable(target_class.view(-1))
            )

        # Discount rewards
        discounted_rewards = discount_rewards(reward_seq, gamma)

        ### Metrics, etc. ###
        # Metrics, logging, and visualizations
        steps_hist.extend(steps)
        reward_hist.extend(list(total_reward))
        success_hist.extend(list(successful.type(torch.FloatTensor)))
        class_loss_hist.append(float(class_loss))

        if (batch_num+1) % viz_freq == 0:
            softmax = nn.Softmax(-1)

            # If a context network was used, visualize the downsampled image
            if context_net is not None:
                visualize_context_net(context_net)

            glimpse_seq = torch.cat(glimpse_seq, 0).data
            loc_seq = torch.cat(loc_seq, 0).data
            states_seq = torch.stack(states_seq, 0).data[:, 0]
            predicted_classes = torch.cat(predicted_classes, 0)[:, 0]
            all_pred_probs = softmax(torch.cat(class_probs_seq, 0)[:, 0]).data
            predicted_probs = all_pred_probs.gather(
                -1, predicted_classes.unsqueeze(-1)
            ).squeeze(-1)
            print('\n\tTarget class: %d | Discounted rewards: [%s]' % (
                target_class[0], ', '.join(
                    '%.3f' % r for r in list(discounted_rewards[:, 0])
            )))
            img = image[0].permute(1, 2, 0)
            glimpse_visualization(
                img, glimpse_seq[:, 0], loc_seq[:, 0], padding, image_size,
                target_class[0], predicted_classes, predicted_probs,
                states = states_seq
            )
    ### End of episode loop ###
    averages = {
        'accuracy'  : np.mean(success_hist),
        'steps'     : np.mean(steps_hist),
        'reward'    : np.mean(reward_hist),
        'class_loss': np.mean(class_loss_hist),
    }
    histories = {
        'accuracy'  : success_hist,
        'steps'     : steps_hist,
        'reward'    : reward_hist,
        'class_loss': class_loss_hist,
    }
    return averages, histories

def train_eval_loop(model, epochs, train_data, val_data, test_data,
        image_size, padding, optimizer, lr_scheduler, class_criterion,
        use_full_class_seq = False, context_net = None, loc_std = 1.,
        epsilon = 0., epsilon_min = 0., error_penalty = 0., gamma = 0.99, 
        reward_all_correct = False, num_explore_epochs = 0, max_ep_len = 30,
        num_eps_anneal_epochs = 1, random_start_loc = True, early_stopping = 0, 
        end_train_ep_early = False, grad_clip = 1., steps_per_epoch = 1,
        ckpt = None, log_freq = 1000, viz_freq = 1000, eval_viz_freq = 1000
        ):
    '''
    `reward_all_correct`: if `True`, awards a point for each step where the agent
    classifies the image correctly. Defaults to `False`.

    `use_full_class_seq`: if `True`, calculates the classification error using
    the classification output from each step, meaning we accumulate gradients
    from each step, too. Defaults to `False`.

    `context_net`: if not `None`, the agent uses the context network to learn
    an initial glimpse location (rather than picking one randomly or always
    starting at the origin), and `random_start_loc` is ignored

    `end_train_ep_early`: if `True`, each episode during training is ended early
    if the agent makes a correct classification; training must be done with a
    batch size of 1. Defaults to `False`.

    `steps_per_epoch`: number of times to step the LR scheduler each epoch (NOT
    the number of steps over all episodes in the epoch...)

    `ckpt`: if a string, indicates the filepath to where saved model checkpoints
    should go
    '''
    WIDTH = 120

    model_info = {
        'weights': {
            'mean': [],
            'abs max': [],
            'var': []
        },
        'biases': {
            'mean': [],
            'abs max': [],
            'var': []
        }
    }
    train_info = {'stats': []}
    val_info = {'stats': []}
    best_stats = ['steps', 'reward', 'accuracy']
    best_train_stats = {stat: None for stat in best_stats}
    best_train_epochs = {stat: 0 for stat in best_stats}
    better_train = {stat: False for stat in best_stats}
    best_val_stats = {stat: None for stat in best_stats}
    best_val_epochs = {stat: 0 for stat in best_stats}
    better_val = {stat: False for stat in best_stats}
    better = False

    eps = epsilon
    d_eps = (epsilon - epsilon_min) / num_eps_anneal_epochs

    for epoch in range(epochs):
        lr_scheduler.step()
        print('Epoch {:3d}/{}) lr = {:.4g}'.format(
            epoch+1, epochs, np.mean(lr_scheduler.get_lr())
        ))

        # Get the exploration rates
        eps_end = epsilon if epoch < num_explore_epochs else max(epsilon_min, eps - d_eps)

        # Run training and validation
        start_time = time.time()
        train_averages, train_histories = train(
            model = model, train_data = train_data, image_size = image_size,
            padding = padding, optimizer = optimizer, lr_scheduler = lr_scheduler,
            class_criterion = class_criterion, context_net = context_net,
            max_ep_len = max_ep_len, use_full_class_seq = use_full_class_seq,
            reward_all_correct = reward_all_correct, gamma = gamma,
            error_penalty = error_penalty, random_start_loc = random_start_loc,
            loc_std = loc_std, epsilon = eps, epsilon_min = eps_end,
            grad_clip = grad_clip, end_train_ep_early = end_train_ep_early,
            viz_freq = viz_freq, steps_per_epoch = steps_per_epoch,
            log_freq = log_freq,
        )
        elapsed_time = time.time() - start_time  # Just want training time
        train_info['stats'].append({
            'averages': train_averages,
            'histories': train_histories
        })

        VAL_TITLE_STR = '---===  VALIDATION  ===---'
        print(' ' * (  (WIDTH - len(VAL_TITLE_STR))//2  ) + VAL_TITLE_STR)
        val_averages, val_histories = eval(
            model = model, eval_data = val_data, image_size = image_size,
            padding = padding, class_criterion = class_criterion, gamma = gamma,
            error_penalty = error_penalty, random_start_loc = random_start_loc,
            use_full_class_seq = use_full_class_seq, context_net = context_net,
            end_train_ep_early = end_train_ep_early, max_ep_len = max_ep_len,
            reward_all_correct = reward_all_correct, viz_freq = eval_viz_freq
        )
        val_info['stats'].append({
            'averages': val_averages,
            'histories': val_histories
        })

        eps = eps_end

        # Early stopping
        for stat in best_stats:
            for bests, best_epochs, avgs, betters in zip(
                    [best_train_stats, best_val_stats],
                    [best_train_epochs, best_val_epochs],
                    [train_averages, val_averages],
                    [better_train, better_val]):
                new_stat = avgs[stat]; best_stat = bests[stat]
                if best_stat is None or (new_stat <= best_stat if stat == 'steps' else new_stat >= best_stat):
                    bests[stat] = new_stat
                    best_epochs[stat] = epoch
                    betters[stat] = True
        better = any(better_train.values()) or any(better_val.values())

        # Metrics, diagnostics, etc.
        params = [p for p in model.parameters() if p.grad is not None]
        weights = np.array(torch.cat([
            w.data.view(-1) for w in params if w.dim() > 1
        ]))
        biases = np.array(torch.cat([
            b.data.view(-1) for b in params if b.dim() == 1
        ]))
        model_info['weights']['abs max'].append(max(np.abs(weights)))
        model_info['weights']['mean'].append(np.mean(weights))
        model_info['weights']['var'].append(np.var(weights))
        model_info['biases']['abs max'].append(max(np.abs(biases)))
        model_info['biases']['mean'].append(np.mean(biases))
        model_info['biases']['var'].append(np.var(biases))

        DIAG_TITLE_STR = '---===  MODEL PARAM DIAGNOSTICS  ===---'
        print(' ' * (  (WIDTH - len(DIAG_TITLE_STR))//2  ) + DIAG_TITLE_STR)
        # Model weights/biases histogram
        visualize_model_parameters(model)

        # Visualize the weights from the location network
        LOCNET_PARAMS_STR = 'Location network weights and biases'
        print(' ' * ((WIDTH - len(LOCNET_PARAMS_STR))//2) + LOCNET_PARAMS_STR)
        fig, axs = plt.subplots(1, 2, figsize = (12, 2.5))
        loc_wt_min = model.loc_net.weight.data.min()
        loc_wt_max = model.loc_net.weight.data.max()
        wt_im = axs[0].imshow(model.loc_net.weight.data, aspect = 'auto', cmap = CMAP)
        axs[0].set_title(
            'Weights (min: {:.3e}, max: {:.3e})'.format(loc_wt_min, loc_wt_max)
        )
        axs[0].set_yticks([0, 1]); axs[0].set_yticklabels(['y', 'x'])
        fig.colorbar(wt_im, ax = axs[0])
        loc_b_min = model.loc_net.bias.data.min()
        loc_b_max = model.loc_net.bias.data.max()
        axs[1].imshow(model.loc_net.bias.data.unsqueeze(1), aspect = 'equal', cmap = CMAP)
        axs[1].set_title(
            'Biases (min: {:.3e}, max: {:.3e})'.format(loc_b_min, loc_b_max)
        )
        axs[1].set_yticks([0, 1]); axs[1].set_yticklabels(['y', 'x'])
        axs[1].set_xticks([])
        fig.tight_layout()
        plt.show()

        # Epoch metrics printout
        print('-' * WIDTH)
        print(
            'Elapsed time: {:7.2f} sec | TRAIN >> class loss: {:6.4f} | steps: {:6.3f} {} | reward: {: 6.4f} {} | acc.: {:.3%} {}'.format(
                elapsed_time, train_averages['class_loss'],
                train_averages['steps'], '  ' if better_train['steps'] else ':(',
                train_averages['reward'], '  ' if better_train['reward'] else ':(',
                train_averages['accuracy'], '' if better_train['accuracy'] else ':('
        ))
        print(
            ' '*26 + '| VAL   >> class loss: {:6.4f} | steps: {:6.3f} {} | reward: {: 6.4f} {} | acc.: {:.3%} {}'.format(
                val_averages['class_loss'],
                val_averages['steps'], '  ' if better_val['steps'] else ':(',
                val_averages['reward'], '  ' if better_val['reward'] else ':(',
                val_averages['accuracy'], '' if better_val['accuracy'] else ':('
        ))
        print('=' * WIDTH)
        print('\n')
        if better:
            stagnant = 0
            better_train = {stat: False for stat in best_stats}
            better_val = {stat: False for stat in best_stats}
            better = False
            if ckpt is not None:
                torch.save(model, ckpt)
        else:
            stagnant += 1
            if stagnant >= early_stopping and early_stopping > 0:
                break

    # Evaluate the trained model on the test set
    print('\n\t\t\t---~~~=== End of training ===~~~---\n')
    print('Evaluating on the TEST dataset...')
    test_averages, test_histories = eval(
        model = model, eval_data = test_data, image_size = image_size,
        padding = padding, class_criterion = class_criterion, gamma = gamma,
        use_full_class_seq = use_full_class_seq, error_penalty = error_penalty,
        random_start_loc = random_start_loc, context_net = context_net,
        end_train_ep_early = end_train_ep_early, max_ep_len = max_ep_len,
        reward_all_correct = reward_all_correct, viz_freq = eval_viz_freq
    )
    print(
        '\t>> class loss: {:6.4f} | steps: {:6.3f} | reward: {: 6.4f} | acc.: {:.3%}'.format(
            test_averages['class_loss'], test_averages['steps'],
            test_averages['reward'], test_averages['accuracy'],
    ))

    test_info = {
        'averages': test_averages,
        'histories': test_histories
    }
    # Attach the best train/val stats
    train_info['bests'] = {
        stat: {
            stat: best_train_stats[stat],
            'epoch': best_train_epochs[stat] + 1
        } for stat in best_stats
    }
    val_info['bests'] = {
        stat: {
            stat: best_val_stats[stat],
            'epoch': best_val_epochs[stat] + 1
        } for stat in best_stats
    }
    return model_info, train_info, val_info, test_info, epoch+1

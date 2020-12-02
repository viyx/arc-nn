import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def sample_while(model, dataloader):
    model.eval()
    steps = model.target_size - 1 # exclude end_episode_token
    results = []
    for x, _ in dataloader:
        # x = x.to(model.device)
        x_now = x[:, :-steps] # (b, 3)

        for step in tqdm.trange(steps):
            # print('start')
            # x = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
            # print(x.shape)
            logits, _ = model(x_now)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            _, ix = torch.topk(probs, k=1, dim=-1)
            x_now = torch.cat((x_now, ix), dim=1)
            # break
        x_now = x_now[:,-steps:].view(-1, 30, 31).detach().cpu()
        results.append(x_now)
        break
        # print('break')
    results = torch.vstack(results)
    return results


# @torch.no_grad()
# def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
#     """
#     take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
#     the sequence, feeding the predictions back into the model each time. Clearly the sampling
#     has quadratic complexity unlike an RNN that is only linear, and has a finite context window
#     of block_size, unlike an RNN that has an infinite context window.
#     """
#     block_size = model.get_block_size()
#     model.eval()
#     for k in range(steps):
#         x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
#         logits, _ = model(x_cond)
#         # pluck the logits at the final step and scale by temperature
#         logits = logits[:, -1, :] / temperature
#         # optionally crop probabilities to only the top k options
#         if top_k is not None:
#             logits = top_k_logits(logits, top_k)
#         # apply softmax to convert to probabilities
#         probs = F.softmax(logits, dim=-1)
#         # sample from the distribution or take the most likely
#         if sample:
#             ix = torch.multinomial(probs, num_samples=1)
#         else:
#             _, ix = torch.topk(probs, k=1, dim=-1)
#         # append to the sequence and continue
#         x = torch.cat((x, ix), dim=1)

#     return x


# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib import colors

# def plot_one(ax, data , train_or_test, input_or_output):
#     main_colors = ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
#          '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']

#         # black, blue, red, green, yellow, gray, magenta, orange, sky, blood

#         # self.end_line_token = n_colors + 0    # array of shape (10, 3) has 10 end_lines
#         # self.promt_token = n_colors + 1       # promt after every x
#         # self.end_episode_token = n_colors + 2
#         # self.pad_token = n_colors + 3

#         # deepblue, 
#     special_colors = ['cyan', 'beige', 'ghostwhite', 'lightcoral']
#     cmap = colors.ListedColormap(main_colors + special_colors)
#     norm = colors.Normalize(vmin=0, vmax = \
#         (len(main_colors) + len(special_colors)) - 1)
    
#     ax.imshow(data, cmap=cmap, norm=norm)
#     ax.grid(True, which='both', color='lightgrey', linewidth=0.5)    
#     ax.set_yticks([x-0.5 for x in range(1+len(data))])
#     ax.set_xticks([x-0.5 for x in range(1+len(data[0]))])     
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_title(train_or_test + ' ' + input_or_output)


# def plot_task(arcdataset, taskname, predictions):
#     """
#     Plots the first train and test pairs of a specified task,
#     using same color scheme as the ARC app
#     """
#     x_train, y_train, x_test, y_test = arcdataset[taskname]
#     num_train = len(x_train)

#     chunk1 = zip(x_train, y_train,)
#     chunk2 = zip([x_test[0], x_test[0]], [y_test[0], predictions])
#     chunks = [chunk1, chunk2]

#     # fig = plt.figure(figsize=(2*3*num_train,2*3*2))
#     # fig = plt.figure(figsize=(18, 12))
#     fig = plt.figure()
#     gs = gridspec.GridSpec(nrows=len(chunks) * 2, ncols=num_train)

#     for i, chunk in enumerate(chunks):
#         for j, (x,y) in enumerate(chunk):
#             ax1 = fig.add_subplot(gs[2*i, j])
#             ax2 = fig.add_subplot(gs[2*i+1, j])
#             plot_one(ax1, x, str(2*i) + '_' + str(j), 'input')
#             plot_one(ax2, y, str(2*i+1) + '_' + str(j), 'output')
    
#     return fig
from mingpt.utils import set_seed
set_seed(44)

import math
import time
import numpy as np
from copy import deepcopy
import pickle
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from torch.utils.data import Subset
from tqdm import tqdm
from matplotlib import pyplot as plt

# from data import get_othello, plot_probs, plot_mentals
# from data.othello import permit, start_hands, OthelloBoardState, permit_reverse
import Checkers
from mingpt.dataset import CharDatasetNoMoveType
from mingpt.model import GPT, GPTConfig, GPTforProbeIA
from mingpt.utils import sample, intervene, print_board
from mingpt.probe_model import BatteryProbeClassification, BatteryProbeClassificationTwoLayer

# plt.rc('text', usetex=True)
plt.rc('font', **{'size': 14.0})
# plt.rc('text.latex', preamble=r'\usepackage{lmodern}')

mid_dim = 128
# how_many_history_step_to_use = 99
exp = f"state_tl{mid_dim}"

probes = {}
layer_s = 5
layer_e = 9
for layer in range(layer_s, layer_e):
    p = BatteryProbeClassificationTwoLayer(torch.cuda.current_device(), probe_class=5, num_task=32, mid_dim=mid_dim)
    load_res = p.load_state_dict(torch.load(f"C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers2/ckpts/battery_checkers/{exp}/layer{layer}/checkpoint.ckpt"))
    p.eval()
    probes[layer] = p

testDataset = CharDatasetNoMoveType(True, True)

mconf = GPTConfig(testDataset.vocab_size, testDataset.block_size, n_layer=8, n_head=8, n_embd=512)
championship= False
models = {}
for layer in range(layer_s, layer_e):
    model = GPTforProbeIA(mconf, probe_layer=layer)
    # model = GPT(mconf)
    load_res = model.load_state_dict(torch.load("C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers2/ckpts/gpt_no_move_type.ckpt" if not championship else "./ckpts/gpt_championship.ckpt"))
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = model.to(device)
    _ = model.eval()
    models[layer] = model

def states_to_piece_positions(game_env, next_states):
    """Given a list of next states, produce a list of two coordinates for each
    possible next state.  The first coordinate will be the location of the
    piece that was moved, and the second coordinate will be the location that
    the piece moved to.
    """
    moves_list = []
    state = game_env.state
    board = state[0] + 2*state[1] + 3*state[2] + 4*state[3]
    for nstate in next_states:
        nboard = nstate[0] + 2*nstate[1] + 3*nstate[2] + 4*nstate[3]
        board_diff = board - nboard
        xnew, ynew = np.where(board_diff < 0)
        xnew, ynew = xnew[0], ynew[0]
        new_val = abs(nboard[xnew,ynew])
        xold, yold = np.where(board_diff == new_val)
        try:
            xold, yold = xold[0], yold[0]
        except IndexError: # Man promoted to king
            new_val -= 1 # Value of man is 1 less than king
            xold, yold = np.where(board_diff == new_val)
            xold, yold = xold[0], yold[0]
        moves_list.append([(xold+1,yold+1),(xnew+1,ynew+1)])
    return moves_list

pos_dict = {(1,2):1, (1,4):2, (1,6):3, (1,8):4, (2,1):5, (2,3):6, (2,5):7, (2,7):8, (3,2):9, (3,4):10, (3,6):11, (3,8):12,(4,1):13, (4,3):14, (4,5):15, (4,7):16,
            (5,2):17, (5,4):18, (5,6):19, (5, 8):20, (6,1):21, (6,3):22, (6,5):23, (6,7):24, (7,2):25, (7,4):26, (7,6):27, (7,8):28, (8,1):29, (8,3):30, (8,5):31, (8,7):32}

reverse_pos_dict = {v:k for k,v in pos_dict.items()}
def translate_move(m):
    return pos_dict[m[0]], pos_dict[m[1]]

def probs_to_board(dist, itos=True, filler=0):
    board = np.ones((8,8))*filler
    for i, p in enumerate(dist):
        num = i + 1
        if itos:
            num = int(testDataset.itos[i])
        y, x = reverse_pos_dict[num]
        board[y-1,x-1] = p
    return board

def plot_probs(ax, probs):
    assert probs.numel() == 32
    probs = probs_to_board(probs)
    annot = [f"{_:.2f}" for _ in probs.flatten().tolist()]
    # for valid_index in valids:
    #     annot[valid_index] = ("\\underline{" + annot[valid_index] + "}")
#     print(annot)
    sns.heatmap(probs, ax=ax, vmin=0, vmax=vv, square=True,
            annot=np.array(annot).reshape(8, 8), cmap=sns.color_palette("Blues", as_cmap=True), fmt="", cbar=False)
    return ax

def plot_mentals(ax, logits):
    assert logits.shape[0] == 32
    assert logits.shape[1] == 5
    probs = torch.softmax(logits, dim=-1)  # [32, 5]
    probs, preds = torch.max(probs, dim=-1)  # [32, ], [32, ]
    # probs = probs.detach().cpu().numpy().reshape(8, 8)
    probs = probs_to_board(probs, itos=False, filler=4)
    preds = probs_to_board(preds, itos=False, filler=4)
    annot = []
    for ele in preds.flatten().tolist():
        if ele == 0:
            annot.append(game_env.player2_man)
        elif ele == 1:
            annot.append(game_env.player2_king)
        elif ele == 2:
            annot.append(game_env.player1_man)
        elif ele == 3:
            annot.append(game_env.player1_king)
        else:
            annot.append(" ")
    sns.heatmap(probs, ax=ax, vmin=0, vmax=1., square=True,
            annot=np.array(annot).reshape(8, 8), cmap=sns.color_palette("Blues", as_cmap=True), fmt="", cbar=False)
    return ax

partial_game = [testDataset.itos[i.item()] for i in testDataset[3][0][:57]]
print(partial_game)
game_env = Checkers.Checkers()
# update game environment with the partial game's moves
game_env.get_gt(partial_game, "get_state")
game_env.print_board()
legal_next_states = game_env.legal_next_states
pre_intv_valids = states_to_piece_positions(game_env, legal_next_states)
pre_intv_valids = [translate_move(move) for move in pre_intv_valids]
print(pre_intv_valids)

# turn white uncrowned to black uncrowned

wtd = {
    "intervention_position": '17',
    "intervention_from": 2,
    "intervention_to": 4,
}

wtd_list = [wtd]
wtd = {
    "intervention_position": '18',
    "intervention_from": 4,
    "intervention_to": 2,
}
wtd_list.append(wtd)


partial_game = testDataset[3][0][:57].to(device)
pre_intv_pred, _ = model(partial_game[None, :])
pre_intv_pred = pre_intv_pred[0, -1, :-2]
pre_intv_pred = torch.softmax(pre_intv_pred, dim=0)
dist = probs_to_board(pre_intv_pred)
# pre_intv_pred = torch.cat([pre_intv_pred[:27], padding, pre_intv_pred[27:33], padding, pre_intv_pred[33:]], dim=0)
fig=plt.figure(figsize=(10, 6), dpi= 80, facecolor='w', edgecolor='k')
vv = 0.5
sns.heatmap(dist, vmin=0., vmax=vv, square=True, annot=True, fmt=".2f")
plt.show()

htd = {"lr": 1e-3, "steps": 3000, "reg_strg": 0.2}
for wtd in wtd_list:
    move = int(wtd["intervention_position"])
    r, c = reverse_pos_dict[move]
    game_env.state[wtd["intervention_from"], r-1, c-1] = 0
    game_env.state[wtd["intervention_to"], r-1, c-1] = 1
    game_env.history[-1] = game_env.state
    game_env.legal_next_states = game_env._check_moves(game_env.history, (9,9))
    if wtd["intervention_to"] <= 1:
        wtd["intervention_to"] +=2
    elif wtd["intervention_to"] < 4:
        wtd["intervention_to"] -=2
print("\nPost intervention:")
game_env.print_board()
legal_next_states = game_env.legal_next_states
post_intv_valids = states_to_piece_positions(game_env, legal_next_states)
post_intv_valids = [translate_move(move) for move in post_intv_valids]
print(post_intv_valids)

fig, axs = plt.subplots(layer_e - layer_s, 2, figsize=(8 * (1), 8 * (layer_e - layer_s)), dpi=80, facecolor='w',
                        edgecolor='k')
# two rows for the intervened layer layer_s, one for the rest
if len(axs.shape) == 1:
    axs = axs[:, None]

p = probes[layer_s]
whole_mid_act = models[layer_s].forward_1st_stage(partial_game[None, :])  # [B, T, F=512]

# intervene at the earlest interested layer
mid_act = whole_mid_act[0, -1]
pre_intv_logits = p(mid_act[None, :])[0].squeeze(0)  # [32, 5]
plot_mentals(axs[0, 0], pre_intv_logits)
axs[0, 0].set_title(f"Pre-intervention Probe Result \n at the {layer_s}-th Layer")
# plt.show()
labels_pre_intv = pre_intv_logits.detach().argmax(dim=-1)
new_mid_act = mid_act.clone()
for wtd in wtd_list:
    new_mid_act = intervene(testDataset, p, new_mid_act, labels_pre_intv, wtd, htd, plot=True)
    pre_intv_logits = p(new_mid_act[None, :])[0].squeeze(0)  # [32, 5]
    labels_pre_intv = pre_intv_logits.detach().argmax(dim=-1)
post_intv_logits = p(new_mid_act[None, :])[0].squeeze(0)  # [32, 5]
plot_mentals(axs[0, 1], post_intv_logits)
axs[0, 1].set_title(f"Post-intervention Probe Result \n at the {layer_s}-th Layer")
# swap in
whole_mid_act[0, -1] = new_mid_act

for i, layer in enumerate(range(layer_s, layer_e - 1)):  # 4, 5, 6, 7, indices of the layers to be passed
    p = probes[layer + 1]
    whole_mid_act = models[layer_s].forward_2nd_stage(whole_mid_act, layer, layer + 1)[0]  # [1, T, F=512]

    # intervene the output of the features freshly out
    mid_act = whole_mid_act[0, -1]
    pre_intv_logits = p(mid_act[None, :])[0].squeeze(0)  # [64, 3]
    plot_mentals(axs[i + 1, 0], pre_intv_logits)
    axs[i + 1, 0].set_title(f"Post-intervention Probe Result \n at the {layer + 1}-th Layer")
    labels_pre_intv = pre_intv_logits.detach().argmax(dim=-1)
    new_mid_act = mid_act.clone()
    for wtd in wtd_list:
        new_mid_act = intervene(testDataset, p, new_mid_act, labels_pre_intv, wtd, htd, plot=True)
        pre_intv_logits = p(new_mid_act[None, :])[0].squeeze(0)  # [64, 3]
        labels_pre_intv = pre_intv_logits.detach().argmax(dim=-1)
    post_intv_logits = p(new_mid_act[None, :])[0].squeeze(0)  # [64, 3]
    plot_mentals(axs[i + 1, 1], post_intv_logits)
    axs[i + 1, 1].set_title(f"Post-intervention Probe Result \n at the {layer + 1}-th Layer")
    # swap in
    whole_mid_act[0, -1] = new_mid_act
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(20, 6), dpi= 80, facecolor='w', edgecolor='k')

plot_probs(axs[0], pre_intv_pred)
axs[0].set_title(f"Pre-intervention Prediction Heatmap")

tb_resumed = whole_mid_act
post_intv_pred, _ = models[layer_s].predict(tb_resumed)
post_intv_pred = post_intv_pred[0, -1, :-2]
post_intv_pred = torch.softmax(post_intv_pred, dim=0)
# post_intv_pred = torch.cat([post_intv_pred[:27], padding, post_intv_pred[27:33], padding, post_intv_pred[33:]], dim=0)

plot_probs(axs[1], post_intv_pred)
axs[1].set_title(f"Post-intervention Prediction Heatmap")
plt.show()
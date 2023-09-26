import copy

from mingpt.utils import set_seed
set_seed(44)
import os
import math
import time
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
# from data import get_othello
# from data.othello import permit, start_hands, OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mingpt.utils import sample
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
import Checkers
# from play_Checkers import states_to_piece_positions
test_dataset = CharDataset(kings=True, test=True)
mconf = GPTConfig(test_dataset.vocab_size, test_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

load_res = model.load_state_dict(torch.load("C:/Users/Arik/Documents/uni/NLP/Project/NLPProjectCheckers/ckpts/gpt_base_dataset.ckpt"))
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = model.to(device)
model = model.eval()
total_nodes = 0
success_nodes = 0

def is_Legal(prev_prev_word, prev_word, new_word):
    l = [str(i) for i in range(1, 33)]
    # print(l.extend('K'))
    if new_word == -100:
        return False

    if prev_prev_word == 'K':
        if prev_word in l:
            if new_word not in ['x','-']:
                return False
    elif prev_prev_word in l:
        if prev_word in l:
            if new_word in l or new_word == 'K':
                return False
    elif prev_prev_word == '-':
        if prev_word in l:
            if new_word in ['-','x']:
                return False
    elif prev_prev_word == 'x':
        if prev_word in l:
            if new_word == '-':
                return False

    if prev_word == '-':
        if new_word in ['x','-','K']:
            return False
    elif prev_word == 'x':
        if new_word in ['x','-','K']:
            return False
    elif prev_word == 'K':
        if new_word in ['x','-','K']:
            return False

    return True

pos_dict = {(1,2):1, (1,4):2, (1,6):3, (1,8):4, (2,1):5, (2,3):6, (2,5):7, (2,7):8, (3,2):9, (3,4):10, (3,6):11, (3,8):12,(4,1):13, (4,3):14, (4,5):15, (4,7):16,
            (5,2):17, (5,4):18, (5,6):19, (5, 8):20, (6,1):21, (6,3):22, (6,5):23, (6,7):24, (7,2):25, (7,4):26, (7,6):27, (7,8):28, (8,1):29, (8,3):30, (8,5):31, (8,7):32}

reverse_pos_dict = {v:k for k,v in pos_dict.items()}
def translate_move(m):
    return pos_dict[m[0]], pos_dict[m[1]]

def words_list_to_move(words):
    if words[0] == 'K':
        words = words[1:]
    l = []
    for word in words:
        if word in [str(i) for i in reverse_pos_dict.keys()]:
            l.append(reverse_pos_dict[int(word)])
    return l
# 2-4

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

def check_legal_move(game_env, move):
    # moves_str = ""
    # moves_str_without_K = ""
    last = -18
    equals = False

    l = words_list_to_move(move)
    # l = (l[0], l[1])
        # _, m, eat, done = get_random_input()
    legal_next_states = game_env.legal_next_states
    moves_list = states_to_piece_positions(game_env, legal_next_states)
    player_before = int(game_env.state[4, 0, 0])
    if l in moves_list:
        id = moves_list.index(l)
        a, b, done = game_env.step(legal_next_states[id], (9,9))
        player_after = int(game_env.state[4, 0, 0])
        if abs(l[0][0] - l[1][0]) == 2:
            # eat = True
            if move[-2] == 'x':
                return True, player_before != player_after
            else:
                return False, False
        else:
            # eat = False
            if move[-2] == '-':
                return True, player_before != player_after
            else:
                return False, False
    else:
        return False, False


def get_real_move(next_words):
    prev = ''
    l = [str(i) for i in range(1, 33)]
    for id, word in enumerate(next_words):
        if (word in l or word == 'K') and prev in l:
            return id
        prev = word
    return -1


syntax_error_counter = 0
semantic_error_counter = 0
x_instead_of_line_counter = 0
line_instead_of_x_counter = 0
error_move_number = []
king_guesses = [] #(move numbers, whether king exists)
moves_per_game = []
guessed_padding = 0


bar = tqdm(test_dataset.data[:1000])

for whole_game in bar:
    game_env_global = Checkers.Checkers(neural_net=None)

    game_env_local = Checkers.Checkers(neural_net=None)
    length_of_whole_game = len(whole_game)
    prev_prev_word = ""
    prev_word = whole_game[0]
    length_of_partial_game = 1
    get_true_game = True
    move = []
    start_index = 1
    while length_of_partial_game < length_of_whole_game:
    # for length_of_partial_game in range(1, length_of_whole_game):


        if get_true_game == True:
            total_nodes += 1
            context = whole_game[:length_of_partial_game]
            length_of_partial_game += 1
        else:
            context = completion
        x = torch.tensor([test_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(device)
        y = sample(model, x, 1, temperature=1.0)[0]
        # print(y)
        new_word = test_dataset.itos[int(y[-1])]
        completion = [test_dataset.itos[int(i)] for i in y]
        if new_word == -100:
            guessed_padding += 1
        if new_word == 'K':
            white_kings = game_env_local.state[3]
            black_kings = game_env_local.state[1]
            king_guesses.append((len(game_env_local.history), sum(sum(white_kings))+sum((black_kings)) > 0))

        if not is_Legal(prev_prev_word, prev_word, new_word):
            syntax_error_counter += 1

            get_true_game = True
            id = get_real_move(whole_game[start_index:])
            if id == -1:
                moves_per_game.append(len(game_env_local.history))
                break
            move_global = whole_game[start_index - 1: start_index + id]
            length_of_partial_game = start_index + id + 1
            # length_of_partial_game = start_index + id
            while len(move_global) > 1:
                partial_move = move_global[:4] if move_global[0]=="K" else move_global[:3]
                _, _ = check_legal_move(game_env_global, partial_move)
                move_global = move_global[3:] if move_global[0]=="K" else move_global[2:]
            # _, _ = check_legal_move(game_env_global, move_global)
            game_env_local = copy.deepcopy(game_env_global)
            start_index = length_of_partial_game
            prev_prev_word = whole_game[start_index-2]
            prev_word = whole_game[start_index-1]
            move = [prev_word]

            error_move_number.append(len(game_env_local.history))
            # move = []
            # get_true_game = True
        else:
            get_true_game = False
            move.append(new_word)
            if (move[0] == "K" and len(move) == 4) or (len(move) == 3 and move[0] != "K"):
                # print(completion)
                # print(move)
                # game_env_local.print_board()
                legal, swap_player = check_legal_move(game_env_local, move)
                # print(words_list_to_move(move), legal)

                if legal and not swap_player:
                    # game_env_local.print_board()
                    move = [move[-1]]
                    success_nodes += 1
                    prev_prev_word = prev_word
                    prev_word = new_word
                elif legal and swap_player:
                    # game_env_local.print_board()
                    id = get_real_move(whole_game[start_index:])
                    success_nodes += 1
                    if id == -1:
                        moves_per_game.append(len(game_env_local.history))
                        break
                    move_global = whole_game[start_index - 1: start_index + id]
                    length_of_partial_game = start_index + id + 1
                    # length_of_partial_game = start_index + id
                    while len(move_global) > 1:
                        partial_move = move_global[:4] if move_global[0] == "K" else move_global[:3]
                        _, _ = check_legal_move(game_env_global, partial_move)
                        move_global = move_global[3:] if move_global[0] == "K" else move_global[2:]
                    # _, _ = check_legal_move(game_env_global, move_global)
                    game_env_local = copy.deepcopy(game_env_global)
                    start_index = length_of_partial_game
                    prev_prev_word = whole_game[start_index - 2]
                    prev_word = whole_game[start_index - 1]
                    get_true_game = True
                    move = [prev_word]
                    # move = []

                else: #not legal
                    semantic_error_counter += 1
                    try:
                        game_env_hypothetical = copy.deepcopy(game_env_local)
                        hypothetical_move = move
                        hypothetical_move[-2] = '-' if hypothetical_move[-2] == 'x' else 'x'
                        legal, swap_player = check_legal_move(game_env_hypothetical, hypothetical_move)
                        if legal:
                            if hypothetical_move[-2] == 'x':
                                x_instead_of_line_counter += 1
                            else:
                                line_instead_of_x_counter += 1
                    except:
                        pass
                    id = get_real_move(whole_game[start_index:])
                    if id == -1:
                        moves_per_game.append(len(game_env_local.history))
                        break
                    move_global = whole_game[start_index - 1: start_index + id]
                    length_of_partial_game = start_index + id + 1
                    # length_of_partial_game = start_index + id
                    while len(move_global) > 1:
                        partial_move = move_global[:4] if move_global[0] == "K" else move_global[:3]
                        _, _ = check_legal_move(game_env_global, partial_move)
                        move_global = move_global[3:] if move_global[0] == "K" else move_global[2:]
                    # _, _ = check_legal_move(game_env_global, move_global)
                    game_env_local = copy.deepcopy(game_env_global)
                    start_index = length_of_partial_game
                    prev_prev_word = whole_game[start_index - 2]
                    prev_word = whole_game[start_index - 1]
                    get_true_game = True
                    move = [prev_word]

                    error_move_number.append(len(game_env_local.history))
                    # move = []
            else:
                prev_prev_word = prev_word
                prev_word = new_word


    bar.set_description(f"{success_nodes/total_nodes*100:.2f}% pass rate: {success_nodes}/{total_nodes} among all searched nodes. guessed padding {guessed_padding} times)")
print(f"{success_nodes/total_nodes*100:.2f}% pass rate: {success_nodes}/{total_nodes} among all searched nodes\n"
                        f"Other stats: {np.array(moves_per_game).mean()} avg moves per game, {syntax_error_counter} syntax errors, {semantic_error_counter} semantic errors.\n{x_instead_of_line_counter} times would have been correct if guessed x instead of -, "
                        f"{line_instead_of_x_counter} times would have been correct if guessed - instead of x. Err move no' mean: {np.array(error_move_number).mean()}, err move no' std: {np.array(error_move_number).std()}\n"
                        f"Guessed king {len(king_guesses)} times, out of them in {sum([b for a,b in king_guesses])} cases there was a king on the board.\n"
                        f"Mean king guess move no': {np.array([a for a,b in king_guesses]).mean()}, std king guess move no': {np.array([a for a,b in king_guesses]).std()}")


    # 3x4->(1,5),(2,4)
#     white_kings = game_env.state[3]
#     black_kings = game_env.state[1]
#
#     player = int(game_env.state[4, 0, 0])
# # ((1,4), (1,8))
#     trans = translate_move(m)
#     if not equals:
#         if player == 0 and black_kings[m[1][0] - 1, m[1][1] - 1] == 1 or player == 1 and white_kings[
#             m[1][0] - 1, m[1][1] - 1] == 1:
#             moves_str += 'K'
#         moves_str += str(trans[0])
#     equals = False
#     if eat:
#         moves_str += 'x'
#     else:
#         moves_str += '-'
#     moves_str += str(trans[1])
#
#     if last != player:
#         moves_str += ' '
#     else:
#         equals = True
#     last = player
#     if done:
#         break


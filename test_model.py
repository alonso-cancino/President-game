import numpy as np
import random
from itertools import chain, combinations
import itertools
import torch
import matplotlib.pyplot as plt
from torch import nn
from collections import deque,namedtuple,Counter
import copy
from time import time
import torch.optim as optim
import math
from linear_agent import *
from president_game import *

device = 'cpu'

plays = [[0,0]]
for number in range(0,10):
    for amount in range(1,5):
        plays.append([amount,number])

N_TEST_GAMES = int(input('Number of test games :'))


AGENT = 'net'

TRAINED_WEIGHTS_PATH = input('Agent\'s q_network state_dict path:')
HUMAN_TESTING = False
N_PLAYERS = int(TRAINED_WEIGHTS_PATH[22])

MAX_GAME_STEPS = 350

state_dict = torch.load(TRAINED_WEIGHTS_PATH)
net = DQN(82+N_PLAYERS-1,41).to(device)
net.load_state_dict(state_dict)

game = president_game(N_PLAYERS)

agent_placement_history = []
random_agent_1_placement_history = []
random_agent_2_placement_history = []
random_agent_3_placement_history = []

game_number = 0

while game_number < N_TEST_GAMES:
    
    game.reset()
    game_done = False
    game_steps = 0

    while not game_done and game_steps <= MAX_GAME_STEPS:

        game_copy = copy.deepcopy(game)

        still_playing = [hand for hand in game.players_data if len(hand)!=0]

        if len(still_playing)<= 1:
            
            game.placements += [s for s in range(4) if s not in game.placements]
            game_done = True
                
        e_p_p = game.encoded_possible_plays()
        pos_plays = [p[0] for p in e_p_p]
        encoded_pos_plays = [p[1] for p in e_p_p]
        
        if game.active_player == 0 and AGENT == 'net':

            state = game.get_torch_state()
            #print(encoded_pos_plays)
            net_output = net(state).detach().numpy()
            #print(net_output)
            masked_output = masked_weights(encoded_pos_plays,net_output)
            #print(masked_output)
            selected_play = encoded_pos_plays.index(plays[np.nanargmax(masked_output)])

            print('With stack ',game.stack,' and possible plays ',[list(p) for p in pos_plays],' agent decided to play ',list(pos_plays[selected_play]))

        elif game_active_player == 1 and HUMAN_TESTING == True:

            game.render()

            selected_play = input('Select your play :')

        else:
            
            selected_play = random.randint(0,len(encoded_pos_plays)-1)

        game.play_select(selected_play)
        
        if len(game.players_data[game_copy.active_player]) == 0 and game_copy.active_player not in game.placements:
                game.placements.append(game_copy.active_player)

        game_steps += 1

    game_number += 1

    print(game.placements)
    
    if N_PLAYERS == 2:

        agent_placement_history.append(game.placements.index(0)+1)
        random_agent_1_placement_history.append(game.placements.index(1)+1)
        
    if N_PLAYERS == 4:
        
        random_agent_2_placement_history.append(game.placements.index(2)+1)
        random_agent_3_placement_history.append(game.placements.index(3)+1)
        
    plt.show()  

if N_PLAYERS == 2:
    print('Posicion promedio en todas las partidas de los agentes aleatorios :',np.mean(random_agent_1_placement_history))
    fig,ax = plt.subplots(nrows=1,ncols=1)
    ax.hist([agent_placement_history,random_agent_1_placement_history],bins=2,label = ['trained deep q learning agent','random agent'])
    ax.legend(prop={'size':10})
    ax.set_title('Agent placement history')
    plt.savefig('agent_placement_history.png')
    plt.show()  
elif N_PLAYERS == 4:
    print('Posicion promedio en todas las partidas de los agentes aleatorios :',np.mean(random_agent_1_placement_history),np.mean(random_agent_2_placement_history),np.mean(random_agent_3_placement_history))
    fig,ax = plt.subplots(nrows=1,ncols=1)
    ax.hist([agent_placement_history,random_agent_1_placement_history,random_agent_2_placement_history,random_agent_3_placement_history],bins=4,label = ['trained deep q learning agent','random agent 1','random agent 2','random agent 3'])
    ax.legend(prop={'size':10})
    ax.set_title('Agent placement history')
    plt.savefig('agent_placement_history.png')
                 

print('Historia de posiciones del agente :',agent_placement_history)
print('Posicion promedio en todas las partidas :',np.mean(agent_placement_history))
print('Posicion promedio en las ultimas 100 partidas :',sum(agent_placement_history[-100:])/len(agent_placement_history[-100:]))



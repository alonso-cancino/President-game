import gym
import numpy as np
import random
from itertools import chain, combinations
import itertools
from gym import spaces
import torch
import gym
import matplotlib.pyplot as plt
from torch import nn
from collections import deque,namedtuple
import copy
from time import time
import torch.optim as optim
from collections import Counter

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','game'))

plays = [[0,0]]
for number in range(0,10):
    for amount in range(1,5):
        plays.append([amount,number])

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return [list(lst) for lst in chain.from_iterable(combinations(s, r) for r in range(len(s)+1))]
        
def play_lst_eval(lst):
    hand_evaluation = 0
    for c in lst:
        hand_evaluation += int(c[0])+1

def encode_play(lst):
    pos_plays = []
    for play in lst:
        if play == []:
            pos_plays.append([0,0])
        elif play != []:
            if len(set([p[0] for p in play])) == 1:
                pos_plays.append([len(play),int(play[0][0])])
    return pos_plays

def encode_state(card_lst):
    deck = []
    for pinta in ['O','B','C','E']:
        for n in range(10):
            deck.append(str(n)+pinta)
    encoded = [0]*40
    for card in card_lst:
        encoded[deck.index(card)]=1
    return encoded


def masked_weights(pos_plays,weights,plays=plays):
    
    return [weights[ind] if plays[ind] in pos_plays else np.nan for ind in range(len(plays))]

def dive(game,rw_funct='weight'):
    
    copied_game = copy.deepcopy(game)
    active_player = copied_game.active_player
    pos_plays1 = copied_game.possible_plays()
    next_step_rewards = []
    
    #print(len(next_step_rewards))
    
    for play1 in range(len(pos_plays1)):
        
        aux_game1 = copy.deepcopy(copied_game)
        aux_game1.play_select(play1)
        pos_plays2 = aux_game1.possible_plays(skip=False)
        
        next_step_rewards.append([])
        
        for play2 in range(len(pos_plays2)):
            
            aux_game2 = copy.deepcopy(aux_game1)
            aux_game2.play_select(play2)
            pos_plays3 = aux_game2.possible_plays(skip=False)
            
            for play3 in range(len(pos_plays3)):
                
                aux_game3 = copy.deepcopy(aux_game2)
                aux_game3.play_select(play3)
                pos_plays4 = aux_game3.possible_plays(skip=False)
                
                for play4 in range(len(pos_plays4)):
                    
                    #print(play1,play2,play3,play4)
                    
                    aux_game4 = copy.deepcopy(aux_game3)
                    player = aux_game4.active_player
                    aux_game4.play_select(play4)
                    pos_plays5 = aux_game4.possible_plays(skip=False)

                    if rw_funct == 'weight':
                        reward_lst = [sum([10-(int(c[0])) for c in play]) for play in pos_plays5]
                    elif rw_funct == 'time':
                        reward_lst = [-1 if len(aux_game4.players_data[player])!=0 else 0]
                    elif rw_funct == 'net':
                        pass
                    
                    #print(len(next_step_rewards))
                    
                    next_step_rewards[play1].append(max(reward_lst))
    
    next_avg_max_rewards = []
    #print(next_step_rewards)
    for lst in next_step_rewards:
        next_avg_max_rewards.append(np.mean(lst))
        
    return pos_plays1, next_avg_max_rewards

class president_game:
    
    def __init__(self,n_j):
        self.n_j = n_j
        self.make_deck()
    
    def reset(self):
        self.last_player = None
        self.active_player = None
        self.stack = []
        self.history = []
        self.deal()
        self.active_player_0()
        self.placements = []
        
    def reset_from_data(self,players_data,a_p,l_p,stack,history,placements):
        self.players_data = players_data
        self.active_player = a_p
        self.last_player = l_p
        self.stack = stack
        self.history = history
        self.placements = placements
        
    def make_deck(self):
        self.deck = []
        for pinta in ['O','B','C','E']:
            for n in range(10):
                self.deck.append(str(n)+pinta)
                
    def deal(self):
        self.history = []
        self.players_data = []
        self.stack = []
        available_indices = list(range(40))
        for player in range(self.n_j):
            sample = random.sample(available_indices,int(40/self.n_j))
            self.players_data.append([self.deck[ind] for ind in sample])
            available_indices = list(set(available_indices) - set(sample))
    
    def hand_eval(self, player):
        hand_evaluation = 0
        for c in self.players_data[player]:
            hand_evaluation += -int(c[0])-1
        return hand_evaluation
    
    def active_player_0(self):
        if self.active_player == None:
            for player in range(self.n_j):
                if '0O' in self.players_data[player]:
                    self.active_player = player
    
    def possible_plays(self, player = None,skip=True):
        
        if player == None:
            active_hand = self.players_data[self.active_player]
        else:
            active_hand = self.players_data[player]
            
        if self.last_player == self.active_player:
            self.stack = ['*']
            
        if len(active_hand) == 0 and self.stack == ['*']:
            if skip == True:
                self.active_player = ( self.active_player + 1) % self.n_j
                self.last_player = ( self.last_player + 1 ) % self.n_j
            else:
                self.last_player = ( self.last_player +1 )% self.n_j
                
            return [[]]
            
        curr_quantity = len(self.stack)
        if curr_quantity == 0:
            if ('0O' in active_hand):
                pos_plays = [lst for lst in list(powerset([c for c in active_hand if c[0]=='0'])) if ('0O' in lst)]
                #for p in range(len(pos_plays)):
                    #print(str(p)+': ',pos_plays[p])
                return pos_plays          
        elif self.stack == ['*']:
            pos_plays = [lst for lst in list(powerset(active_hand)) if len(list(set([c[0] for c in lst])))==1]
            return pos_plays
        else:
            curr_number = int(self.stack[0][0])
            pos_plays = [lst for lst in itertools.combinations(active_hand, curr_quantity) if (len(list(set([c[0] for c in lst])))==1) and (int(lst[0][0]) > curr_number)]
            #print([lst for lst in itertools.combinations(active_hand, curr_quantity) if (len(list(set([c[0] for c in lst])))==1)])
            pos_plays.append([])
            return pos_plays
    
    def encoded_possible_plays(self):
        return [(play,encode_play([play])[0]) for play in self.possible_plays()]
        
    def play_select(self, selection):
            
        pos_plays = self.possible_plays()
        
        if pos_plays[selection] != []:
            self.stack = list(pos_plays[selection])
            self.history +=list(pos_plays[selection])
            self.last_player = self.active_player
            self.players_data[self.active_player] = list(set(self.players_data[self.active_player])-set(pos_plays[selection]))
        self.active_player = (self.active_player + 1)% self.n_j
        
        return self

    def get_torch_state(self):
        
        hand = self.players_data[self.active_player]
        history = self.history
        encoded_hand = encode_state(hand)
        encoded_history = encode_state(history)
        hand_lens = [len(self.players_data[ind]) for ind in range(self.n_j) if ind != self.active_player]
        
        if self.stack != ['*']:
            stack = encode_play([self.stack])
        else:
            stack = [[0,0]]
            
        lst_state = encoded_hand + encoded_history + hand_lens + stack[0]
        
        state = torch.tensor(lst_state,dtype = torch.float32)
        
        return state

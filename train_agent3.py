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
import math

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

class president_game:
    
    def __init__(self,n_j):
        self.n_j = n_j
        self.make_deck()
        
    def make_deck(self):
            self.deck = []
            for pinta in ['O','B','C','E']:
                for n in range(10):
                    self.deck.append(str(n)+pinta)
    
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
                
    def deal(self):
        self.history = []
        self.players_data = []
        self.stack = []
        available_indices = list(range(40))
        for player in range(self.n_j):
            sample = random.sample(available_indices,int(40/self.n_j))
            self.players_data.append([self.deck[ind] for ind in sample])
            available_indices = list(set(available_indices) - set(sample))

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
        return [encode_play([play])[0] for play in self.possible_plays()]
        
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
    
device = 'cpu'
class DQN(nn.Module):
    def __init__(self,state_space_dim,action_space_dim):
        super().__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(state_space_dim,64),
            nn.ReLU(),
            nn.Linear(64,64*2),
            nn.ReLU(),
            nn.Linear(64*2,action_space_dim)
        )
        
    def forward(self,x):
        x = x.to(device)
        return self.linear(x)


def reward_fn(game,play):
    x = len(game.players_data[game.active_player])
    if x==0:
        return 0
    p = game.possible_plays()
    if len(p[play]) < x:
        return -1
    elif len(p[play]) == x:
        return len([hand for hand in game.players_data if len(hand)!=0])-1
        

def exp_next_rw(game,net):
    
    #new_game = president_game(game.n_j)
    #p_d = game.players_data
    #a_p = game.active_player
    #l_p = game.last_player
    #stack = game.stack
    #history = game.history
    #placements = game.placements
    #new_game.reset_from_data(p_d,a_p,l_p,stack,history,placements)
    
    pos_plays = game.possible_plays()
    next_action_rw = []
    
    for ind, ply in enumerate(pos_plays):
        new_game = copy.deepcopy(game)
        new_game.play_select(ind)
        step = 1
        while step <= game.n_j:
            step += 1
            encoded_pos_plays = new_game.encoded_possible_plays()
            #print(encoded_pos_plays)
            state = new_game.get_torch_state()
            with torch.no_grad():
                net_output = net(state)
                masked_output = masked_weights(encoded_pos_plays,net_output.detach().numpy())
                selected_play = encoded_pos_plays.index(plays[np.nanargmax(masked_output)])
            #print(masked_output,selected_play)
            new_game.play_select(selected_play)
        encoded_pos_plays = new_game.encoded_possible_plays()
        state = new_game.get_torch_state()
        masked_output = net(state).detach().numpy()
        next_action_rw.append((ind,np.nanmax(masked_output)))
    #print(next_action_rews)
    return next_action_rw

#net = DQN(85,41).to('cpu')
#game = president_game(4)
#game.reset()
#exp_next_rw(game,net)


def optimize_model():
    
    #for param in policy_net.parameters():
    #    print(param.requires_grad,param.grad)

    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    states = []
    actions = []
    rewards = []
    next_states = []
    games = []
    
    rewards_lst = []
    
    for state in batch.state:
        states.append(state.cpu().detach().numpy())
    for action in batch.action:
        actions.append(action.cpu().detach().numpy())
    for reward in batch.reward:
        rewards.append(reward.detach().numpy())
    for next_state in batch.next_state:
        if next_state!= None:
            next_states.append(next_state.detach().numpy())
    for game in batch.game:
        games.append(game)

    t_states = torch.tensor(np.array(states),dtype = torch.float32)
    state_action_values = policy_net(t_states)
    next_state_values = [[]]*len(batch[0])
    
    for index,game in enumerate(games):
        
        pos_plays = game.possible_plays()
        rewards = [0]*41
        encoded_pos_plays = [encode_play([p])[0] for p in pos_plays]
        
        next_max_action_values = exp_next_rw(game,target_net)

        vals = [v[1] for v in next_max_action_values]
        vals_aux = [0]*len(plays)

        for s,i in [(s,plays.index(play)) for (s,play) in enumerate(encoded_pos_plays)]:
            rewards[i] = reward_fn(game,s)
            vals_aux[i] = next_max_action_values[s][1]
            
        next_state_values[index] = vals_aux
        rewards_lst.append(rewards)
    #print(next_state_values)
    
    next_state_values = torch.tensor(next_state_values,dtype = torch.float32)
    t_rewards = torch.tensor(rewards_lst, dtype = torch.float32)

    expected_state_action_values = (next_state_values * GAMMA) + t_rewards
    
    optimizer.zero_grad()
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)
    print(loss)
    loss.backward()
    
    #for param in policy_net.parameters():
    #    print(param.requires_grad,param.grad)
    
    optimizer.step()


if __name__ == '__main__':

    device = 'cpu'

    N_GAMES = int(input('Number of games :'))
    N_PLAYERS = int(input('Number of players :'))
    MAX_GAME_STEPS = 350
    TARGET_UPDATE = 1

    BATCH_SIZE = 64
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 1000

    policy_net = DQN(80+(N_PLAYERS-1)+2,41).to(device)
    target_net = DQN(80+(N_PLAYERS-1)+2,41).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    #for param in policy_net.parameters():
    #    print(param.requires_grad,param.grad)
    
    #print(target_net.state_dict())
    params_t0 = copy.deepcopy(target_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(),lr=0.01)
    memory = ReplayMemory(10000)
    game_number = 0
    steps_done = 0
    j = president_game(N_PLAYERS)
    while game_number < N_GAMES:
        start = time()
        game_done = False
        game_steps = 0
        overall_game_rewards = [0]*N_PLAYERS
        j.reset()
        while (not game_done) and (game_steps <= MAX_GAME_STEPS):
            game = copy.deepcopy(j)
            #print(game_steps)
            still_playing = [hand for hand in j.players_data if len(hand)!=0]
            if len(still_playing)<= 1:
                j.placements += [s for s in range(N_PLAYERS) if s not in j.placements]
                game_done = True
                
            a_p = copy.deepcopy(j.active_player)
            pos_plays = j.possible_plays()
            n_a_p = j.active_player
            
            #if a_p != n_a_p:
            #    print('beep boop skipped a player')
            
            encoded_pos_plays = [encode_play([play])[0] for play in pos_plays]
            play_rewards = [reward_fn(j,play) for play in range(len(pos_plays))]
            state = j.get_torch_state()
            rand = random.random()
            eps_threshold = EPS_END + (EPS_START-EPS_END)*math.exp(-1*steps_done/EPS_DECAY)
            
            if rand > eps_threshold:
                with torch.no_grad():
                    net_output = policy_net(state)
                    masked_output = masked_weights(encoded_pos_plays,net_output.detach().cpu().numpy())   
                    selected_play = encoded_pos_plays.index(plays[np.nanargmax(masked_output)])
            else:
                selected_play = random.randint(0,len(pos_plays)-1)
                
            overall_game_rewards[j.active_player] += play_rewards[selected_play]
            action = torch.tensor(encoded_pos_plays[selected_play],dtype = torch.float32)
            reward = torch.tensor(play_rewards[selected_play],dtype = torch.float32)
            
            j.play_select(selected_play)
            new_state = j.get_torch_state()
            memory.push(state,action,new_state,reward,game)
            
            if len(j.players_data[game.active_player]) == 0 and game.active_player not in j.placements:
                j.placements.append(game.active_player)
            
            #print(f'Game_step {game_steps}, and placements {j.placements}')
            
            game_steps += 1
            steps_done += 1
            
        print(overall_game_rewards)
        start_opt = time()
        optimize_model()
        end = time()
        print('Game ',game_number,' took ',end-start,'s with placements ',j.placements,' training took ',end-start_opt,'s.')
        game_number += 1
        
        if game_number % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
    params_t1 = target_net.state_dict()
    torch.save(params_t1,f'trained_model_weights_{N_PLAYERS}p_{N_GAMES}g.pt')
    
    for c in params_t0:
        print(params_t0[c]-params_t1[c])

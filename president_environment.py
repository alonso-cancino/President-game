from president_utils import *

class President(gym.env):
    def __init__(self,n_players):
        super().__init__()
        self.n_players = n_players
        self.action_space.n = 50
        self.observation_space.n = 2**(40*(n_players+1))*(n_players**2)*5*10
        # Las acciones son de la forma [valor de la carta, cantidad de copias]
        self.action_space = spaces.MultiDiscrete([5,10])
        # las observaciones son de forma [one_hot(mano #1),...,one_hot(mano #n_p),one_hot(historia),
        #                                                  ... jugador_activo, último jugador, stack]
        self.observation_space = spaces.MultiDiscrete([2]*(40*(n_players+1))+[n_players]*2+[5,10])
        self.action_space.n = 5*10
        self.observation_space.n = 2**(40*(n_players+1)*(n_players**2)*(5*10)
                                       
    def reset(self):
        hands = deal_hands()
        return state
        
    def step(self,action,agent,reward = 'game-placement'):
        if player in agent_lst:
            if reward == 'game-placement':
                agent_reward += get_reward(self.placements,player)
        if len(self.placements) == self.n_players:
            done = True
        if is_valid_play()
        new_state = self.get_new_state(action)
        return new_state, agent_reward, done
    
    def render(self, godmode=False):
        state = get_state()
        self.state = state

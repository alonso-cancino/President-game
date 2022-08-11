class culo_game:
    def __init__(self,n_j):
        self.players_data = []
        self.stack = []
        self.history = []
        self.n_j = n_j
        self.deck = []
        self.active_player = None
        self.last_player = None
        available_indices =list(range(40))
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
            
    def active_player_0(self):
        if self.active_player == None:
            for player in range(self.n_j):
                if '0O' in self.players_data[player]:
                    self.active_player = player
        
    def possible_plays(self, player = False):
        if player == False:
            active_hand = self.players_data[self.active_player]
        else:
            active_hand = self.players_data[player]
        curr_quantity = len(self.stack)
        if curr_quantity == 0:
            if ('0O' in active_hand):
                pos_plays = [lst for lst in list(powerset([c for c in active_hand if c[0]=='0'])) if ('0O' in lst)]
                #for p in range(len(pos_plays)):
                    #print(str(p)+': ',pos_plays[p])
                return pos_plays          
        elif self.stack == ['*']:
            pos_plays = [lst for lst in list(powerset(active_hand)) if len(list(set([c[0] for c in lst])))==1]
            pos_plays.append([])
            return pos_plays
        else:
            curr_number = int(self.stack[0][0])
            pos_plays = [lst for lst in itertools.combinations(active_hand, curr_quantity) if (len(list(set([c[0] for c in lst])))==1) and (int(lst[0][0]) > curr_number)]
            #print([lst for lst in itertools.combinations(active_hand, curr_quantity) if (len(list(set([c[0] for c in lst])))==1)])
            pos_plays.append([])
            return pos_plays
     
    def play_select(self, selection = None):
        if self.last_player == self.active_player:
            self.stack = ['*']
            self.history.append('*')
        pos_plays = self.possible_plays()
        for ind in range(len(pos_plays)):
            print(ind,': ',pos_plays[ind])
        if selection == None:
            selection = int(input('Que jugada elige :'))
        if pos_plays[selection] != []:
            self.stack = list(pos_plays[selection])
            self.history.append(pos_plays[selection])
            self.last_player = self.active_player
            self.players_data[self.active_player] = list(set(self.players_data[self.active_player])-set(pos_plays[selection]))
        self.active_player = (self.active_player + 1)% self.n_j
        return self
        
    def hand_eval(self, player):
        hand_evaluation = 0
        for c in self.players_data[player]:
            hand_evaluation += -int(c[0])-1
        return hand_evaluation
        

from torch import nn

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

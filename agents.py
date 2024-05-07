import torch
import torch.nn as nn

class ActorCriticDiscrete(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorCriticDiscrete, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        actor_output = self.actor(state)
        actor_output = self.softmax(actor_output)
        critic_output = self.critic(state)
        return actor_output, critic_output
    
class ActorCriticContinuous(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorCriticContinuous, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.log_std = nn.Parameter(torch.zeros(output_size))
        
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        actor_output = self.actor(state)
        critic_output = self.critic(state)
        std = torch.exp(self.log_std)
        return actor_output, critic_output, std
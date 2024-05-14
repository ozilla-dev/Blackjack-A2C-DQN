import numpy as np
import torch
import torch.nn as nn
from collections import deque, namedtuple

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
    
class DeepQLearning(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepQLearning, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, state):
        output = self.model(state)
        return output
    
class ExperienceReplay:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        
    def sample_batch(self):
        batch = np.radnom.sample(self.memory, self.batch_size)
        return batch
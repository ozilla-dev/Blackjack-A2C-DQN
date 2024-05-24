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
        
        self.softmax = nn.Softmax(dim=0) # Select the action dimension

    def forward(self, state):
        """Get the actor and critic output for the given state by applying the forward pass of the network and softmax on the actor output.

        Parameters:
            state (torch.Tensor): The input state for the network.

        Returns:
            actor_output (torch.Tensor): Model output for the actor network.
            critic_output (torch.Tensor): Model output for the critic network.
        """
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
        """Get the model output for the given state by applying the forward pass of the network.

        Parameters:
            state (torch.Tensor): The input state for the network.

        Returns:
            output (torch.Tensor): Model output for the network.
        """
        output = self.model(state)
        return output
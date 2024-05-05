import numpy as np
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorCritic, self).__init__()
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
        
        self.softmax = nn.Softmax(dim=-1) # the action dimension is -1

    def forward(self, state):
        actor_output = self.actor(state)
        actor_output = self.softmax(actor_output)
        critic_output = self.critic(state)
        return actor_output, critic_output
    
def A2C(learning_rate, n_repetitions, n_timesteps, gamma):
    environment = gym.make('CartPole-v1')
    input_size = environment.observation_space.shape[0]
    hidden_size = 16
    output_size = environment.action_space.n

    model = ActorCritic(input_size, hidden_size, output_size)
    optimizer_actor = torch.optim.Adam(model.actor.parameters(), lr=learning_rate) # minimizes the loss
    optimizer_critic = torch.optim.Adam(model.critic.parameters(), lr=learning_rate) # minimizes the loss
    
    rewards = []
    for repetition in range(n_repetitions):
        state, _ = environment.reset()
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0
        done = False
        for timestep in range(n_timesteps):
            probabilities, values = model.forward(state)
            distribution = torch.distributions.Categorical(probabilities)

            action = distribution.sample()
            
            next_state, reward, done, _, _ = environment.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32)
            _, next_values = model.forward(next_state)
            advantage = reward + (gamma * next_values * (1 - int(done))) - values # leave out the future rewards if the episode is done
            
            total_reward += reward
            state = next_state

            actor_loss = -distribution.log_prob(action) * advantage.detach()# detach the advantage because we don't want to update the critic
            critic_loss = (advantage.square()).mean()
            total_reward += reward
            
            optimizer_actor.zero_grad() # reset the gradients
            optimizer_critic.zero_grad() # reset the gradients
            
            actor_loss.backward()
            critic_loss.backward()
            
            optimizer_actor.step()
            optimizer_critic.step()
            if done:
                break
        
        rewards.append(total_reward)
        print(f"Repetition {repetition}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}, Reward: {total_reward}")
    torch.save(model.state_dict(), 'model_weights3.pth')    
    
    plt.plot(rewards)
    plt.xlabel('Repetition')
    plt.ylabel('Reward')
    plt.title('Actor 2 Critic')
    plt.show()
    
A2C(learning_rate=0.001, n_repetitions=1000, n_timesteps=100000, gamma=0.99)



# After trainings
environment = gym.make('CartPole-v1', render_mode='human')
input_size = environment.observation_space.shape[0]
hidden_size = 16
output_size = environment.action_space.n

model = ActorCritic(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('model_weights3.pth'))

state, _ = environment.reset()
state = torch.tensor(state, dtype=torch.float32)
done = False
while not done:
    environment.render()
    probabilities, _ = model.forward(state)
    distribution = torch.distributions.Categorical(probabilities)
    action = distribution.sample()
    state, _, done, _, _ = environment.step(action.item())
    state = torch.tensor(state, dtype=torch.float32)
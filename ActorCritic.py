import numpy as np
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
from agents import ActorCriticDiscrete, ActorCriticContinuous, DeepQLearning

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    environment = gym.make("Blackjack-v1", render_mode='human')
    return environment
    
def A2C(input_size, hidden_size, output_size, environment, seed, learning_rate, n_repetitions, gamma):
    model = ActorCriticDiscrete(input_size, hidden_size, output_size)
    optimizer_actor = torch.optim.Adam(model.actor.parameters(), lr=0.0005) # minimizes the loss
    optimizer_critic = torch.optim.Adam(model.critic.parameters(), lr=learning_rate) # minimizes the loss
    
    rewards = []
    for repetition in range(n_repetitions):
        state, _ = environment.reset(seed=seed)
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0
        done = False
        count = 0
        while not done:
            count += 1
            probabilities, values = model(state)
            distribution = torch.distributions.Categorical(probabilities)

            action = distribution.sample()
            
            next_state, reward, done, truncation, _ = environment.step(action.item())
            if truncation:
                done = True
            next_state = torch.tensor(next_state, dtype=torch.float32)
            _, next_values = model(next_state)
            advantage = reward + (gamma * next_values.detach() * (1 - int(done))) - values # leave out the future rewards if the episode is done
            actor_loss = -distribution.log_prob(action) * advantage.detach() # detach the advantage because we don't want to update the critic
            critic_loss = advantage.square()
            total_reward += reward
            
            optimizer_actor.zero_grad() # reset the gradients
            optimizer_critic.zero_grad() # reset the gradients
            
            actor_loss.backward()
            critic_loss.backward()
            
            optimizer_actor.step()
            optimizer_critic.step()
            state = next_state
        if total_reward >= environment.spec.reward_threshold:
            print("Threshold reached")
                
        
        rewards.append(total_reward)
        print(f"Repetition {repetition}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}, Reward: {total_reward}, Count: {count}")
    environment.close()
    torch.save(model.state_dict(), 'model_weights.pth')
    
    plt.plot(rewards)
    plt.xlabel('Repetition')
    plt.ylabel('Reward')
    plt.title('Actor 2 Critic')
    plt.show()
    plt.savefig('CartPole.png')
    
def blackjack(environment, seed, learning_rate, n_repetitions, gamma):
    torch.manual_seed(seed)
    np.random.seed(seed)
    environment = gym.make("Blackjack-v1")
    input_size = len(environment.observation_space)
    hidden_size = 32
    output_size = environment.action_space.n
    model = ActorCriticDiscrete(input_size, hidden_size, output_size)
    optimizer_actor = torch.optim.Adam(model.actor.parameters(), lr=0.0005) # minimizes the loss
    optimizer_critic = torch.optim.Adam(model.critic.parameters(), lr=learning_rate) # minimizes the loss
    
    wins = np.zeros(n_repetitions)
    draws = np.zeros(n_repetitions)
    losses = np.zeros(n_repetitions)
    for repetition in range(n_repetitions):
        state, _ = environment.reset()
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0
        done = False
        count = 0
        while not done:
            environment.render()
            count += 1
            probabilities, values = model(state)
            distribution = torch.distributions.Categorical(probabilities)

            action = distribution.sample()
            
            next_state, reward, done, truncation, _ = environment.step(action.item())
            if truncation:
                done = True
            if reward > 0:
                wins[repetition] += 1
            elif reward < 0:
                losses[repetition] += 1
            else:
                draws[repetition] += 1
            next_state = torch.tensor(next_state, dtype=torch.float32)
            _, next_values = model(next_state)
            advantage = reward + (gamma * next_values.detach() * (1 - int(done))) - values # leave out the future rewards if the episode is done
            actor_loss = -distribution.log_prob(action) * advantage.detach() # detach the advantage because we don't want to update the critic
            critic_loss = advantage.square()
            total_reward += reward
            
            optimizer_actor.zero_grad() # reset the gradients
            optimizer_critic.zero_grad() # reset the gradients
            
            actor_loss.backward()
            critic_loss.backward()
            
            optimizer_actor.step()
            optimizer_critic.step()
            state = next_state
        # print(f"Repetition {repetition}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}, Reward: {total_reward}, Count: {count}")
    environment.close()
    torch.save(model.state_dict(), 'blackjack.pth')
    return wins, draws, losses

def blackjack_DQL(environment, seed, learning_rate, n_repetitions, gamma):
    torch.manual_seed(seed)
    np.random.seed(seed)
    environment = gym.make("Blackjack-v1")
    input_size = len(environment.observation_space)
    hidden_size = 32
    output_size = environment.action_space.n

    model = DeepQLearning(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    batch_size = 64
    experience_replay = ExperienceReplay(10000, batch_size)
    epsilon = 0.1
    for repetition in range(n_repetitions):
        state, _ = environment.reset()
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = environment.action_space.sample()
            else:
                q_values = model(state)
                action = torch.argmax(q_values).item()

            next_state, reward, done, truncation, _ = environment.step(action)
            if truncation:
                done = True
            next_state = torch.tensor(next_state, dtype=torch.float32)
            experience_replay.memory.append((state, action, next_state, reward, done))
            state = next_state
            total_reward += reward

            if len(experience_replay.memory) > batch_size:
                experiences = random.sample(experience_replay.memory, experience_replay.batch_size)
                states, actions, next_states, rewards, dones = zip(*experiences)
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)
                next_states = np.array(next_states)
                dones = np.array(dones)
                q_values = model(torch.tensor(states, dtype=torch.float32))
                next_q_values = model(torch.tensor(next_states, dtype=torch.float32))
                
                target_q_values = q_values.clone()
                for i in range(len(experiences)):
                    if dones[i]:
                        target_q_values[i, actions[i]] = rewards[i]
                    else:
                        target_q_values[i, actions[i]] = rewards[i] + gamma * torch.max(next_q_values[i]).item()
                loss = nn.MSELoss()(q_values, torch.tensor(target_q_values, dtype=torch.float32))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            state = next_state
                
        if repetition % 10 == 0:
            print(f"Repetition {repetition}, Reward: {total_reward}")

    environment.close()
    torch.save(model.state_dict(), 'blackjack_dql.pth')
    return rewards
        

def test(input_size, hidden_size, output_size, weights, A2C):
    environment = gym.make("Blackjack-v1")
    if A2C:
        model = ActorCriticDiscrete(input_size, hidden_size, output_size)
    else:
        model = DeepQLearning(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(weights))
    total_wins = 0
    total_losses = 0
    total_draws = 0
    for i in range(100000):
        state, _ = environment.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        while not done:
            environment.render()
            if A2C:
                probabilities, _ = model(state)
                distribution = torch.distributions.Categorical(probabilities)
                action = distribution.sample()
            else:
                action = torch.argmax(model(state))
            next_state, reward, done, _, _ = environment.step(action.item())
            if done:
                if reward > 0:
                    total_wins += 1
                elif reward < 0:
                    total_losses += 1
                else:
                    total_draws += 1
            state = torch.tensor(next_state, dtype=torch.float32)
    environment.close()
    print(f"Wins: {total_wins}, Losses: {total_losses}, Draws: {total_draws}")
    
def test_random():
    environment = gym.make("Blackjack-v1")
    total_wins = 0
    total_losses = 0
    total_draws = 0
    for i in range(100000):
        environment.reset()
        done = False
        while not done:
            action = environment.action_space.sample()
            _, reward, done, _, _ = environment.step(action)
            if done:
                if reward > 0:
                    total_wins += 1
                elif reward < 0:
                    total_losses += 1
                else:
                    total_draws += 1
    environment.close()
    print(f"Wins: {total_wins}, Losses: {total_losses}, Draws: {total_draws}")

def experiment(tests, seed):
    environment = gym.make("Blackjack-v1")
    
    # Initialize the numpy arrays
    wins = np.array([])
    draws = np.array([])
    losses = np.array([])

    for i in range(tests):
        wins_i, draws_i, losses_i = blackjack(environment, seed, learning_rate=0.001, n_repetitions=2000, gamma=0.5)
        
        # Concatenate the results of each test
        wins = np.concatenate((wins, wins_i))
        draws = np.concatenate((draws, draws_i))
        losses = np.concatenate((losses, losses_i))

    # Calculate the cumulative counts
    cumulative_wins = np.cumsum(wins)
    cumulative_draws = np.cumsum(draws)
    cumulative_losses = np.cumsum(losses)

    # Plot the cumulative counts
    plt.plot(cumulative_wins, label='Wins')
    plt.plot(cumulative_draws, label='Draws')
    plt.plot(cumulative_losses, label='Losses')
    plt.xlabel('Repetition')
    plt.ylabel('Cumulative Count')
    plt.title('Cumulative Counts of Wins, Draws, and Losses Over All Repetitions')
    plt.legend()
    plt.show()
    
def main():
    seeds = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for seed in seeds:
        environment = set_seed(seed)
        input_size = len(environment.observation_space)
        hidden_size = 32
        output_size = environment.action_space.n
        discrete = True
        experiment(10, seed)
        # blackjack(environment, seed, learning_rate=0.001, n_repetitions=20000, gamma=0.5)
    test(input_size=input_size, hidden_size=hidden_size, output_size=output_size, weights='blackjack.pth', discrete=discrete)
    test_random()
    
class ExperienceReplay:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        
    def sample_batch(self):
        batch = np.radnom.sample(self.memory, self.batch_size)
    
if __name__ == '__main__':
    main()
    
import numpy as np
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from scipy.signal import savgol_filter
from agents import ActorCriticDiscrete, DeepQLearning, ExperienceReplay

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    environment = gym.make("Blackjack-v1")
    return environment
    
def A2C_blackjack(environment, seed, learning_rate, n_repetitions, gamma):
    total_rewards = np.zeros(n_repetitions)
    
    environment = set_seed(seed)
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
            total_rewards[repetition] = reward
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
    torch.save(model.state_dict(), 'A2C_blackjack.pth')
    return total_rewards

def DQL_blackjack(environment, seed, learning_rate, n_repetitions, gamma):
    total_rewards = np.zeros(n_repetitions)
    
    environment = set_seed(seed)
    input_size = len(environment.observation_space)
    hidden_size = 32
    output_size = environment.action_space.n

    model = DeepQLearning(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    batch_size = 64
    experience_replay = ExperienceReplay(10000, batch_size)
    epsilon = 0.1
    for repetition in range(n_repetitions):
        state, _ = environment.reset(seed=seed)
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
            total_rewards[repetition] = reward
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
    environment.close()
    torch.save(model.state_dict(), 'DQL_blackjack.pth')
    return total_rewards
        

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
    for i in range(10000):
        state, _ = environment.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        while not done:
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
    return total_wins, total_losses, total_draws
    
def test_random():
    environment = gym.make("Blackjack-v1")
    total_wins = 0
    total_losses = 0
    total_draws = 0
    for i in range(10000):
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
    return total_wins, total_losses, total_draws

def experiment(tests, seeds, n_repetitions):
    for agent in ['A2C', 'DQL']:
        total_rewards = np.zeros((len(seeds), n_repetitions))
        for i, seed in enumerate(seeds):
            environment = set_seed(seed)
            if agent == 'A2C':
                rewards = A2C_blackjack(environment, seed=seed, learning_rate=0.001, n_repetitions=n_repetitions, gamma=0.5)
            elif agent == 'DQL':
                rewards = DQL_blackjack(environment, seed=seed, learning_rate=0.001, n_repetitions=n_repetitions, gamma=0.5)
            total_rewards[i] = rewards
        mean_rewards = np.mean(total_rewards, axis=0)
        window = 100
        smoothed_rewards = savgol_filter(mean_rewards, window, 1)
        plt.plot(smoothed_rewards)
        plt.xlabel('Repetitions')
        plt.ylabel('Reward')
        plt.ylim(-1, 1)
        plt.title(f'Reward vs Repetitions with {agent} model')
        plt.savefig(f'{agent}_rewards.png')
        plt.close()
    
    human_rates = [4222, 848, 4910]
    for agent in ['Random', 'A2C', 'DQL']:
        total_wins = 0
        total_draws = 0
        total_losses = 0
        for i, seed in enumerate(seeds):
            environment = set_seed(seed)
            input_size = len(environment.observation_space)
            hidden_size = 32
            output_size = environment.action_space.n
            if agent == 'A2C':
                wins, losses, draws = test(input_size=input_size, hidden_size=hidden_size, output_size=output_size, weights='A2C_blackjack.pth', A2C=True)
            elif agent == 'DQL':
                wins, losses, draws = test(input_size=input_size, hidden_size=hidden_size, output_size=output_size, weights='DQL_blackjack.pth', A2C=False)
            else:
                wins, losses, draws = test_random()
            total_wins += wins
            total_draws += draws
            total_losses += losses
        average_wins = total_wins / tests
        average_draws = total_draws / tests
        average_losses = total_losses / tests
        
        labels = ['Wins', 'Draws', 'Losses']
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x-0.2, [average_wins, average_draws, average_losses], width, label = 'Model')
        plt.bar(x+0.2, human_rates, width, color = 'r', label = 'Human')
        plt.xticks(x, labels)
        plt.xlabel('Rates')
        plt.ylabel('Average Rate')
        plt.title(f'Average Wins, Draws, and Losses with {agent} model')    
        plt.ylim(0, 10000)
        plt.legend()
        plt.savefig(f'{agent}_blackjack.png')
        plt.close()
    
def main():
    seeds = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    environment = gym.make("Blackjack-v1")
    input_size = len(environment.observation_space)
    hidden_size = 32
    output_size = environment.action_space.n
    # A2C_blackjack(environment, seed=20, learning_rate=0.001, n_repetitions=20000, gamma=0.5)
    experiment(len(seeds), seeds, 20000)
    # test(input_size=input_size, hidden_size=hidden_size, output_size=output_size, seed=20, weights='blackjack.pth', A2C=True)
    
if __name__ == '__main__':
    main()
    
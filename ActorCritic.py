import numpy as np
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from scipy.signal import savgol_filter
from collections import deque
from agents import ActorCriticDiscrete, DeepQLearning

def A2C_blackjack(model_info, hidden_size, learning_rate, n_repetitions, gamma):
    """Advantage Actor-Critic for the Blackjack environment. The model is saved as a .pth file when the training is done.
    
    Parameters:
        model_info (str): The information about the model, so that the model can be saved as a .pth file.
        hidden_size (int): The size of the hidden layer.
        learning_rate (float): The learning rate for the optimizer.
        n_repetitions (int): The number of repetitions to train the model on.
        gamma (float): The discount factor.
    
    Returns:
        total_reward (np.ndarray): The total rewards for each repetition.
    """
    environment = gym.make("Blackjack-v1")
    total_rewards = np.zeros(n_repetitions)
    
    input_size = len(environment.observation_space)
    output_size = environment.action_space.n
    
    model = ActorCriticDiscrete(input_size, hidden_size, output_size)
    optimizer_actor = torch.optim.Adam(model.actor.parameters(), lr=learning_rate) # minimizes the loss
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
    environment.close()
    torch.save(model.state_dict(), f'A2C_blackjack_{model_info}.pth')
    return total_rewards

def DQL_blackjack(model_info, hidden_size, learning_rate, n_repetitions, gamma):
    """Deep Q-Learning for the Blackjack environment using an experience buffer. The model is saved as a .pth file when the training is done.
    
    Parameters:
        model_info (str): The information about the model, so that the model can be saved as a .pth file.
        hidden_size (int): The size of the hidden layer.
        learning_rate (float): The learning rate for the optimizer.
        n_repetitions (int): The number of repetitions to train the model on.
        gamma (float): The discount factor.
    
    Returns:
        total_reward (np.ndarray): The total rewards for each repetition.
    """
    environment = gym.make("Blackjack-v1")
    total_rewards = np.zeros(n_repetitions)
    
    input_size = len(environment.observation_space)
    output_size = environment.action_space.n

    model = DeepQLearning(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    batch_size = 64
    memory = deque(maxlen=10000)
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
            total_rewards[repetition] = reward
            if truncation:
                done = True
            next_state = torch.tensor(next_state, dtype=torch.float32)
            memory.append((state, action, next_state, reward, done))
            state = next_state
            total_reward += reward

            if len(memory) > batch_size:
                experiences = random.sample(memory, batch_size)
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
                mse_loss = nn.MSELoss()
                loss = mse_loss(q_values, target_q_values.clone().detach().to(dtype=torch.float32))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                state = next_state
    environment.close()
    torch.save(model.state_dict(), f'DQL_blackjack_{model_info}.pth')
    return total_rewards

def test(input_size, hidden_size, output_size, weights, A2C):
    """Test the trained model on the Blackjack environment.

    Parameters:
        input_size (int): The size of the input.
        hidden_size (int): The size of the hidden layer.
        output_size (int): The size of the output.
        weights (str): The path to the weights of the model.
        A2C (bool): Whether the model is an Actor-Critic model or not.
        
    Returns:
        total_wins (int): The total number of wins.
        total_losses (int): The total number of losses.
        total_draws (int): The total number of draws.
    """
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
    return total_wins, total_losses, total_draws
    
def test_random():
    """Test the random model on the Blackjack environment.

    Returns:
        total_wins (int): The total number of wins.
        total_losses (int): The total number of losses.
        total_draws (int): The total number of draws.
    """
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
    return total_wins, total_losses, total_draws

def experiment(environment, n_tests, n_repetitions):
    """Run all the experiments for the Blackjack environment and both models and plot the results.

    Parameters:
        environment (gym.Env): The environment to run the experiments on.
        n_tests (int): The number of tests to run.
        n_repetitions (int): The number of repetitions to train the models on.
    """
    optimal_values = {'A2C': {"Evaluation": np.full(n_repetitions, -np.inf), "Learning_rate": 0},
                      'DQL': {"Evaluation": np.full(n_repetitions, -np.inf), "Learning_rate": 0}}
    
    input_size = len(environment.observation_space)
    output_size = environment.action_space.n
    
    learning_rates = [0.0001, 0.0005, 0.001]
    window = 180
    
    optimal_values = train_models(n_tests, n_repetitions, learning_rates, window)
    plot_optimal_curves(optimal_values, window)
    test_models(n_tests, learning_rates, input_size, output_size)
        
def train_models(n_tests, n_repetitions, learning_rates, window):
    """Train two models on the Blackjack environment and plot the results.

    Parameters:
        n_tests (int): The number of tests to run.
        n_repetitions (int): The number of repetitions to train the models on.
        learning_rates (list): The learning rates to train the models on.
        window (int): The window size for the smoothing function.

    Returns:
        optimal_values (dict): The optimal curves for the two models.
    """
    optimal_values = {'A2C': {"Evaluation": np.full(n_repetitions, -np.inf), "Learning_rate": 0},
                      'DQL': {"Evaluation": np.full(n_repetitions, -np.inf), "Learning_rate": 0}}
    for agent in ['A2C', 'DQL']:
        plt.figure()
        for learning_rate in learning_rates:
            model_info = f'{learning_rate}'
            total_rewards = np.zeros((n_tests, n_repetitions))
            for i in range(n_tests):
                if agent == 'A2C':
                    rewards = A2C_blackjack(model_info, hidden_size=32, learning_rate=learning_rate, n_repetitions=n_repetitions, gamma=0.99)
                elif agent == 'DQL':
                    rewards = DQL_blackjack(model_info, hidden_size=128, learning_rate=learning_rate, n_repetitions=n_repetitions, gamma=0.3)
                total_rewards[i] = rewards
            mean_rewards = np.mean(total_rewards, axis=0)
            smoothed_rewards = savgol_filter(mean_rewards, window, 1)
            if np.sum(mean_rewards) > np.sum(optimal_values[agent]["Evaluation"]):
                optimal_values[agent]["Evaluation"] = mean_rewards
                optimal_values[agent]["Learning_rate"] = learning_rate
            plt.plot(smoothed_rewards, label=f'Learning Rate: {learning_rate}')
        plt.xlabel('Number of Repetitions')
        plt.ylabel('Average Reward')
        plt.ylim(-1, 1)
        plt.legend()
        plt.savefig(f'{agent}_rewards.png')
        plt.close()
    return optimal_values

def plot_optimal_curves(optimal_values, window):
    """Plot the optimal curves for the two models.

    Parameters:
        optimal_values (dict): The optimal values for the two models.
        window (int): The window size for the smoothing function.
    """
    plt.figure()
    for agent in ['A2C', 'DQL']:
        optimal_value = optimal_values[agent]["Evaluation"]
        smoothed_rewards = savgol_filter(optimal_value, window, 1)
        plt.plot(smoothed_rewards, label=f'{agent} - Learning Rate: {optimal_values[agent]["Learning_rate"]}')
    plt.xlabel('Number of Repetitions')
    plt.ylabel('Average Reward')
    plt.ylim(-1, 1)
    plt.legend()
    plt.savefig('optimal_rewards.png')
    plt.close()
    
def test_models(n_tests, learning_rates, input_size, output_size):
    """Test the random and trained models on the Blackjack environment and plot the results.

    Parameters:
        environment (gym.Env): The environment to test the models on.
        n_tests (int): The number of tests to run.
        learning_rates (list): The learning rates to test the models on.
        input_size (int): The size of the input.
        output_size (int): The size of the output.
    """
    environment = gym.make("Blackjack-v1")
    human_rates = [4222, 848, 4910]
    labels = ['Wins', 'Draws', 'Losses']
    for agent in ['Random', 'A2C', 'DQL']:
        plt.figure()
        random_count = 0
        for lr, learning_rate in enumerate(learning_rates):
            total_wins = 0
            total_draws = 0
            total_losses = 0
            for i in range(n_tests):
                if agent == 'A2C':
                    wins, losses, draws = test(environment=environment, input_size=input_size, hidden_size=32, output_size=output_size, weights=f'A2C_blackjack_{learning_rate}.pth', A2C=True)
                elif agent == 'DQL':
                    wins, losses, draws = test(environment=environment, input_size=input_size, hidden_size=128, output_size=output_size, weights=f'DQL_blackjack_{learning_rate}.pth', A2C=False)
                elif random_count == 0:
                    wins, losses, draws = test_random(environment=environment)
                total_wins += wins
                total_draws += draws
                total_losses += losses
            random_count += 1
            average_wins = total_wins / n_tests
            average_draws = total_draws / n_tests
            average_losses = total_losses / n_tests
            
            x = np.arange(len(labels))
            # The following code is used to correctly align and plot the bars based on the agent
            if agent == 'A2C' or agent == 'DQL':
                width = 0.15
                offset = lr * width
                plt.bar(x - width + offset, [average_wins, average_draws, average_losses], width, label = f'Learning Rate: {learning_rate}')
            elif agent == 'Random':
                width = 0.35
                if random_count == 1:
                    plt.bar(x - 0.2, [average_wins, average_draws, average_losses], width, label = 'Random')
        if agent == 'A2C' or agent == 'DQL':
            plt.bar(x - width + len(learning_rates) * width, human_rates, width, color = 'r', label = 'Human')
            plt.xticks(x + (width / 2), labels)
        else:
            plt.bar(x + 0.2, human_rates, width, color = 'r', label = 'Human')
            plt.xticks(x, labels)
            
        plt.xlabel('Rates')
        plt.ylabel('Number of occurences')   
        plt.ylim(0, 10000)
        plt.legend()
        plt.savefig(f'{agent}_blackjack.png')
        plt.close()
    
def main():
    """Run the experiment function with the right parameters."""
    environment = gym.make("Blackjack-v1")
    n_tests = 10
    n_repetitions = 5000
    experiment(environment, n_tests, n_repetitions)
    
if __name__ == '__main__':
    main()
    
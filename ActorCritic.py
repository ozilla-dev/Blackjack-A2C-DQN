import numpy as np
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from agents import ActorCriticDiscrete, ActorCriticContinuous

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    environment = gym.make("LunarLander-v2", render_mode='human')
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
    
def test(input_size, hidden_size, output_size, weights, discrete):
    environment = gym.make("LunarLander-v2", render_mode='human')
    if discrete:
        model = ActorCriticDiscrete(input_size, hidden_size, output_size)
    else:
        model = ActorCriticContinuous(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(weights))
    total_wins = 0
    for i in range(15):

        state, _ = environment.reset()
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        while not done:
            environment.render()
            if discrete:
                probabilities, _ = model(state)
                distribution = torch.distributions.Categorical(probabilities)
                action = distribution.sample()
                next_state, reward, done, _, _ = environment.step(action.item())
                if done:
                    if reward > 0:
                        print("Player wins!")
                    elif reward < 0:
                        print("Dealer wins!")
                    else:
                        print("It's a draw!")
                state = torch.tensor(next_state, dtype=torch.float32)
            else:
                actor_output, _, std = model(state)
                distribution = torch.distributions.Normal(actor_output, std)
                action = torch.tanh(distribution.sample())
                next_state, _, done, _, _ = environment.step(np.array([action.item()]))
                state = torch.tensor(next_state, dtype=torch.float32)
    environment.close()
    print(total_wins)
    
def mountain_car(input_size, hidden_size, output_size, environment, seed, learning_rate, n_repetitions, gamma):
    model = ActorCriticContinuous(input_size, hidden_size, output_size)
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
            actor_output, values, std = model(state)
            distribution = torch.distributions.Normal(actor_output, std)
            action = torch.tanh(distribution.sample())
            next_state, reward, done, truncation, _ = environment.step(np.array([action.item()]))
            if truncation:
                done = True
            next_state = torch.tensor(next_state, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            _, next_values, _ = model(next_state)
            advantage = reward + (gamma * next_values.detach() * (1 - int(done))) - values # leave out the future rewards if the episode is done
            actor_loss = -distribution.log_prob(action) * advantage.detach() # detach the advantage because we don't want to update the critic
            critic_loss = advantage.square()
            
            optimizer_actor.zero_grad() # reset the gradients
            optimizer_critic.zero_grad() # reset the gradients
            
            actor_loss.backward()
            critic_loss.backward()
            
            optimizer_actor.step()
            optimizer_critic.step()
            
            total_reward += reward
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
    plt.savefig('MountainCar.png')
    
    
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
    
    rewards = []
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
        rewards.append(total_reward)
        print(f"Repetition {repetition}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}, Reward: {total_reward}, Count: {count}")
    environment.close()
    torch.save(model.state_dict(), 'blackjack.pth')
    
    plt.plot(rewards)
    plt.xlabel('Repetition')
    plt.ylabel('Reward')
    plt.title('Actor 2 Critic')
    plt.show()
    plt.savefig('BlackJack.png')
    return rewards
    
def asteroid(environment, seed, learning_rate, n_repetitions, gamma):
    torch.manual_seed(seed)
    np.random.seed(seed)
    environment = gym.make("LunarLander-v2")
    input_size = environment.observation_space.shape[0]
    hidden_size = 32
    output_size = environment.action_space.n
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
        rewards.append(total_reward)
        print(f"Repetition {repetition}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}, Reward: {total_reward}, Count: {count}")
    environment.close()
    torch.save(model.state_dict(), 'lunarlander.pth')
    
    plt.plot(rewards)
    plt.xlabel('Repetition')
    plt.ylabel('Reward')
    plt.title('Actor 2 Critic')
    plt.show()
    plt.savefig('lunarlander.png')
    
def experiment():
    total_rewards = []
    environment = gym.make("Blackjack-v1")
    for i in range(10):
        total_reward = blackjack(environment, learning_rate=0.001, n_repetitions=20000, gamma=0.3)
        total_rewards.append(total_reward)
    average_rewards = np.mean(total_rewards, axis=0)
    plt.plot(average_rewards)
    plt.xlabel('Repetition')
    plt.ylabel('Reward')
    plt.show()
    
def main():
    seed = 20
    environment = set_seed(seed)
    # input_size = environment.observation_space.shape[0]
    # hidden_size = 32
    # output_size = environment.action_space.shape[0]
    # A2C(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
    #   environment=environment, seed=seed, learning_rate=0.001, n_repetitions=1000, gamma=0.99)
    # mountain_car(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
    #              environment=environment, seed=seed, learning_rate=0.001, n_repetitions=1000, gamma=0.99)
    # blackjack(environment, seed, learning_rate=0.001, n_repetitions=1000, gamma=0.5)
    # asteroid(environment, seed, learning_rate=0.001, n_repetitions=1000, gamma=0.9)
    # discrete = True
    # input_size = len(environment.observation_space)
    # hidden_size = 32
    # output_size = environment.action_space.n
    # experiment()
    input_size = environment.observation_space.shape[0]
    hidden_size = 32
    output_size = environment.action_space.n
    discrete = True
    asteroid(environment=environment, seed=seed, learning_rate=0.001, n_repetitions=400, gamma=0.99)
    test(input_size=input_size, hidden_size=hidden_size, output_size=output_size, weights='lunarlander.pth', discrete=discrete)
    
    
if __name__ == '__main__':
    main()
    
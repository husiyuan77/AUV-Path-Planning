import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from matplotlib import pyplot as plt
from datetime import *
from real2D_env222 import Env
import math,random,os,csv


# 记录loss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class prioritized_replay_buffer(object):
    def __init__(self, capacity, alpha, beta, beta_increment):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.pos = 0
        self.memory = []
        self.priorities = np.zeros([self.capacity], dtype=np.float32)

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)

        max_prior = np.max(self.priorities) if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append([observation, action, reward, next_observation, done])
        else:
            self.memory[self.pos] = [observation, action, reward, next_observation, done]
        self.priorities[self.pos] = max_prior
        self.pos += 1
        self.pos = self.pos % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < self.capacity:
            probs = self.priorities[: len(self.memory)]
        else:
            probs = self.priorities
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probs[indices]) ** (- self.beta)
        if self.beta < 1:
            self.beta += self.beta_increment
        weights = weights / np.max(weights)
        weights = np.array(weights, dtype=np.float32)

        observation, action, reward, next_observation, done = zip(* samples)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init
        self.training = True
        self.weight_mu = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(self.output_dim, self.input_dim))
        self.register_buffer('weight_epsilon', torch.FloatTensor(self.output_dim, self.input_dim))

        self.bias_mu = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(self.output_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(self.output_dim))

        self.reset_parameter()
        self.reset_noise()



    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

    def reset_parameter(self):
        mu_range = 1 / np.sqrt(self.input_dim)

        self.weight_mu.detach().uniform_(-mu_range, mu_range)
        self.bias_mu.detach().uniform_(-mu_range, mu_range)

        self.weight_sigma.detach().fill_(self.std_init / np.sqrt(self.input_dim))
        self.bias_sigma.detach().fill_(self.std_init / np.sqrt(self.output_dim))

    def _scale_noise(self, size):
        noise = torch.randn(size)
        noise = noise.sign().mul(noise.abs().sqrt())
        return noise

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))


class ddqn(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(ddqn, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.fc = nn.Linear(self.observation_dim, 128)
        self.noisy1 = nn.Linear(128, 128)
        self.noisy2 = NoisyLinear(128, self.action_dim)



    def forward(self, observation):
        x = self.fc(observation)
        x = F.relu(x)
        x = self.noisy1(x)
        x = F.relu(x)
        x = self.noisy2(x)
        return x

    def act(self, observation, epsilon):
        if random.random() > epsilon:
            # q_value = self.forward(torch.from_numpy(observation).float().unsqueeze(0).to(device))
            q_value = self.forward(observation)
            action = q_value.max(1)[1].detach()[0].item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action

    def reset_noise(self):
        # self.noisy1.reset_noise()
        self.noisy2.reset_noise()
    def load_w(self):
        fname_model = os.path.join("./noisemodel2022-09-20 21:02:03.368831/", '3900model.dump')
        #fname_optim = os.path.join(self.model_dir, '756optim.dump')

        if os.path.isfile(fname_model):
            print("load")
            self.load_state_dict(torch.load(fname_model))


if __name__ == '__main__':
    epsilon_init = 0.005#0.9
    epsilon_min = 0.005
    epsilon_decay = 0.99
    capacity = 10000
    exploration = 200
    update_freq = 1000
    batch_size = 64
    episode = 1000
    render = False
    learning_rate = 1e-3
    gamma = 0.99
    alpha = 0.6
    beta = 0.4
    beta_increment_step = 1000000

    nstate_list = []
    env = Env()
    #env = env.unwrapped
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    beta_increment = (1 - beta) / beta_increment_step
    target_net = ddqn(observation_dim, action_dim)
    eval_net = ddqn(observation_dim, action_dim)
    eval_net.load_w()
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    buffer = prioritized_replay_buffer(capacity, alpha=alpha, beta=beta, beta_increment=beta_increment)
    epsilon = epsilon_init

    start =datetime.now()

    weight_reward = None
    for i in range(1):
        count = 0
        obs = env.reset()
        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay
        reward_total = 0
        #eval_net.save_param(i)
        if render:
            env.render()
        while True:
            action = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)), epsilon)
            count += 1
            next_obs, reward, done = env.step(action)
         
            nstate_list.append(next_obs)
  
            reward_total += reward
            obs = next_obs

            # if render:
            #     env.render()
            # if len(buffer) > exploration:
            #     train(buffer, eval_net, target_net, batch_size, count, update_freq, gamma, optimizer)
            if count>500:
                break
            if done:
                nstate_list.append(next_obs)
                nstate_list.append('')
                if not weight_reward:
                    weight_reward = reward_total
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                print('episode: {}  epsilon: {:.2f}  reward: {}  weight_reward: {:.3f}'.format(i+1, epsilon, reward_total, weight_reward))
                break
    end =datetime.now()
    env.final(nstate_list)
    print(end-start)
    plt.rcParams['figure.figsize'] = (20.0, 10.0)
    fig, ax = plt.subplots()

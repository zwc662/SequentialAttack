import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

"""
From: https://github.com/pytorch/pytorch/issues/1959
There's an official LayerNorm implementation in pytorch now, but it hasn't been included in 
pip version yet. This is a temporary version
This slows down training by a bit
"""
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

nn.LayerNorm = LayerNorm


class Actor(nn.Module):
    def __init__(self, kernel_size):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size = kernel_size)
        self.conv2 = nn.Conv2d(16, 4, kernel_size = kernel_size)
        self.pool1 = nn.MaxPool2d(2, 2)


        self.conv1_ = nn.Conv2d(4, 16, kernel_size = kernel_size, stride = 2)
        self.conv2_ = nn.Conv2d(16, 3, kernel_size = kernel_size, stride = 2)


    def forward(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = F.relu(self.conv1_(x))
        mu = F.tanh(self.conv2_(x))
        return mu

class Critic(nn.Module):
    def __init__(self, kernel_size):
        super(Critic, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size = kernel_size)
        self.conv2 = nn.Conv2d(16, 4, kernel_size = kernel_size)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(4, 16, kernel_size = kernel_size, stride = 2)
        self.conv4 = nn.Conv2d(16, 3, kernel_size = kernel_size, stride = 2)


        self.conv1_ = nn.Conv2d(3, 16, kernel_size = kernel_size)
        self.conv2_ = nn.Conv2d(16, 4, kernel_size = kernel_size)
        self.pool1_ = nn.MaxPool2d(2,2)
        self.conv3_ = nn.Conv2d(4, 16, kernel_size = kernel_size, stride = 2)
        self.conv4_ = nn.Conv2d(16, 3, kernel_size = kernel_size, stride = 2)

        self.conv = nn.Conv2d(3, 1, kernel_size = kernel_size, strid = 1)



    def forward(self, inputs, actions):
        x = inputs
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        mu = F.tanh(self.conv4(x))
        
        x_ = actions
        x_ = self.conv1_(x_)
        x_ = F.relu(x_)
        x_ = self.conv2d_(x_)
        x_ = F.relu(x_)
        x_ = self.pool1_(x_)
        x_ = F.relu(self.conv3(x_))
        mu_ = F.tanh(self.conv4(x_))

        y =  torch.sum(self.conv(mu_ + mu), 0) 

        return y

class Critic_(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.conv2 = nn.Conv2d(3, 3, 1)

        self.linear1 = nn.Linear(3 * input_size[-1] * input_size[-2], hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size + 3 * input_size[-1] * input_size[-2], hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x = self.conv1(inputs)
        x = x.view(x.size(0), -1) 
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        y = self.conv2(actions)
        y = y.view(y.size(0), -1)
        x = self.linear2(torch.cat((x, y), 1))
        x = self.ln2(x)
        x = F.relu(x)
        V = self.V(x)
        return V



class DDPG(object):
    def __init__(self, gamma, tau, kernel_size, hidden_size, input_size, device):
        self.actor = Actor(kernel_size).to(device)
        self.actor_target = Actor(kernel_size).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(kernel_size).to(device)
        self.critic_target = Critic(kernel_size).to(device)
        #self.critic = Critic(input_size, hidden_size).to(device)
        #self.critic_target = Critic(input_size, hidden_size).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        self.mask = None
        

    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor((Variable(state)))

        self.actor.train()
        mu = mu.data

        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise())
        action = mu.clamp(-1, 1)

        if action.size() != state.size():
            bg = 0.0 * state
            if self.mask is None:
                start = max(0, int((bg.size()[-2] - action.size()[-2])/2))
                end = start + min(action.size()[-2], bg.size()[-2])
                start_ = max(0, int((-action.size()[-1] + bg.size()[-1])/2))
                end_ = start_ + min(action.size()[-1], bg.size()[-1])
                self.mask = [start, end, start_, end_]
            else:
                start, end, start_, end_ = self.mask[:]
            bg[:, :, start: end, start_ : end_] = bg[..., start:end, start_: end_] + action
            action = bg
        return action


    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))
        
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()

        policy_loss = -self.critic((state_batch),self.actor((state_batch)))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            param += torch.randn(param.shape) * param_noise.current_stddev

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix) 
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

import glob
import os
import sys
import time

try:
    #sys.path.append(glob.glob('../../PythonAPI')[0])
    sys.path.append(glob.glob('../../bird_view')[0])
    sys.path.append(glob.glob('../../training/')[0])
    sys.path.append(glob.glob('../')[0])
except IndexError as e:
    pass

from bird_view.utils import carla_utils as cu

#from phase2_utils import LocationLoss

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter
from .ddpg import DDPG, VALUE
from .replay_memory import ReplayMemory, Transition


class vae(torch.nn.Module):
    def __init__(self,input_size, output_size):
        super(vae, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size = 1),
                nn.ReLU(True),
                nn.Conv2d(6, 16, kernel_size = 1),
                nn.ReLU(True),
                nn.Conv2d(16, 16, kernel_size = 1),
                nn.ReLU(True),
                nn.Conv2d(16, 16, kernel_size = 1),
                nn.ReLU(True),
                nn.Conv2d(16, 16, kernel_size = 1),
                nn.ReLU(True),
                nn.Conv2d(16, 16, kernel_size = 1),
                nn.ReLU(True))

        self.encoder_ = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size = 60),
                nn.ReLU(True),
                nn.Conv2d(6, 6, kernel_size = 120),
                nn.ReLU(True))

        self.decoder = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size = 1),
                nn.ReLU(True),
                nn.Conv2d(16, 16, kernel_size = 1),
                nn.ReLU(True),
                nn.Conv2d(16, 16, kernel_size = 1),
                nn.ReLU(True),
                nn.Conv2d(16, 16, kernel_size = 1),
                nn.ReLU(True),
                nn.Conv2d(16, 6, kernel_size = 1),
                nn.ReLU(True),
                nn.Conv2d(6, 3, kernel_size = 1),
                nn.Tanh())
        self.decoder_ = nn.Sequential(
                nn.Conv2d(6, 6, kernel_size = 120),
                nn.ReLU(True),
                nn.Conv2d(6, 3, kernel_size = 60),
                nn.Tanh())
        
        '''
        self.linear = nn.Sequential(
                nn.Linear(16 * input_size[-2] * input_size[-1], 16 * input_size[-2] * input_size[-1]),
                nn.ReLU(True),
                nn.Linear(16 * input_size[-2] * input_size[-1], 16 * input_size[-2] * input_size[-1]),
                nn.ReLU(True),)
        '''


    def forward(self, x):
        x_ = self.encoder_(x)
        #x_ = self.linear(x_)
        x_ = self.decoder_(x_)
        return x_



class mlp(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(mlp, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.hd = torch.nn.functional.tanh # instead of Heaviside step fn  
    
    def forward(self, x):
        output = self.fc(x)
        output = self.hd(output) # instead of Heaviside step fn
        return output


class BaseAttack(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, birdview, speed, command):
        return self.model(birdview, speed, command)




class pgd(BaseAttack):
    def __init__(self, model, device, mode = 'lin', eps = 0.3, alpha = 2/255, iters = 10):
        super().__init__(model, device)
        self.loss = nn.functional.mse_loss
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        
        self.func = lambda x: self.eta * (1 - x) + x
        self.nn = None
        self.eta = 0.0
        self.gamma=  0.5
        
        if(mode == 'lin'): 
            self.func = lambda x, y : y * (1 - x) + x
        elif(mode == 'con'): 
            self.func = lambda x, y: y + x 
        elif(mode == 'mlp'): 
            self.eta = 0.3
            self.size = 40
            self.nn = mlp(self.size*self.size, self.size * self.size).cuda()
            self.optimizer = optim.SGD(self.nn.parameters(), lr = 0.01, momentum = .9)
            self.func = lambda x, y: x + torch.clamp(self.demask(x, self.nn(self.mask(x).cuda()).to(self.device)), min = -self.eps, max = self.eps)
        elif(mode == 'vae'):
            self.eta = 0.3
            self.size = 40
            self.trans = None
            self.nn = vae([160, 384], self.size * self.size).cuda()
            self.nn.train()
            self.optimizer = optim.SGD(self.nn.parameters(), lr = 0.01, momentum = .9)
            self.func = lambda x, y: x + torch.clamp(self.nn(x), min = -self.eps, max = self.eps)
        else: self.func = lambda x, y: x

    def mask(self, birdview):
        oh = birdview.size()[2]
        ow = birdview.size()[3]

        #print(birdview[:, :, int(oh/2) - 5: int(oh/2) + 5, int(ow/2) - 5: int(ow/2) + 5].sum(1).flatten().cuda())
        return birdview[:, :, int(oh/2 - self.size/2): int(oh/2 + self.size/2), int(ow/2 - self.size/2): int(ow/2 + self.size/2)].sum(1).flatten().cuda()

        mask = torch.zeros(birdview.size())
        mask[:, int(ow/2) - 5:int(ow/2) + 5, int(ow/2)-5:int(ow/2) + 5] = 1.
        #mask = torch.ones(birdview.size())
        return mask

    def demask(self, birdview, mask):
        oh = birdview.size()[2]
        ow = birdview.size()[3]
        demask = torch.zeros(birdview.size()).cuda()
        demask[:, :, int(oh/2 - self.size/2) : int(oh/2 + self.size/2), int(ow/2 - self.size/2): int(ow/2 + self.size/2)] = torch.reshape(mask, (self.size, self.size)) 
        return demask

   
    def attack_func(self, birdview, speed, command, target_command = None):
        self.iters = 5
        ori_birdview = birdview.data.to(self.device)

        if self.trans is None and self.nn is not None:
            input_size = ori_birdview.size()
            output_size = self.nn(ori_birdview).size()
            self.trans = T.Compose([
                T.ToPILImage(),
                T.Pad((int((input_size[-2] - output_size[-2])/2), int((input_size[-1] - output_size[-1])/2)), 0, 'constant'),
                T.ToTensor()])
            self.func = lambda x, y: x + y 
        if self.nn is None:
            self.eta = self.eta * torch.ones(ori_birdview.size()).to(self.device)

        birdview = torch.clamp(self.func(ori_birdview, self.eta), min = 0, max = 1)

        if target_command is None: 
            target_command = torch.tensor([[0, 1, 0., 0]]).to(self.device)
            self.loss = lambda x, y: -nn.functional.mse_loss(x, y)
        else:
            target_command = target_command.to(self.device)

        with torch.no_grad():
            if self.model.all_branch:
                target_loc, locs = self.predict(ori_birdview, speed, target_command)
                target_loc = target_loc.data
            else:
                target_loc = self.predict(ori_birdview, speed, target_command).data

        for i in range(self.iters):
            if self.nn is None:
                self.eta.requires_grad = True
                birdview = torch.clamp(self.func(ori_birdview, self.eta), min = 0, max = 1)
            else:
                eta = self.nn(ori_birdview)
                if eta.size() != ori_birdview.size():
                    bg = torch.zeros(ori_birdview.size()).cuda()
                    start = max(0, int((bg.size()[-2] - eta.size()[-2])/2))
                    end = start + min(eta.size()[-2], bg.size()[-2])
                    start_ = max(0, int((-eta.size()[-1] + bg.size()[-1])/2))
                    end_ = start_ + min(eta.size()[-1], bg.size()[-1])
                    bg[:, :, start: end, start_ : end_] = bg[..., start:end, start_: end_] + eta 
                    eta = bg
                eta = self.eps * eta #torch.clamp(eta, min = - self.eps, max = self.eps)
                birdview = torch.clamp(self.func(ori_birdview, eta), min = 0, max = 1)

            self.grad_itr(birdview, speed, command, target_loc)
            
            if self.nn is None:
                #self.eta = self.eta + self.alpha * self.eta.grad.flatten().sum()
                self.eta = self.eta + self.alpha * self.eta.grad.sign()

                self.eta = torch.clamp(self.eta,  min = -self.eps, max = self.eps)
                self.eta = self.eta.detach_()
                #eta = self.eta * torch.ones(ori_birdview.size()).to(self.device)
            else:
                self.optimizer.step()
        
        adv_birdview = torch.clamp(self.func(ori_birdview, eta.detach_()), min = 0, max = 1).detach_()

        return adv_birdview
        

    def attack_online(self, birdview, speed, command, target_command = None):
        self.eta = self.gamma * self.eta
        ori_birdview = birdview.data.to(self.device)
        birdview = ori_birdview + self.eta

        if target_command is None: 
            target_command = torch.tensor([[0, 1, 0., 0]]).to(self.device)
            self.loss = lambda x, y: -nn.functional.mse_loss(x, y)
        else:
            target_command = target_command.to(self.device)

        with torch.no_grad():
            if self.model.all_branch:
                target_loc, locs = self.predict(birdview, speed, target_command)
                target_loc = target_loc.data
            else:
                target_loc = self.predict(birdview, speed, target_command).data
        #self.eta.requires_grad = True
        birdview.requires_grad = True
        self.grad_itr(birdview, speed, command, target_loc)
        adv_birdview = birdview + self.alpha * birdview.grad.sign()
        self.eta = torch.clamp(adv_birdview - ori_birdview, min = -self.eps, max = self.eps) #* self.mask(adv_birdview).to(self.device)
        adv_birdview = torch.clamp(ori_birdview + self.eta, min = 0, max = 1).detach_()

        self.eta = adv_birdview - ori_birdview

        return adv_birdview

    
    def attack(self, birdview, speed, command, target_command = None):
        ori_birdview = birdview.data.to(self.device)

        if target_command is None: 
            target_command = torch.tensor([[0, 0, 1., 0]]).to(self.device)
            self.loss = lambda x, y: -nn.functional.mse_loss(x, y)
        else:
            target_command = target_command.to(self.device)

        with torch.no_grad():
            if self.model.all_branch:
                target_loc, locs = self.predict(birdview, speed, target_command)
                target_loc = target_loc.data
            else:
                target_loc = self.predict(birdview, speed, target_command).data

        for i in range(self.iters):
            birdview.requires_grad = True
            self.grad_itr(birdview, speed, command, target_loc)
            adv_birdview = birdview + self.alpha * birdview.grad.sign()
            """
            plt.figure()
            plt.imshow(birdview.squeeze(0).cpu().numpy().transpose(1,2,0))
            plt.show()
            """
            self.eta = torch.clamp(adv_birdview - ori_birdview, min = -self.eps, max = self.eps) #* self.mask(adv_birdview).to(self.device)
            birdview = torch.clamp(ori_birdview + self.eta, min = 0, max = 1).detach_()
        return birdview

    def grad_itr(self, birdview, speed, command, target_loc):
        if self.model.all_branch:
            pred_loc, locs = self.model(birdview, speed, command)
        else:
            pred_loc = self.model(birdview, speed, command)
            
        self.model.zero_grad()

        loss = self.loss(pred_loc, target_loc).to(self.device)
        loss.backward(retain_graph = True)

class value(BaseAttack):
    def __init__(self, model, device, eps = 0.3, alpha = 2/255, iters = 10, gamma = 0.99, replay_size = 10000):
        super().__init__(model, device)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters

        self.episode = 0
        self.eta = 0.3
        self.size = 40
        self.batch_size = 123

        self.gamma = gamma
        self.replay_size = replay_size

        self.target_command = torch.tensor([[0, 1, 0., 0]]).to(self.device)
        self.loss = lambda x, y: -nn.functional.mse_loss(x, y)
        self.input_size = (3, 160, 384)
        self.kernel_size = 16
        self.hidden_size = 53505
        self.tau = 0.001
        self.agent = VALUE(self.gamma, self.tau, self.kernel_size, self.hidden_size, self.device)
        self.memory = ReplayMemory(self.replay_size)
        self.buffer = {"state": [], "action": [], "reward": []}

        self.writer = SummaryWriter()
        self.updates_per_step = 5
        self.updates = 0
        self.step = 0
        

    def single_step(self, ori_birdview, speed, command, target_command = None):
        if target_command is not None: 
            self.target_command = target_command.to(self.device)

        #with torch.no_grad():
        if True:
            if self.model.all_branch:
                target_loc, locs = self.predict(ori_birdview, speed, self.target_command)
                target_loc = target_loc.data
            else:
                target_loc = self.predict(ori_birdview, speed, self.target_command).data

            eta = self.agent.select_action(ori_birdview)
            birdview = torch.clamp(ori_birdview + self.eps * eta, min = 0, max = 1)

            if self.model.all_branch:
                pred_loc, locs = self.model(birdview, speed, command)
            else:
                pred_loc = self.model(birdview, speed, command)

            reward = -nn.functional.mse_loss(pred_loc, target_loc).to(self.device).unsqueeze(0)

            self.buffer["state"].append(birdview)
            self.buffer["action"].append(eta)
            self.buffer["reward"].append(reward)

            return eta 

   
    def attack_func(self, birdview, speed, command, target_command = None):
        ori_birdview = birdview.data.to(self.device)
        eta = self.single_step(ori_birdview, speed, command)
        
        if len(self.buffer["state"]) >= 2:
            info = [self.buffer["state"][0], 
                    self.buffer["action"][0], 
                    torch.ones(self.buffer["state"][0].size(0)), 
                    self.buffer["state"][1], 
                    self.buffer["reward"][0]]
            print(info[-1])
            self.memory.push(*info)
            self.buffer["state"] = [self.buffer["state"][-1]]
            self.buffer["action"] = [self.buffer["action"][-1]]
            self.buffer["reward"] = [self.buffer["reward"][-1]]

        if len(self.memory) > self.batch_size:
            for _ in range(self.updates_per_step):
                transitions = self.memory.sample(self.batch_size)
                batch = Transition(*zip(*transitions))

                value_loss = self.agent.update_parameters(batch)
                writer.add_scalar('loss/value', value_loss, updates)

                updates += 1
            self.agent.save_model(env_name = 'target_0010', suffix = "step_{}_time_{}".format(step, time.strftime("%Y%m%d-%H%M%S")))

        self.step += 1
        adv_birdview = torch.clamp(ori_birdview + self.eta * eta.detach_(), min = 0, max = 1).detach_()
        return adv_birdview
        
        


class ddpg(BaseAttack):
    def __init__(self, model, device, eps = 0.3, alpha = 2/255, iters = 10, gamma = 0.99, replay_size = 10000):
        super().__init__(model, device)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        
        self.episode = 0
        self.eta = 0.3
        self.size = 40
        self.batch_size = 128

        self.gamma = gamma
        self.replay_size = replay_size

        self.target_command = torch.tensor([[0, 1, 0., 0]]).to(self.device)
        self.loss = lambda x, y: -nn.functional.mse_loss(x, y)
        self.input_size = (3, 160, 384)
        self.output_size = [3, 160, 384]
        self.kernel_size = 16
        self.hidden_size = 128
        self.tau = 0.001
        self.agent = DDPG(self.gamma, self.tau, self.kernel_size, self.hidden_size, self.input_size, self.device)
        self.memory = ReplayMemory(self.replay_size)
        self.buffer = {"state": [], "action": [], "reward": []}

        self.writer = SummaryWriter()
        self.updates_per_step = 5
        self.updates = 0
        self.step = 0
        

    def single_step(self, ori_birdview, speed, command, target_command = None):
        if target_command is not None: 
            self.target_command = target_command.to(self.device)

        with torch.no_grad():

            if self.model.all_branch:
                target_loc, locs = self.predict(ori_birdview, speed, self.target_command)
                target_loc = target_loc.data
            else:
                target_loc = self.predict(ori_birdview, speed, self.target_command).data

            eta = self.agent.select_action(ori_birdview)
            birdview = torch.clamp(ori_birdview + self.eps * eta, min = 0, max = 1)

            if self.model.all_branch:
                pred_loc, locs = self.model(birdview, speed, command)
            else:
                pred_loc = self.model(birdview, speed, command)

            reward = self.loss(pred_loc, target_loc).to(self.device).unsqueeze(0)

            self.buffer["state"].append(ori_birdview)
            self.buffer["action"].append(eta)
            self.buffer["reward"].append(reward)

            return eta 

   
    def attack_func(self, birdview, speed, command, target_command = None):
        ori_birdview = birdview.data.to(self.device)
        eta = self.single_step(ori_birdview, speed, command)
        
        if len(self.buffer["state"]) >= 2:
            info = [self.buffer["state"][0], 
                    self.buffer["action"][0], 
                    torch.ones(self.buffer["state"][0].size(0)), 
                    self.buffer["state"][1], 
                    self.buffer["reward"][0]]
            print(info[-1])
            self.memory.push(*info)
            self.buffer["state"] = [self.buffer["state"][-1]]
            self.buffer["action"] = [self.buffer["action"][-1]]
            self.buffer["reward"] = [self.buffer["reward"][-1]]

        if len(self.memory) > self.batch_size:
            for _ in range(self.updates_per_step):
                transitions = self.memory.sample(self.batch_size)
                batch = Transition(*zip(*transitions))

                value_loss, policy_loss = self.agent.update_parameters(batch)
                writer.add_scalar('loss/value', value_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)

                updates += 1
            self.agent.save_model(env_name = 'target_0010', suffix = "step_{}_time_{}".format(step, time.strftime("%Y%m%d-%H%M%S")))

        self.step += 1
        adv_birdview = torch.clamp(ori_birdview + self.eta * eta.detach_(), min = 0, max = 1).detach_()
        return adv_birdview
        
        


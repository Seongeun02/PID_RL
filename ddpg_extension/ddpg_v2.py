import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import yaml

import CSTRenv_v2

import warnings
warnings.filterwarnings("ignore")

# read yaml file
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Hyperparameters
lr_mu = config['hyperparameters']['lr_mu']
lr_q = config['hyperparameters']['lr_q']
gamma = config['hyperparameters']['gamma']
batch_size = config['hyperparameters']['batch_size']
buffer_limit = config['hyperparameters']['buffer_limit']
tau = config['hyperparameters']['tau']

# OU process parameters
theta = config['noise']['theta']
dt = config['noise']['dt']
sigma = config['noise']['sigma']

# Training parameters
num_episodes = config['training']['num_episodes']
print_interval = config['training']['print_interval']
start_train_size = config['training']['start_train_size']
train_iterations = config['training']['train_iterations']

# Replay Buffer
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float).to(DEVICE), \
               torch.tensor(a_lst, dtype=torch.float).to(DEVICE), \
               torch.tensor(r_lst, dtype=torch.float).to(DEVICE), \
               torch.tensor(s_prime_lst, dtype=torch.float).to(DEVICE), \
               torch.tensor(done_mask_lst, dtype=torch.float).to(DEVICE)
    
    def size(self):
        return len(self.buffer)
    

# actor network
class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))  # range [-1,1]
        return mu

# critic network
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(2, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

# OU process
class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = theta, dt, sigma
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
# train
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask  = memory.sample(batch_size)
    
    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s,a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s,mu(s)).mean() # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
        
if __name__ == '__main__':
    DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print("Using PyTorch version: {}, Device: {}".format(torch.__version__, DEVICE))
    
    env = CSTRenv_v2.CSTRenv()
    memory = ReplayBuffer()

    ## critic 
    q, q_target = QNet().to(DEVICE), QNet().to(DEVICE)
    q_target.load_state_dict(q.state_dict())
    
    ## actor for Kp
    mu_p, mu_target_p = MuNet().to(DEVICE), MuNet().to(DEVICE)
    mu_target_p.load_state_dict(mu_p.state_dict())
    
    ## actor for Ki
    mu_i, mu_target_i = MuNet().to(DEVICE), MuNet().to(DEVICE)
    mu_target_i.load_state_dict(mu_i.state_dict())

    score = []
    print_interval = 20

    mu_p_optimizer = optim.Adam(mu_p.parameters(), lr=lr_mu)
    mu_i_optimizer = optim.Adam(mu_i.parameters(), lr=lr_mu)
    q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(num_episodes):
        s = env.reset()  # initial state
        done = False

        count = 0
        update_phase = "Kp"
        while count < 200 and not done:
            s_tensor = torch.from_numpy(np.array(s)).float().to(DEVICE)
            if update_phase == "Kp":
                a = mu_p(s_tensor).cpu().detach().numpy()
            elif update_phase == "Ki":
                a = mu_i(s_tensor).cpu().detach().numpy()
            a = [a[i] + ou_noise()[0] for i in range(len(a))]
            
            s_prime, r, done, update_phase = env.step(a)
            memory.put((s,a,r/100.0,s_prime,done))
            score.append(r)
            s = s_prime
            count += 1
                
            if memory.size() > start_train_size:
                if update_phase == 'Kp':
                    for i in range(train_iterations):
                        train(mu_p, mu_target_p, q, q_target, memory, q_optimizer, mu_p_optimizer)
                        soft_update(mu_p, mu_target_p)
                        soft_update(q,  q_target)
                elif update_phase == 'Ki':
                    for i in range(train_iterations):
                        train(mu_i, mu_target_i, q, q_target, memory, q_optimizer, mu_i_optimizer)
                        soft_update(mu_i, mu_target_i)
                        soft_update(q,  q_target)                    
                
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.5f}".format(n_epi, sum(score[count-print_interval:count])/print_interval))

    env.close()

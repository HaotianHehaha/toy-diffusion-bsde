import torch 
import torch.nn as nn
import numpy as np
from path_sample import brownian_motion

class BSDENet(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, score_func, time_steps=100,device = 'cpu'):
        super(BSDENet, self).__init__()
        # layers for z
        self.z_input_fc = nn.Linear(input_size+2, hidden_size[0]) 
        self.z_input_bn = nn.BatchNorm1d(hidden_size[0])
        self.relu = nn.ReLU()

        self.z_hidden_layers = nn.ModuleList()
        self.z_hidden_bns = nn.ModuleList()

        for i in range(1,len(hidden_size)):
            self.z_hidden_layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
            self.z_hidden_bns.append(nn.BatchNorm1d(hidden_size[i]))

        self.z_output_fc = nn.Linear(hidden_size[-1], out_size)

        # layers for y
        self.y0_input_fc = nn.Linear(input_size, hidden_size[0]) 
        self.y0_input_bn = nn.BatchNorm1d(hidden_size[0])

        self.y0_hidden_layers = nn.ModuleList()
        self.y0_hidden_bns = nn.ModuleList()

        for i in range(1,len(hidden_size)):
            self.y0_hidden_layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
            self.y0_hidden_bns.append(nn.BatchNorm1d(hidden_size[i]))

        self.y0_output_fc = nn.Linear(hidden_size[-1], 1)

        self.score_func = score_func
        self.time_steps = time_steps
        self.t = torch.linspace(0,1,time_steps+1)
        self.h = 1/time_steps
        self.device = device


    def z_calculation(self, x,t,y):
        inpt = torch.cat((x,t,y),dim=1)
        out = self.z_input_bn(self.z_input_fc(inpt))
        out = self.relu(out)

        for _, (fc,bn) in enumerate(zip(self.z_hidden_layers,self.z_hidden_bns)):
            out = bn(fc(out))
            out = self.relu(out)

        out = self.z_output_fc(out)
        return out
    
    def y0_calculation(self, x):
        out = self.y0_input_bn(self.y0_input_fc(x))
        out = self.relu(out)

        for _, (fc,bn) in enumerate(zip(self.y0_hidden_layers,self.y0_hidden_bns)):
            out = bn(fc(out))
            out = self.relu(out)

        out = self.y0_output_fc(out)
        return out
    
    def beta(self,t):
        beta = 1
        return beta * torch.ones_like(t)
    
    # compute the loss base on the score function
    def Loss(self, x, score):
        score_generate = self.score_func(x)
        loss = torch.mean(torch.sum((score_generate-score)**2,dim=1))
        #loss_2 = empirical_mmd(x,sample_true)*alpha
        #print('loss_1:{:.2f},loss_2:{:.2f}'.format(loss_1,loss_2))
        return loss
    

    def sample(self,x):
        bm_incre = torch.tensor(brownian_motion(x.shape,self.time_steps, self.h),dtype=torch.float32).to(self.device)
        trajectory = []
        trajectory.append(x.cpu().detach().numpy())
        for i in range(self.time_steps):
            t = self.t[i]
            if t==0:
                y = self.y0_calculation(x)
            
            # embed t into the input
            t_embed = torch.full((x.shape[0], 1), t).to(self.device)
            z = self.z_calculation(x,t_embed,y)
            mu = 1/2*self.beta(1-t).to(self.device)
            sigma = torch.sqrt(self.beta(1-t)).to(self.device)

            x = x + (mu*x+sigma*z)*self.h + sigma*bm_incre[:,i,:]
            y = y +(-mu + 1/2* torch.einsum('ab,ab->a',z,z).reshape(-1,1))*self.h + torch.einsum('ab,ab->a',z,bm_incre[:,i,:]).reshape(-1,1)
            trajectory.append(x.cpu().detach().numpy())
        t = self.t[-1]
        sigma = torch.sqrt(self.beta(1-t)).to(self.device)

        t_embed = torch.full((x.shape[0], 1), t).to(self.device)
        z = self.z_calculation(x,t_embed,y)
        score = z/sigma

        trajectory = np.array(trajectory)

        return trajectory, score.cpu().detach().numpy(), y.cpu().detach().numpy()
    
    def forward(self, x): 
        bm_incre = torch.tensor(brownian_motion(x.shape,self.time_steps, self.h),dtype=torch.float32).to(self.device)
        for i in range(self.time_steps):
            t = self.t[i]
            if t==0:
                y = self.y0_calculation(x)
            
            # embed t into the input
            t_embed = torch.full((x.shape[0], 1), t).to(self.device)
            z = self.z_calculation(x,t_embed,y)
            mu = 1/2*self.beta(1-t).to(self.device)
            sigma = torch.sqrt(self.beta(1-t)).to(self.device)

            x = x + (mu*x+sigma*z)*self.h + sigma*bm_incre[:,i,:]

            # prevent x to be inf
            x[x>20] = 0
            x[x<-20] = 0

            y = y +(-mu + 1/2* torch.einsum('ab,ab->a',z,z).reshape(-1,1))*self.h + torch.einsum('ab,ab->a',z,bm_incre[:,i,:]).reshape(-1,1)

        t = self.t[-1]
        sigma = torch.sqrt(self.beta(1-t)).to(self.device)

        #embed t into the input
        t_embed = torch.full((x.shape[0], 1), t).to(self.device)
        z = self.z_calculation(x,t_embed,y)
        score = z/sigma

        loss = self.Loss(x,score)

        return loss





import torch 
import torch.nn as nn

class Diffusion(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, time_steps=100,device = 'cpu'):
        super(Diffusion, self).__init__()
        self.input_fc = nn.Linear(input_size+1, hidden_size[0]) 
        self.input_bn = nn.BatchNorm1d(hidden_size[0])
        self.relu = nn.ReLU()

        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()

        for i in range(1,len(hidden_size)):
            self.hidden_layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_size[i]))

        self.output_fc = nn.Linear(hidden_size[-1], out_size)

        self.time_steps = time_steps
        self.t = torch.linspace(0,1,time_steps+1)
        self.h = 1/time_steps
        self.device = device


    def noise_predict(self, x,t):
        inpt = torch.cat((x,t),dim=1)
        out = self.input_bn(self.input_fc(inpt))
        out = self.relu(out)

        for _, (fc,bn) in enumerate(zip(self.hidden_layers,self.hidden_bns)):
            out = bn(fc(out))
            out = self.relu(out)

        out = self.output_fc(out)
        return out
    
    def add_noise(self,x):
        noise = torch.randn_like(x,device=self.device)
        rand_t = torch.randint(1,self.time_steps+1,(x.shape[0],1),device=self.device)/self.time_steps
        x_noisy = torch.exp(-1/2*self.beta(rand_t)*rand_t)*x + torch.sqrt(1-torch.exp(-self.beta(rand_t)*rand_t))*noise
        return x_noisy, noise, rand_t

    def beta(self,t):
        beta = 1
        return beta * torch.ones_like(t)
    
    def sample(self,x):
        for t_0 in self.t.flip(0):
            t = torch.full((x.shape[0], 1), t_0.item()).to(self.device)
            if t_0 >self.h:
                x = x + (1/2*self.beta(t)*x+ self.beta(t)* self.noise_predict(x,t)*(-1.0/torch.sqrt(1-torch.exp(-self.beta(t)*t))))*self.h + torch.sqrt(self.beta(t)*self.h)* torch.randn_like(x)
        return x
    
    def sample_ode(self,x):
        for t_0 in self.t.flip(0):
            t = torch.full((x.shape[0], 1), t_0.item()).to(self.device)
            if t_0 >self.h:
                x = x + (1/2*self.beta(t)*x+ 1/2* self.beta(t)* self.noise_predict(x,t)*(-1.0/torch.sqrt(1-torch.exp(-self.beta(t)*t))))*self.h
        return x

    def forward(self, x): 
        x_noisy, noise, rand_t = self.add_noise(x)
        predict_noise = self.noise_predict(x_noisy,rand_t)
        loss = torch.mean(torch.sum((noise-predict_noise)**2,axis=1))
        return loss

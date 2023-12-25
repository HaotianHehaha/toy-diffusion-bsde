import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from deep_bsde import BSDENet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

current_directory = os.getcwd()
folder_name_to = 'results/only_bsde'

# Concatenate the folder path
folder_path = os.path.join(current_directory,folder_name_to)

# Check if the folder exists, if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

input_size = 2
hidden_size = [10,10]
out_size = 2
time_steps = 100
learning_rate = 1e-3
num_epochs = 200
alpha = 1
batch_size = 500

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {'data': self.data[index]}
        return sample

# define loss function
def gaussian_kernel(x, y, sigma=1.0):
    return torch.exp(-torch.norm(x - y, dim=-1) ** 2 / (2 * sigma ** 2))

def empirical_mmd(X, Y, kernel=gaussian_kernel):
    m, n = len(X), len(Y)

    # Compute the kernel matrices
    Kxx = kernel(X[:, None], X)
    Kxy = kernel(X[:, None], Y)
    #Kyy = kernel(Y[:, None], Y)

    # Compute the MMD
    mmd_squared = (torch.sum(Kxx)-torch.trace(Kxx)) / (m * (m - 1)) - 2 * torch.sum(Kxy) / (m * n) #+(torch.sum(Kyy)-torch.trace(Kyy)) / (n * (n - 1))
    
    return mmd_squared


# Set random seed for reproducibility
torch.manual_seed(30)

# Mean and covariance matrices for three 2D Gaussian distributions
scale = 0.7
mean1 = scale*torch.tensor([2.0, 4.0])
cov1 = scale**2*torch.tensor([[1.2, 0.5],
                     [0.5, 1.2]])

mean2 = scale*torch.tensor([-2.0, -3.0])
cov2 = scale**2*torch.tensor([[1.0, -0.8],
                     [-0.8, 1.0]])
mean3 = scale*torch.tensor([5.0, -1.0])
cov3 = scale**2*torch.tensor([[0.8, 0.],
                     [0., 0.8]])

distribution_para = {'mean':[mean1.detach().numpy(),mean2.detach().numpy(),mean3.detach().numpy()],
                      'cov':[cov1.detach().numpy(),cov2.detach().numpy(),cov3.detach().numpy()]}

# Generate random samples from the three Gaussian distributions
gaussian_1 = torch.distributions.MultivariateNormal(mean1.to(device), cov1.to(device),validate_args=False)
samples1 = gaussian_1.sample((3, ))
gaussian_2 = torch.distributions.MultivariateNormal(mean2.to(device), cov2.to(device),validate_args=False)
samples2 = gaussian_2.sample((2, ))
gaussian_3 = torch.distributions.MultivariateNormal(mean3.to(device), cov3.to(device),validate_args=False)
samples3 = gaussian_3.sample((3, ))
sample_true = torch.cat((samples1,samples2,samples3),dim=0).cpu()

# collect training data
sample_num = 10000
mean_initial = torch.tensor([0.,0.])
cov_initial = torch.tensor([[1., 0.],
                            [0., 1.]])
gaussian_normal = torch.distributions.MultivariateNormal(mean_initial, cov_initial)
x_train_origin = gaussian_normal.sample((sample_num, ))

# collect testing data
test_sample_num = 2000
x_test_origin = gaussian_normal.sample((test_sample_num, ))
torch.save({'training data':x_train_origin, 'testing data':x_test_origin, 'true samples':sample_true, 'distribution_para':distribution_para},os.path.join(folder_path, 'data.pth'))

# Create a DataLoader
my_dataset = MyDataset(x_train_origin)
my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

# define the  score function at initial time
def score_func(x):
    u = x.clone().detach().requires_grad_(True).to(device)#torch.tensor(x,requires_grad=True)
    f = torch.sum(torch.log(0.4*torch.exp(gaussian_1.log_prob(u))+\
                0.2*torch.exp(gaussian_2.log_prob(u))+\
                0.4*torch.exp(gaussian_3.log_prob(u))))
    f.backward()
    score = u.grad
    #clip the gradient to prevent nan and inf
    nan_mask = torch.isnan(score)
    inf_mask = torch.isinf(score)
    score_clip = torch.where(nan_mask, torch.tensor([0.,0.],device=device), score)
    score_clip = torch.where(inf_mask, torch.tensor([0.,0.],device=device), score_clip)
    return score_clip

model = BSDENet(input_size, hidden_size, out_size,score_func, time_steps ,device)
#model.load_state_dict(torch.load(os.path.join(folder_path, 'bsde_model_50.pth'))['model_state_dict'])

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120], gamma=0.1)

# Train the model
train_loss = []
test_loss = []
for epoch in range(num_epochs):
    model.train()
    for _, batch in enumerate(my_dataloader):  
        # Move tensors to the configured device
        data = batch['data'].reshape(-1, input_size).to(device)
        model = model.to(device)

        # Forward pass
        loss = model(data)        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    model.eval()
    loss_1 = model(x_train_origin.to(device))
    train_loss.append(loss_1.item())
    loss_2 = model(x_test_origin.to(device))
    test_loss.append(loss_2.item())
    print ('Epoch [{}/{}], train loss: {:.4f}, test loss: {:.4f}' 
            .format(epoch+1, num_epochs,loss_1.item(), loss_2.item()))

    #Save the model to a file
    if (epoch+1) % 50 ==0:
        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(folder_path,f'bsde_model_{epoch+1}.pth'))
        trajectory,_,_ = model.sample(x_train_origin.to(device))
        plt.figure()
        plt.scatter(trajectory[0,:,0],trajectory[0,:,1],label = 'origin')
        plt.scatter(trajectory[-1,:,0],trajectory[-1,:,1],label = 'generate')
        plt.scatter(sample_true.numpy()[:,0],sample_true.numpy()[:,1],label = 'true')
        plt.scatter(mean1.numpy()[0],mean1.numpy()[1],marker='*',label='mean_1',s=100)
        plt.scatter(mean2.numpy()[0],mean2.numpy()[1],marker='*',label='mean_2',s=100)
        plt.scatter(mean3.numpy()[0],mean3.numpy()[1],marker='*',label='mean_3',s=100)
        plt.title(f'Epoch: {epoch+1}')
        plt.legend()
        plt.show()


train_loss = torch.tensor(train_loss)
test_loss = torch.tensor(test_loss)
torch.save({'train_loss':train_loss,'test_loss':test_loss},os.path.join(folder_path,'loss.pth'))

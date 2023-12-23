import torch
import torch.nn as nn
from diffusion import Diffusion
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

current_directory = os.getcwd()
folder_name_to = 'diffusion/deep'

# Concatenate the folder path
folder_path = os.path.join(current_directory,folder_name_to)

# Check if the folder exists, if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

input_size = 2
hidden_size = [10,32,32,64,64,128,128,64,64,32,32,10]
out_size = 2
time_steps = 100
learning_rate = 1e-3
num_epochs = 50000
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

# Set random seed for reproducibility
torch.manual_seed(42)

# Mean and covariance matrices for three 2D Gaussian distributions
scale = 0.5
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
gaussian_1 = torch.distributions.MultivariateNormal(mean1, cov1)
samples1 = gaussian_1.sample((3, ))
gaussian_2 = torch.distributions.MultivariateNormal(mean2, cov2)
samples2 = gaussian_2.sample((2, ))
gaussian_3 = torch.distributions.MultivariateNormal(mean3, cov3)
samples3 = gaussian_3.sample((3, ))
sample_true = torch.cat((samples1,samples2,samples3),dim=0)

plt.figure()
plt.scatter(sample_true.numpy()[:,0],sample_true.numpy()[:,1],label = 'true')
plt.scatter(mean1.numpy()[0],mean1.numpy()[1],marker='*',label='mean_1',s=100)
plt.scatter(mean2.numpy()[0],mean2.numpy()[1],marker='*',label='mean_2',s=100)
plt.scatter(mean3.numpy()[0],mean3.numpy()[1],marker='*',label='mean_3',s=100)
plt.legend()
plt.show()

model = Diffusion(input_size, hidden_size, out_size, time_steps,device)
#model.load_state_dict(torch.load('diffusion/width30/beta0.5/diffusion_model_200000.pth')['model_state_dict'])

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000,30000], gamma=0.1)

train_loss = []
# Move tensors to the configured device
sample_true_train = sample_true.to(device)
model = model.to(device)

test_sample_num = 2000
mean_initial = torch.tensor([0.,0.])
cov_initial = torch.tensor([[1., 0.],
                            [0., 1.]])
gaussian_normal = torch.distributions.MultivariateNormal(mean_initial, cov_initial)
x_origin = gaussian_normal.sample((test_sample_num, ))

# loss = torch.load('diffusion/width30/beta0.5/loss.pth')
# plt.plot(loss)
# raise ValueError('stop')
for epoch in range(num_epochs):
    model.train()
    # Forward pass
    loss = model(sample_true_train)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_loss.append(loss.item())
    
    if (epoch+1) % 1000 ==0:
        print ('Epoch [{}/{}], train loss: {:.4f}' 
            .format(epoch+1, num_epochs,loss.item()))
    
    if (epoch+1) % 10000 ==0:
        # Save the .pth file in the created folder
        file_path = os.path.join(folder_path, f'diffusion_model_{epoch+1}.pth')

        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
            }, file_path)

        model.eval()
        x_generate = model.sample(x_origin.to(device)).cpu().detach().numpy()
        x_origin_plot = x_origin.numpy()
        plt.figure()
        plt.scatter(x_origin_plot[:,0],x_origin_plot[:,1],label = 'origin')
        plt.scatter(x_generate[:,0],x_generate[:,1],label = 'generate')
        plt.scatter(sample_true.numpy()[:,0],sample_true.numpy()[:,1],label = 'true')
        plt.scatter(mean1.numpy()[0],mean1.numpy()[1],marker='*',label='mean_1',s=100)
        plt.scatter(mean2.numpy()[0],mean2.numpy()[1],marker='*',label='mean_2',s=100)
        plt.scatter(mean3.numpy()[0],mean3.numpy()[1],marker='*',label='mean_3',s=100)
        plt.title(f'Epoch: {epoch+1}')
        plt.legend()
        plt.show()

plt.figure()
plt.plot(train_loss)
plt.show()
file_path = os.path.join(folder_path, f'loss.pth')
torch.save(train_loss,file_path)

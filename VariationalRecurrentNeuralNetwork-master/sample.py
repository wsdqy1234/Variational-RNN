import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import VRNN

#hyperparameters
x_dim = 28
h_dim = 100
z_dim = 16
n_layers =  1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

state_dict = torch.load('saves/vrnn_state_dict_101.pth')
model = VRNN(x_dim, h_dim, z_dim, n_layers, device)
model.load_state_dict(state_dict)
model.to(device)

sample = model.sample(28*6)
plt.imshow(sample.cpu().numpy(), cmap='gray')
plt.savefig("./images/Sample_101.png")
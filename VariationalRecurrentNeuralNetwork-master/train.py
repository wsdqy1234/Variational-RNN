import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from tqdm import tqdm
#------------------------------
from model import VRNN


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""


def train(epoch, device):
	model.train()
	Total_loss = 0
	KLD_loss = 0
	REC_loss = 0
	loop = tqdm(enumerate(train_loader), total = len(train_loader))
	for batch_idx, (data, _) in loop:
		# data:[batch_size, num_channels, H, W]

		# Initialization
		data = data.squeeze().transpose(0, 1) # (H, batch_size, W), where H is treated as time_steps, W is treated as x_dim. Here num_channels=1 and would be squeezed
		data = (data - data.min()) / (data.max() - data.min())
		data = data.to(device)

		# forward + backward + optimize
		optimizer.zero_grad()
		kld_loss, rec_loss, _, _ = model(data)
		loss = kld_loss + rec_loss
		
		Total_loss += loss.item()
		KLD_loss += kld_loss.item()
		REC_loss += rec_loss.item()
		
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), clip) # gradient clipping, no more than clip
		optimizer.step()

		# Print 
		loop.set_description(f"Epoch: {epoch}")
		loop.set_postfix(Total_Loss = Total_loss/len(train_loader.dataset),
			KL_Loss = KLD_loss/len(train_loader.dataset), 
			Rec_Loss = REC_loss/len(train_loader.dataset))


def test(epoch, device):
	"""uses test data to evaluate 
	likelihood of the model"""
	model.eval()
	mean_kld_loss, mean_nll_loss = 0, 0

	with torch.no_grad():
		for i, (data, _) in enumerate(test_loader):                                            
			data = data.squeeze().transpose(0, 1)
			data = (data - data.min()) / (data.max() - data.min())
			data = data.to(device)
			kld_loss, nll_loss, _, _ = model(data)
			mean_kld_loss += kld_loss
			mean_nll_loss += nll_loss

	mean_kld_loss /= len(test_loader.dataset)
	mean_nll_loss /= len(test_loader.dataset)

	print('====> Test set loss: KLD Loss = {:.4f}, Rec Loss = {:.4f} '.format(
		mean_kld_loss, mean_nll_loss))


# Hyperparameters
x_dim = 28		
h_dim = 100		
z_dim = 16
n_layers =  1 	# Number of layers in hidden states
n_epochs = 200
clip = 10		# Gradient clip
learning_rate = 1e-3
batch_size = 128
device = "cpu"

seed = 128
save_every = 10

# Manual seed
torch.manual_seed(seed)
plt.ion()

# GPU
if torch.cuda.is_available(): device = "cuda"
kwargs = {"num_workers":2, "pin_memory":True} if device =='cuda' else {} # Training Settings

# Init model + optimizer + datasets
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
		transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, 
		transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

model = VRNN(x_dim, h_dim, z_dim, n_layers, device).to(device)

# Set eps to ensure loss not to be NaN (Default eps=1e-8 cannot be represented by Float32, and would lead to numeric overflow problem)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-3) 
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
	for epoch in range(1, n_epochs + 1):
		train(epoch, device=device)
		test(epoch, device=device)

		# # Generate an image
		# sample = model.sample(28)
		# plt.imshow(sample.cpu().numpy(), cmap='gray')
		# plt.savefig(fname="Sample.png")

		#saving model
		if epoch % save_every == 1:
			fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
			torch.save(model.state_dict(), fn)
			print('Saved model to '+fn)
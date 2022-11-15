import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 

"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

class VRNN(nn.Module):
	def __init__(self, x_dim, h_dim, z_dim, n_layers, device, bias=False):
		super(VRNN, self).__init__()

		self.x_dim = x_dim
		self.h_dim = h_dim
		self.z_dim = z_dim
		self.n_layers = n_layers
		self.device = device
		self.EPS = torch.finfo(torch.float).eps # very small values, avoid numerical problem during iterations

		# 1. feature-extracting transformations (\phi function definition)
        # -----------------------------------------------------------------
        # \phi^x_\tau(x)
		self.phi_x = nn.Sequential(
			nn.Linear(x_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		# \phi^z_\tau(z)
		self.phi_z = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.ReLU())

		# 2. encoder (Eq.9)
        # -----------------------------------------------------------------
        # \phi^{enc}_\tau()
		self.enc = nn.Sequential(
			nn.Linear(h_dim + h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		# \mu_{z,t}
		self.enc_mean = nn.Linear(h_dim, z_dim)
		# \sigma_{z,t}
		self.enc_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus()) # SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive.

		# 3. prior (Eq.5)
        # -----------------------------------------------------------------
        # \phi^{prior}_\tau(h)
		self.prior = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		# \mu_{0,t}
		self.prior_mean = nn.Linear(h_dim, z_dim)
		# \sigma_{0,t}: Vector where the i-th element is the std of the i-th dimension.
		self.prior_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus())

		# 4. decoder (Eq.6)
        # -----------------------------------------------------------------
        # \phi^{dec}_\tau()
		self.dec = nn.Sequential(
			nn.Linear(h_dim + h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		# \sigma_{x,t}
		self.dec_std = nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Softplus())
		# \mu_{x,t}
		#self.dec_mean = nn.Linear(h_dim, x_dim)
		self.dec_mean = nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Sigmoid())

		# 5. recurrence (Eq.7)
        # -----------------------------------------------------------------
		self.rnn = nn.GRU(input_size=(h_dim + h_dim), hidden_size=h_dim, num_layers=n_layers, bias=bias)


	def forward(self, x):
		all_enc_mean, all_enc_std = [], []
		all_dec_mean, all_dec_std = [], []
		kld_loss = 0
		rec_loss = 0
		rec_loss_fn = torch.nn.MSELoss(reduction='sum')

		h = torch.zeros(self.n_layers, x.size(1), self.h_dim).to(self.device) # h:(n_layers, batch_size, h_dim), Default: 0
		for t in range(x.size(0)):
			phi_x_t = self.phi_x(x[t]) # x[t]:(batch_size, x_dim)

			# encoder to obtain z_t|x_t (Eq.9)
			enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1)) # h[-1]: The last layer (batch_size, h_dim), other layers are packaged into nn.GRU
			enc_mean_t = self.enc_mean(enc_t)
			enc_std_t = self.enc_std(enc_t)

			# prior to get z_t (Eq.5)
			prior_t = self.prior(h[-1])
			prior_mean_t = self.prior_mean(prior_t)
			prior_std_t = self.prior_std(prior_t)

			# sampling and reparameterization z_t conditioned on x_t (Eq.9)
			z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
			phi_z_t = self.phi_z(z_t) # \phi^z_\tau(z_t)

			# decoder to obtain x_t conditioned on z_t to estimate the Reconstruction Loss (Eq.6)
			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			dec_std_t = self.dec_std(dec_t)

			# recurrence from h_t to h_{t+1} (Eq.7)
			_, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h) # unsqueeze the time_step-dimension to 1 (Here the time_steps are traversed)

			# computing losses [negative ELBO = KL(q||p) + Recon_loss(\hat{x}, x)]
			kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t) # KL(q||p)
			# rec_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t]) # loss for continuous data
			# rec_loss += self._nll_bernoulli(dec_mean_t, x[t]) # loss for binary data

			rec_loss += rec_loss_fn(dec_mean_t, x[t]) # Prevent from NaN [log(0)]

			all_enc_std.append(enc_std_t)
			all_enc_mean.append(enc_mean_t)
			all_dec_mean.append(dec_mean_t)
			all_dec_std.append(dec_std_t)

		return kld_loss, rec_loss, (all_enc_mean, all_enc_std), (all_dec_mean, all_dec_std)


	def sample(self, seq_len):

		sample = torch.zeros(seq_len, self.x_dim).to(self.device)

		h = torch.zeros(self.n_layers, 1, self.h_dim).to(self.device)
		for t in range(seq_len):
			#prior
			prior_t = self.prior(h[-1])
			prior_mean_t = self.prior_mean(prior_t)
			prior_std_t = self.prior_std(prior_t)

			#sampling and reparameterization
			z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
			phi_z_t = self.phi_z(z_t)
			
			#decoder
			dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
			dec_mean_t = self.dec_mean(dec_t)
			#dec_std_t = self.dec_std(dec_t)

			phi_x_t = self.phi_x(dec_mean_t)

			#recurrence
			_, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

			sample[t] = dec_mean_t.data
	
		return sample


	def reset_parameters(self, stdv=1e-1):
		for weight in self.parameters():
			weight.data.normal_(0, stdv)


	def _init_weights(self, stdv):
		pass


	def _reparameterized_sample(self, mean, std):
		"""using std to sample"""
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mean)


	def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
		"""Using std to compute KLD"""

		kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
			(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
			std_2.pow(2) - 1)
		return	0.5 * torch.sum(kld_element)

	def _nll_bernoulli(self, theta, x):
		return -torch.sum(x * torch.log(theta + self.EPS) + (1 - x) * torch.log(1 - theta - self.EPS))

	def _nll_gauss(self, mean, std, x):
		return torch.sum(torch.log(std + EPS) + torch.log(2 * torch.pi) / 2 + (x - mean).pow(2)/(2 * std.pow(2)))


import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

class Encoder1(nn.Module):# qφ(z|x, y)
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(64*32*3, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc31 = nn.Linear(hidden_2, z_dim)
        self.fc32 = nn.Linear(hidden_2, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # put x and y together in the same image for simplification
        xc = x.clone()
        # then compute the hidden units
        hidden = self.relu(self.fc1(xc))
        hidden = self.relu(self.fc2(hidden))      
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc1 = self.fc31(hidden)
        z_scale1 = self.fc32(hidden)
        return z_loc1, z_scale1 # 均值，log方差


class Encoder2(nn.Module):# qφ(z|y)
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc21 = nn.Linear(8, 32)
        self.fc22 = nn.Linear(32, hidden_2)
        self.fc31 = nn.Linear(hidden_2, z_dim)
        self.fc32 = nn.Linear(hidden_2, z_dim)
        self.relu = nn.ReLU()

    def forward(self, y):
        # then compute the hidden units
        hidden2 = self.relu(self.fc21(y))
        hidden2 = self.relu(self.fc22(hidden2))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc2 = self.fc31(hidden2)
        z_scale2 = self.fc32(hidden2)
        return z_loc2, z_scale2 # 均值，log方差

class Decoder1(nn.Module): # reconstruction
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()

        self.decoder1 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(1, 16, 8*3)),
            nn.ConvTranspose2d(1, 3, 2, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 2, stride=2, padding=0),
            nn.ReLU(),
        )

        self.norm = nn.LayerNorm(32*3)
        
    def forward(self, z):
        y = self.decoder1(z)
        
        y = y.squeeze(dim = 1)
        y = y.view(-1, 64, 32*3)
        y = self.norm(y)
        
        return y

class Decoder2(nn.Module): # recongnition
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_2)
        self.fc2 = nn.Linear(hidden_2, 32)
        self.fc3 = nn.Linear(32, 8)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)


    def forward(self, z):
        y = self.relu(self.fc1(z))

        y = self.relu(self.fc2(y))
        y = self.fc3(y)
        y = torch.sum(y, dim=0)
        y = self.softmax(y)
        return y

class CVAE(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        # The CVAE is composed of multiple MLPs, such as recognition network
        # qφ(z|x, y), (conditional) prior network pθ(z|x), and generation
        # network pθ(y|x, z). Also, CVAE is built on top of the NN: not only
        # the direct input x, but also the initial guess y_hat made by the NN
        # are fed into the prior network.
        self.prior_net = Encoder1(z_dim, hidden_1, hidden_2)
        self.generation_net = Encoder2(z_dim, hidden_1, hidden_2)
        self.reconstruction_net = Decoder1(z_dim, hidden_1, hidden_2)
        self.recognition_net = Decoder2(z_dim, hidden_1, hidden_2)

    def model(self, proxy_x, pre_m):
        # This is the generative process with recurrent connection
        condition_loc, condition_scale = self.generation_net(pre_m) # qφ(z|y)

        # sample the handwriting style from the prior distribution, which is
        # modulated by the input xs.
        prior_loc, prior_scale = self.prior_net(proxy_x) # qφ(z|x, y)
        
        # reparameterize as zs
        zs = torch.randn_like(prior_loc) * torch.exp(prior_scale * 0.5) + prior_loc
        # the output x_hat is generated from the distribution pθ(x|z, y)
        x_hat = self.reconstruction_net(zs)

        return x_hat, condition_loc, condition_scale, prior_loc, prior_scale, zs

    def guide(self, zs):
        y = self.recognition_net(zs)

        return y

    def forward(self, proxy_x, pre_m):
        x_hat, condition_loc, condition_scale, prior_loc, prior_scale, zs = self.model(proxy_x, pre_m)

        y = self.guide(zs)

        return x_hat, condition_loc, condition_scale, prior_loc, prior_scale, y



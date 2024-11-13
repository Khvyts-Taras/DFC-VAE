import torch
import torch.nn as nn
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
fe_net1 = vgg16.features[0:4].to(device)
def rec_loss(x, x_hat):
    loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    loss += nn.functional.mse_loss(fe_net1(x_hat), fe_net1(x), reduction='sum')/2
    return loss

def loss_f(x, x_hat, mean, log_var, rec_k = 0.1, kld_k = 10):
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return rec_loss(x, x_hat)*rec_k + KLD*kld_k


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        def _DownBlock(inp_n, out_n):
            block = nn.Sequential(nn.Conv2d(inp_n, out_n, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(0.1),
                                  nn.MaxPool2d(2))
            return block

        self.encoder = nn.Sequential(
            _DownBlock(3, 16),
            _DownBlock(16, 32),
            _DownBlock(32, 64),
            _DownBlock(64, 128),
        )

        self.fc_mean = nn.Linear(128*8*8, latent_dim)
        self.fc_log_var = nn.Linear(128*8*8, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        def _UpBlock(inp_n, out_n):
            block = nn.Sequential(nn.ConvTranspose2d(inp_n, out_n, kernel_size=4, stride=2, padding=1),
                                  nn.LeakyReLU(0.1))
            return block

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*8*8),
            nn.LeakyReLU(0.1),

            nn.Unflatten(1, (128, 8, 8)),
            _UpBlock(128, 64),
            _UpBlock(64, 32),
            _UpBlock(32, 16),
            _UpBlock(16, 3),
            nn.Tanh()
        )

    def forward(self, v):
        x = self.decoder(v)
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparametrization(self, mean, log_var):
        epsilon = torch.randn_like(mean)
        v = mean + torch.exp(0.5*log_var) * epsilon
        return v

    def forward(self, x):
        mean, log_var = self.encoder(x)
        v = self.reparametrization(mean, log_var)
        res = self.decoder(v)

        return res, mean, log_var

    def generate_samples(self, n):
        v = torch.randn(n, self.latent_dim).to(device)
        return self.decoder(v)
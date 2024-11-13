from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import matplotlib.pyplot as plt
from model import *
from torchvision.utils import save_image


latent_dim = 150

model = VAE(latent_dim).to(device)
model.load_state_dict(torch.load('models/vae_cats.pt'))


model.eval()
with torch.no_grad():
    generated = model.generate_samples(64)
    save_image(generated, f'generated.png', nrow=8, normalize=True)


import torch
import torch.nn as nn
from torchvision.utils import save_image
from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_dim = 150
img_size = 128



load = 'pretrained/vae_3_0.pt'
model = VAE(latent_dim).to(device)
if load:
    model.load_state_dict(torch.load(load))

with torch.no_grad():
    noise = torch.rand(64, latent_dim).to(device)
    generated_images = model.decoder(noise)/2+0.5
    save_image(generated_images, 'generated.png')
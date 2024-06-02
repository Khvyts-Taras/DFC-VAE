from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.utils import save_image
import os
import torchvision.models as models
from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_dim = 150
lr = 0.001
epochs = 10
img_size = 128

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((img_size, img_size)),
	transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_dir = '/data/CelebA'
dataset = CelebA(root=data_dir, split='train', transform=transform, download=True)

batch_size = 64
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)




vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT) #pretrained=True
fe_net1 = vgg16.features[0:4].to(device)
def rec_loss(x, x_hat):
	loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
	loss += nn.functional.mse_loss(fe_net1(x_hat), fe_net1(x), reduction='sum')/2

	return loss


rec_k = 0.01
kld_k = 10
def loss_f(x, x_hat, mean, log_var):
	KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

	return rec_loss(x, x_hat)*rec_k + KLD*kld_k



model = VAE(latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)

model.train()
for epoch in range(epochs):
	epoch_loss = 0
	for i, (img, _) in enumerate(tqdm(dataloader, desc=f'Epoch {(epoch+1)}/{epochs}')):
		optimizer.zero_grad()
		img = img.to(device)
		rec, mean, log_var = model(img)
		loss = loss_f(rec, img, mean, log_var)

		epoch_loss += loss.item()

		loss.backward()
		optimizer.step()

		if i%100 == 0:
			with torch.no_grad():
				torch.manual_seed(1)
				noise = torch.rand(batch_size, latent_dim).to(device)
				generated_images = model.decoder(noise)/2+0.5

			save_image(generated_images, f'images/image_{epoch}_{i}.png')
			torch.save(model.state_dict(), f'models/vae_{epoch}_{i}.pt')

	print(f'Epoc: {epoch+1}, Loss: {epoch_loss/len(dataloader.dataset)}')
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import matplotlib.pyplot as plt
from model import *
from torchvision.utils import save_image
import os


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

batch_size = 32
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

model = VAE(latent_dim).to(device)
start_epoch = 0
if start_epoch > 0:
    model.load_state_dict(torch.load(f'models/vae_{start_epoch}.pt'))
    print(f"Loaded from epoch {start_epoch}")

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

        if i % 500 == 0:
            with torch.no_grad():
                original = img[:8]
                reconstructed = rec[:8]
                generated = model.generate_samples(8)

                comparison = torch.cat([original, reconstructed, generated], dim=0)
                save_image(comparison, f'images/comparison_epoch{start_epoch+epoch+1}_step{i}.png', nrow=8, normalize=True)
    
    torch.save(model.state_dict(), f'models/vae_{start_epoch+epoch+1}.pt')
    print(f'Epoch: {epoch+1}, Loss: {epoch_loss/len(dataloader.dataset)}')


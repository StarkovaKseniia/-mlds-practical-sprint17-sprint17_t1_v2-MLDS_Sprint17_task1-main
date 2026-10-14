import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_size = 64
hidden_size = 256
image_size = 28 * 28
batch_size = 100
lr = 0.0002
epochs = 10

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Generator network
class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img


# Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        return out


# Initialize networks
generator = Generator(latent_size).to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

# Loss function
criterion = nn.BCELoss()


# Helper function to generate images
def generate_images(generator, num_images):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_size).to(device)
        fake_images = generator(z)
        return fake_images.view(num_images, 28, 28)


# Training loop
def train_gan(generator, discriminator, train_loader, epochs):
    for epoch in range(epochs):
        for i, (images, _) in enumerate(train_loader):
            batch_size = images.size(0)
            
            # Конфігурація міток для реальних та фейкових зображень
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # --- 1. Тренування Дискримінатора ---
            optimizer_d.zero_grad()
            
            # Реальні зображення
            real_images = images.view(batch_size, -1).to(device)
            outputs_real = discriminator(real_images)
            loss_d_real = criterion(outputs_real, real_labels)
            
            # Фейкові зображення від Генератора
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = generator(z)
            outputs_fake = discriminator(fake_images.detach())
            loss_d_fake = criterion(outputs_fake, fake_labels)
            
            # Загальна втрата дискримінатора та крок оптимізації
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()
            
            # --- 2. Тренування Генератора ---
            optimizer_g.zero_grad()
            
            # Генератор намагається обдурити дискримінатор (мітка стає real_labels)
            outputs = discriminator(fake_images)
            loss_g = criterion(outputs, real_labels)
            
            loss_g.backward()
            optimizer_g.step()
            
        # Виведення логів після кожної епохи
        print(f"Epoch [{epoch+1}/{epochs}], Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")


train_gan(generator, discriminator, train_loader, epochs)

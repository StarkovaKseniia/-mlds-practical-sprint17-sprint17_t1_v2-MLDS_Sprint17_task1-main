import torch
import torch.nn as nn


from src.GAN_for_data_augmentation import Generator, Discriminator


# Test 1: Check if the GAN model's generator can create an image tensor of the correct shape (e.g., 28x28).
def test_generator_output_shape():
    latent_size = 64
    generator = Generator(latent_size)
    z = torch.randn(1, latent_size)
    generated_image = generator(z)
    assert generated_image.shape == (
    1, 28 * 28), f"Generator output shape mismatch. Expected (1, 28 * 28), got {generated_image.shape}"


# Test 2: Verify that the discriminator correctly classifies real vs. generated images.
def test_discriminator_classification():
    latent_size = 64
    batch_size = 10
    generator = Generator(latent_size)
    discriminator = Discriminator()

    # Optimizers and loss
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = nn.BCELoss()

    # Generate fake images
    z = torch.randn(batch_size, latent_size)
    fake_images = generator(z)

    # Create random real images (similar size to MNIST)
    real_images = torch.randn(batch_size, 28 * 28)

    # Training the discriminator slightly
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    optimizer_d.zero_grad()
    real_classifications = discriminator(real_images)
    fake_classifications = discriminator(fake_images)

    real_loss = criterion(real_classifications, real_labels)
    fake_loss = criterion(fake_classifications, fake_labels)
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_d.step()

    # Run assertions
    assert torch.mean(real_classifications).item() > 0.4, "Discriminator classifying real images incorrectly."
    assert torch.mean(fake_classifications).item() < 0.5 + 0.05, "Discriminator classifying fake images incorrectly."


# Test 3: Ensure the generator’s loss decreases over time.
def test_generator_loss_decreases():
    latent_size = 64
    generator = Generator(latent_size)
    discriminator = Discriminator()
    criterion = nn.BCELoss()

    z = torch.randn(10, latent_size)
    fake_images = generator(z)
    real_labels = torch.ones(10, 1)

    # Initial loss before training
    initial_loss = criterion(discriminator(fake_images), real_labels).item()

    # Simulate training step for the generator
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    for _ in range(100):  # Increase the steps of optimization
        optimizer_g.zero_grad()
        outputs = discriminator(generator(z))
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

    # Loss after some training
    final_loss = criterion(discriminator(generator(z)), real_labels).item()

    # Run assertion
    assert final_loss < initial_loss, f"Generator's loss did not decrease during training. Initial: {initial_loss}, Final: {final_loss}"

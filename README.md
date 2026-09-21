# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset


## DESIGN STEPS
### STEP 1: 
Import the required libraries, download the MNIST dataset, convert the images into tensors, and create training and testing data loaders.
### STEP 2: 
Generate random noise and add it to the original MNIST images. Clip the pixel values between 0 and 1 to obtain valid noisy images.


### STEP 3: 
Create an encoder using convolutional layers to extract features and reduce the image size from 28 × 28 to 7 × 7. Create a decoder using transposed convolutional layers to reconstruct the image back to 28 × 28.


### STEP 4: 
Use the noisy images as input and the original clean images as target output. Train the model using MSELoss and the Adam optimizer for 5 epochs.


### STEP 5: 

Calculate the reconstruction error during training and display the loss for each epoch. Plot the training-loss graph to observe the learning performance of the model.

### STEP 6: 

Display the Original, Noisy, and Denoised images side-by-side. Compare the reconstructed images with the original images to evaluate the effectiveness of the autoencoder in removing noise.



## PROGRAM

### Name: Kabira A

### Register Number: 212224040146

```
# ============================================================
# DL - CONVOLUTIONAL AUTOENCODER FOR IMAGE DENOISING
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ============================================================
# STUDENT DETAILS
# ============================================================

NAME = "RAGASUDHA R"
REGISTER_NUMBER = "212224230215"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# LOAD MNIST DATASET
# ============================================================

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=10,
    shuffle=False
)


# ============================================================
# ADD NOISE FUNCTION
# ============================================================

def add_noise(images, noise_factor=0.4):

    noise = torch.randn_like(images) * noise_factor

    noisy_images = images + noise

    noisy_images = torch.clamp(
        noisy_images,
        0.0,
        1.0
    )

    return noisy_images


# ============================================================
# DENOISING AUTOENCODER MODEL
# ============================================================

class DenoisingAutoencoder(nn.Module):

    def __init__(self):

        super(DenoisingAutoencoder, self).__init__()

        # -------------------------
        # ENCODER
        # -------------------------

        self.encoder = nn.Sequential(

            # [B, 1, 28, 28]
            #       ↓
            # [B, 16, 14, 14]

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            # [B, 16, 14, 14]
            #       ↓
            # [B, 32, 7, 7]

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU()
        )


        # -------------------------
        # DECODER
        # -------------------------

        self.decoder = nn.Sequential(

            # [B, 32, 7, 7]
            #       ↓
            # [B, 16, 14, 14]

            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            nn.ReLU(),

            # [B, 16, 14, 14]
            #       ↓
            # [B, 1, 28, 28]

            nn.ConvTranspose2d(
                16,
                1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            nn.Sigmoid()
        )


    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x


# ============================================================
# INITIALIZE MODEL
# ============================================================

model = DenoisingAutoencoder().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("-" * 75)
print(
    f"{'Layer (type)':<30}"
    f"{'Output Shape':<25}"
    f"{'Param #':>12}"
)
print("=" * 75)

print(
    f"{'Conv2d-1':<30}"
    f"{'[-1, 16, 14, 14]':<25}"
    f"{160:>12,}"
)

print(
    f"{'ReLU-2':<30}"
    f"{'[-1, 16, 14, 14]':<25}"
    f"{0:>12,}"
)

print(
    f"{'Conv2d-3':<30}"
    f"{'[-1, 32, 7, 7]':<25}"
    f"{4640:>12,}"
)

print(
    f"{'ReLU-4':<30}"
    f"{'[-1, 32, 7, 7]':<25}"
    f"{0:>12,}"
)

print(
    f"{'ConvTranspose2d-5':<30}"
    f"{'[-1, 16, 14, 14]':<25}"
    f"{4624:>12,}"
)

print(
    f"{'ReLU-6':<30}"
    f"{'[-1, 16, 14, 14]':<25}"
    f"{0:>12,}"
)

print(
    f"{'ConvTranspose2d-7':<30}"
    f"{'[-1, 1, 28, 28]':<25}"
    f"{145:>12,}"
)

print(
    f"{'Sigmoid-8':<30}"
    f"{'[-1, 1, 28, 28]':<25}"
    f"{0:>12,}"
)

print("=" * 75)

print("Total params: 9,569")
print("Trainable params: 9,569")
print("Non-trainable params: 0")

print("-" * 75)

print("Input size (MB): 0.00")
print("Forward/backward pass size (MB): 0.25")
print("Params size (MB): 0.04")
print("Estimated Total Size (MB): 0.29")

print("-" * 75)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train(
    model,
    loader,
    criterion,
    optimizer,
    epochs=5
):

    model.train()

    loss_history = []

    for epoch in range(epochs):

        running_loss = 0.0

        for images, _ in loader:

            images = images.to(device)

            # Add noise
            noisy_images = add_noise(
                images
            )

            # Forward pass
            outputs = model(
                noisy_images
            )

            # Calculate loss
            loss = criterion(
                outputs,
                images
            )

            # Backward pass
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()


        # Average loss
        epoch_loss = (
            running_loss /
            len(loader)
        )

        loss_history.append(
            epoch_loss
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {epoch_loss:.4f}"
        )

    return loss_history


# ============================================================
# TRAIN THE MODEL
# ============================================================

loss_history = train(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs=5
)


# ============================================================
# STUDENT DETAILS
# ============================================================

print()

print("Name:", NAME)

print(
    "Register Number:",
    REGISTER_NUMBER
)


# ============================================================
# TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 6),
    loss_history,
    marker="o"
)

plt.title(
    "Training Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# VISUALIZATION FUNCTION
# ============================================================

def visualize_denoising(
    model,
    loader,
    num_images=10
):

    model.eval()

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            # Create noisy images
            noisy_images = add_noise(
                images
            )

            # Reconstruct images
            outputs = model(
                noisy_images
            )

            break


    # Take only required images
    images = images[:num_images]

    noisy_images = noisy_images[:num_images]

    outputs = outputs[:num_images]


    # Convert to NumPy
    images = images.cpu().numpy()

    noisy_images = (
        noisy_images.cpu().numpy()
    )

    outputs = outputs.cpu().numpy()


    # --------------------------------------------------------
    # DISPLAY STUDENT DETAILS
    # --------------------------------------------------------

    print(
        "Name:", NAME
    )

    print(
        "Register Number:",
        REGISTER_NUMBER
    )


    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    plt.figure(
        figsize=(18, 6)
    )


    for i in range(num_images):

        # -------------------------
        # ORIGINAL
        # -------------------------

        ax = plt.subplot(
            3,
            num_images,
            i + 1
        )

        plt.imshow(
            images[i].squeeze(),
            cmap="gray"
        )

        ax.set_title(
            "Original"
        )

        plt.axis("off")


        # -------------------------
        # NOISY
        # -------------------------

        ax = plt.subplot(
            3,
            num_images,
            i + 1 + num_images
        )

        plt.imshow(
            noisy_images[i].squeeze(),
            cmap="gray"
        )

        ax.set_title(
            "Noisy"
        )

        plt.axis("off")


        # -------------------------
        # DENOISED
        # -------------------------

        ax = plt.subplot(
            3,
            num_images,
            i + 1 + 2 * num_images
        )

        plt.imshow(
            outputs[i].squeeze(),
            cmap="gray"
        )

        ax.set_title(
            "Denoised"
        )

        plt.axis("off")


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    plt.suptitle(
        "Original vs Noisy Vs Reconstructed Image",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# DISPLAY DENOISING RESULTS
# ============================================================

visualize_denoising(
    model,
    test_loader,
    num_images=10
)

```

### OUTPUT

### Model Summary



<img width="1030" height="557" alt="image" src="https://github.com/user-attachments/assets/cc97ce1d-acff-4b3d-9c58-ecd7708c6a4b" />


### Training loss



<img width="1001" height="541" alt="image" src="https://github.com/user-attachments/assets/f62c22c4-4e66-4d95-83c2-8658a271e591" />


## Original vs Noisy Vs Reconstructed Image

<img width="1077" height="375" alt="image" src="https://github.com/user-attachments/assets/44e24502-699b-4ff2-bb99-8eb438d39c21" />

## RESULT


Thus, a convolutional autoencoder for image denoising was developed and trained successfully.

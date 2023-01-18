import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_input=784 + 10, n_hidden=1000, latent_dims=20):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.fc_mu = nn.Linear(n_hidden, latent_dims)
        self.fc_var = nn.Linear(n_hidden, latent_dims)

    def forward(self, x, condition):

        x = torch.cat([x, condition], dim=1)
        x = self.layers(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, n_output=784, n_hidden=1000, latent_dims=20 + 10):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dims, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
            nn.Sigmoid(),
        )

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        return self.layers(x)


class ConditionalVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = Encoder()
        self.decode = Decoder()

    def loss_function(self, x, x_hat, mu, log_var):
        reconstruction_loss = nn.functional.binary_cross_entropy(
            x_hat, x, reduction="sum"
        )
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return reconstruction_loss + kl_divergence

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(
            mu, log_var
        )  # Equivilant to sampling from a normal distribution
        return self.decode(z, c), mu, log_var


from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np


def inference(model, c, n=10):
    z = torch.randn(n, 20).to("cuda")
    x_hat = model.decode(z, c)
    return x_hat


def plot_batch_and_save_image(x_hat, epoch):
    fig, ax = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        ax[i].imshow(x_hat[i].view(28, 28).cpu().detach().numpy())
        # add epoch number to the title
        ax[i].set_title(f"Number: {i}")
        ax[i].axis("off")

    plt.savefig(f"results/epoch-{epoch}.png")


if __name__ == "__main__":

    def get_data():
        train_ds = MNIST(
            root="data", train=True, download=True, transform=transforms.ToTensor()
        )

        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

        return train_dl

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(DEVICE)
    data_loader = get_data()
    model = ConditionalVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)

    EPOCHS = 50

    for epoch in range(EPOCHS):

        with tqdm(data_loader) as data_loader:
            for batch in data_loader:
                x, y = batch
                c = torch.nn.functional.one_hot(y, 10).float().to(DEVICE)

                x = x.view(x.shape[0], -1).to(DEVICE)

                x_hat, mu, log_var = model(x, c)

                loss = model.loss_function(x, x_hat, mu, log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            random_nums = torch.arange(10)
            random_nums = (
                torch.nn.functional.one_hot(random_nums, 10).float().to(DEVICE)
            )
            x_hat = inference(model, random_nums)
            plot_batch_and_save_image(x_hat, epoch)

        print(f"Epoch {epoch} Loss: {loss}")

    model.to("cpu")
    torch.save(model.state_dict(), "model.pth")

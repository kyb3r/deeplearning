from torch.utils.data import DataLoader, Dataset, random_split

import torch
from torch import nn

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from fastai.data.all import untar_data

# Load the dataset into a Pandas DataFrame
url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
path = untar_data(url)

import pandas as pd

ratings = pd.read_csv(path / "ratings.csv")

num_movies = ratings.movieId.nunique()
num_users = ratings.userId.nunique()

# Make a mapping of movieID to index starting from 0
movie2idx = {o: i for i, o in enumerate(ratings.movieId.unique())}
# ratings.movieId = ratings.movieId.apply(lambda x: movie2idx[x])

# Make a mapping of userID to index starting from 0
user2idx = {o: i for i, o in enumerate(ratings.userId.unique())}
# ratings.userId = ratings.userId.apply(lambda x: user2idx[x])


# Construct pytorch dataset of users and movies with ratings
class MovieLensDataset(Dataset):
    def __init__(self, ratings):
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        x = self.ratings.iloc[idx]
        return (
            user2idx[int(x.userId)],
            movie2idx[int(x.movieId)],
            torch.tensor(x.rating, dtype=torch.float32).unsqueeze(-1),
        )


dataset = MovieLensDataset(ratings[:500000])

train_ds, val_ds = random_split(dataset, [0.9, 0.1])
train_dataloader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=128, shuffle=True)


# Create a model
class CollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=50):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)

    def forward(self, user_ids, movie_ids):
        user_factors = self.user_factors(user_ids)
        movie_factors = self.movie_factors(movie_ids)

        affinities = (user_factors * movie_factors).sum(dim=1)
        return affinities.view(-1, 1)


class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=50, n_hidden=500):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)

        self.layers = nn.Sequential(
            nn.Linear(2 * n_factors, n_hidden),
            nn.Dropout(0.4),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 250),
            nn.Dropout(0.4),
            nn.LayerNorm(250),
            nn.ReLU(),
            nn.Linear(250, 1),
        )

    def forward(self, user_ids, movie_ids):
        user_factors = self.user_factors(user_ids)
        movie_factors = self.movie_factors(movie_ids)

        x = torch.cat([user_factors, movie_factors], dim=1)
        return torch.sigmoid(self.layers(x)) * 5.5


model = NeuralCollaborativeFiltering(num_users, num_movies).cuda()
optim = torch.optim.Adam(model.parameters(), lr=1e-2)

all_train_losses = []
all_val_losses = []

for epoch in range(10):
    train_losses = []
    for user_ids, movie_ids, ratings in tqdm(train_dataloader):
        scores = model(user_ids.cuda(), movie_ids.cuda())
        loss = nn.functional.mse_loss(scores, ratings.cuda())
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_losses.append(loss.item())

    with torch.no_grad():
        val_losses = []
        model.eval()
        for user_ids, movie_ids, ratings in val_dataloader:
            scores = model(user_ids.cuda(), movie_ids.cuda())
            loss = nn.functional.mse_loss(scores, ratings.cuda())
            val_losses.append(loss.item())
        model.train()

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    all_train_losses.append(train_loss)
    all_val_losses.append(val_loss)
    print(f"Epoch: {epoch}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Val loss: {val_loss:.4f}")
    # plot and save train loss image

plt.plot(all_train_losses[3:], label="train")
plt.plot(all_val_losses[3:], label="val")
plt.savefig("train_loss-non-neural.png")


# Save the model

from torch.utils.data import DataLoader

import torch
from torch import nn

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from fastai.data.all import untar_data

# Load the dataset into a Pandas DataFrame
url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
path = untar_data(url)

print(path)


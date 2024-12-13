# Dataset
# https://zenodo.org/records/4835108 -> Audio Data .flac
# https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz -> metadata

# Tutorials etc.
# https://github.com/piotrkawa/audio-deepfake-adversarial-attacks/blob/main/src/datasets/deepfake_asvspoof_dataset.py
# https://www.youtube.com/watch?v=gfhx4dr6gJQ&list=PLhA3b2k8R3t2Ng1WW_7MiXeh1pfQJQi_P&index=9
# https://www.kaggle.com/code/utcarshagrawal/birdclef-audio-pytorch-tutorial
# https://learn.microsoft.com/en-us/training/modules/intro-audio-classification-pytorch/4-speech-model
# https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html#define-the-network
# https://www.kaggle.com/code/utcarshagrawal/birdclef-audio-pytorch-tutorial/notebook
# https://www.youtube.com/watch?v=gfhx4dr6gJQ&list=PLhA3b2k8R3t2Ng1WW_7MiXeh1pfQJQi_P&index=9


# Import packages and custom functions
import numpy as np
import torch
import torch.multiprocessing
from torchvision import datasets, transforms
import torchvision
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

import sys

from src.metrics import acc
from src import utils
from src.utils.config import config

# Variables
NUM_WORKERS = 0
DATASET_PATH = 'dataset'
batch_size = 16

# Define the data transformation -> https://pytorch.org/audio/stable/transforms.html
transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])


def image_display_spectrogram(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


"""
Display all the spectrogram of sounds within a batch
@param batches: Batch of data from a dataloader
"""


def batches_display(batches):
    dataiter = iter(batches)
    images, _ = next(dataiter)
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    # show images
    image_display_spectrogram(img_grid, one_channel=False)


def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=True, figsize=(20, 10))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        axes[x].set_title(list(signals.keys())[i])
        axes[x].plot(list(signals.values())[i])
        axes[x].get_xaxis().set_visible(False)
        axes[x].get_yaxis().set_visible(False)
        i += 1

# main에서 이거씀
def dataset(device):
    output_path_f = '../dataset/fake'
    output_path_r = '../dataset/real'
    print(f'Length of fake dataset: {len(os.listdir(output_path_f))}')
    print(f'Length of real dataset: {len(os.listdir(output_path_r))}')

    # Load the dataset
    print(f"Loading images from dataset at {DATASET_PATH}")
    dataset = torchvision.datasets.ImageFolder('../dataset', transform=transform)

    # train / test split
    val_ratio = 0.2
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"{train_size} images for training, {val_size} images for validation")

    # Load training dataset into batches
    train_batches = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                drop_last=True,
                                                shuffle=False,
                                                num_workers=NUM_WORKERS)
    # Load validation dataset into batches
    val_batches = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=batch_size * 2,
                                              shuffle=True,
                                              drop_last=False,
                                              num_workers=NUM_WORKERS)

    # display 32 (batch_size*2) sample from the first validation batch
    batches_display(val_batches)

    return train_batches, val_batches
import torch
import random

class config:
    seed=2022
    num_fold = 5
    sample_rate= 6000
    hop_length=512
    n_mels=64
    duration=5
    num_classes = 2
    batch_size = 16
    model_name = 'hagishiro'
    epochs = 10
    learning_rate = 1e-4
    nfilt=26
    nfeat=13
    nfft=512
    rate=16000
    step = int (rate/10)
    num_workers = 0
    randomseed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    random.seed(randomseed)
    torch.manual_seed(randomseed)
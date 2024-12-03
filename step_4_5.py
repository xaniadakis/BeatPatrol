from utils import plot_label_freq, GenreClassifier, SpectrogramDataset, CLASS_MAPPING, get_device, train_lstm_classifier
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import copy
import os
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim


DATA_PATH = "/home/alex/Downloads/archive(1)/data/"
EPOCHS = 1
LR = 0.01
BATCH_SIZE = 32

if __name__ == "__main__":
    print("Loading Data")
    full_dataset = SpectrogramDataset(DATA_PATH + "fma_genre_spectrograms/", class_mapping=None, train=True)
    mel_specs_genre = SpectrogramDataset(DATA_PATH + "fma_genre_spectrograms/", class_mapping=CLASS_MAPPING, train=True)

    plot_label_freq(mel_specs_genre, full_dataset, save_title="spectogram_genre_labels.png")

    train_dl = DataLoader(mel_specs_genre, batch_size=BATCH_SIZE, shuffle=True)

    model = GenreClassifier(input_dim=mel_specs_genre.feat_dim, num_classes=len(set(mel_specs_genre.labels)), 
                        rnn_size=64, num_layers=1, bidirectional=False)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)

    #device = get_device()
    device = torch.device("cpu")
    print(f"Using {device}")
    model.to(device)
    mel_spec_genre_losses = train_lstm_classifier(epochs=EPOCHS, model=model, dataloader=train_dl, 
                          device=device, optimizer=optimizer, criterion=nn.CrossEntropyLoss())
    
    """beat_mel_specs_genre = SpectrogramDataset(DATA_PATH + 'fma_genre_spectrograms_beat/', class_mapping=CLASS_MAPPING, train=True, feat_type='mel', max_length=-1)
    
    train_loader_beat_mel, val_loader_beat_mel = torch_train_val_split(beat_mel_specs, 32 ,32, val_size=.33)"""



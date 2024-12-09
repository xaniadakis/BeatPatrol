import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import copy
import os
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, precision_recall_fscore_support
from time import time
from scipy.stats import spearmanr
import torch
import gc
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple,trunc_normal_
#from torch.amp import autocast
from tqdm import tqdm


CLASS_MAPPING = {
    "Rock": "Rock",
    "Psych-Rock": "Rock",
    "Indie-Rock": None,
    "Post-Rock": "Rock",
    "Psych-Folk": "Folk",
    "Folk": "Folk",
    "Metal": "Metal",
    "Punk": "Metal",
    "Post-Punk": None,
    "Trip-Hop": "Trip-Hop",
    "Pop": "Pop",
    "Electronic": "Electronic",
    "Hip-Hop": "Hip-Hop",
    "Classical": "Classical",
    "Blues": "Blues",
    "Chiptune": "Electronic",
    "Jazz": "Jazz",
    "Soundtrack": None,
    "International": None,
    "Old-Time": None,
}


def torch_train_val_split(
    dataset, batch_train, batch_eval, val_size=0.2, shuffle=True, seed=420
):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_eval, sampler=val_sampler)
    return train_loader, val_loader


def read_spectrogram(spectrogram_file, feat_type):
    spectrogram = np.load(spectrogram_file)
    # spectrograms contains a fused mel spectrogram and chromagram    
    if feat_type=='mel':
        return spectrogram[:128, :].T
    elif feat_type=='chroma':
        return spectrogram[128:, :].T

    return spectrogram.T


class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[: self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1

class MultiTaskDataset(Dataset):
    def __init__(self, features, labels, max_length=None):
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.lengths = [f.shape[0] for f in features]
        self.max_length = max(self.lengths) if max_length is None else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.zero_pad_and_stack(self.features[idx]), dtype=torch.float32)
        label = self.labels[idx]
        length = min(self.lengths[idx], self.max_length)
        return features, label, length


class SpectrogramDataset(Dataset):
    def __init__(
        self, path, class_mapping=None, train=True, feat_type='mel', max_length=-1, regression=None
    ):
        t = "train" if train else "test"
        p = os.path.join(path, t)
        self.regression = regression

        self.full_path = p
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_labels(self.index, class_mapping)
        self.feats = [read_spectrogram(os.path.join(p, f), feat_type) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        self.label_transformer = LabelTransformer()
        if isinstance(labels, (list, tuple)):
            if not regression:
                self.labels = np.array(
                    self.label_transformer.fit_transform(labels)
                ).astype("int64")
            else:
                self.labels = np.array(labels).astype("float64")

    def get_files_labels(self, txt, class_mapping):
        with open(txt, "r") as fd:
            lines = [l.rstrip().split("\t") for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            if self.regression:
                l = l[0].split(",")
                files.append(l[0] + ".fused.full.npy")
                labels.append(l[self.regression])
                continue
            label = l[1]
            if class_mapping:
                label = class_mapping[l[1]]
            if not label:
                continue
            fname = l[0]
            if fname.endswith(".gz"):
                fname = ".".join(fname.split(".")[:-1])
            
            # necessary fixes for the custom dataset used in the lab
            if 'fma_genre_spectrograms_beat' in self.full_path.split('/'):
                fname = fname.replace('beatsync.fused', 'fused.full')            
            if 'test' in self.full_path.split('/'):
                fname = fname.replace('full.fused', 'fused.full')
            
            files.append(fname)
            labels.append(label)
        return files, labels

    def __getitem__(self, item):
        length = min(self.lengths[item], self.max_length)
        features = self.zero_pad_and_stack(self.feats[item])
        label = self.labels[item]

        # Ensure features are float
        features = torch.tensor(features, dtype=torch.float)

        return features, label, length


    def __len__(self):
        return len(self.labels)
    

def plot_label_freq(merged_dataset, full_dataset, save_title):
    # calculate label frequencies
    merged_label_counts = Counter(merged_dataset.labels)
    full_label_counts = Counter(full_dataset.labels)

    # extract data for  plot
    merged_labels, merged_frequencies = zip(*sorted(merged_label_counts.items()))
    full_labels, full_frequencies = zip(*sorted(full_label_counts.items()))

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    axes[0].bar(range(len(merged_labels)), merged_frequencies, color='blue', alpha=0.7)
    axes[0].set_title("Merged Dataset Label Distribution")
    axes[0].set_xlabel("Labels")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xticks(range(len(merged_labels)))
    axes[0].set_xticklabels(
        merged_dataset.label_transformer.inverse_transform(merged_labels), rotation=45
    )

    axes[1].bar(range(len(full_labels)), full_frequencies, color='green', alpha=0.7)
    axes[1].set_title("Full Dataset Label Distribution")
    axes[1].set_xlabel("Labels")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xticks(range(len(full_labels)))
    axes[1].set_xticklabels(
        full_dataset.label_transformer.inverse_transform(full_labels), rotation=45
    )

    plt.tight_layout()
    plt.savefig(f"assets/{save_title}", dpi=300)
    plt.show()

class PadPackedSequence(nn.Module):
    def __init__(self):
        """Wrap sequence padding in nn.Module
        Args:
            batch_first (bool, optional): Use batch first representation. Defaults to True.
        """
        super(PadPackedSequence, self).__init__()
        self.batch_first = True
        self.max_length = None

    def forward(self, x):
        """Convert packed sequence to padded sequence
        Args:
            x (torch.nn.utils.rnn.PackedSequence): Packed sequence
        Returns:
            torch.Tensor: Padded sequence
        """
        out, lengths = pad_packed_sequence(
            x, batch_first=self.batch_first, total_length=self.max_length  # type: ignore
        )
        lengths = lengths.to(out.device)
        return out, lengths  # type: ignore


class PackSequence(nn.Module):
    def __init__(self):
        """Wrap sequence packing in nn.Module
        Args:
            batch_first (bool, optional): Use batch first representation. Defaults to True.
        """
        super(PackSequence, self).__init__()
        self.batch_first = True

    def forward(self, x, lengths):
        """Pack a padded sequence and sort lengths
        Args:
            x (torch.Tensor): Padded tensor
            lengths (torch.Tensor): Original lengths befor padding
        Returns:
            Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor]: (packed sequence, sorted lengths)
        """
        lengths = lengths.to("cpu")
        out = pack_padded_sequence(
            x, lengths, batch_first=self.batch_first, enforce_sorted=False
        )

        return out


class LSTMBackbone(nn.Module):
    def __init__(
        self,
        input_dim,
        rnn_size=128,
        num_layers=1,
        bidirectional=False,
        dropout=0.1,
    ):
        super(LSTMBackbone, self).__init__()
        self.batch_first = True
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        self.input_dim = input_dim
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.hidden_size = rnn_size
        self.pack = PackSequence()
        self.unpack = PadPackedSequence()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, lengths):
        """LSTM forward
        Args:
            x (torch.Tensor):
                [B, S, F] Batch size x sequence length x feature size
                padded inputs
            lengths (torch.tensor):
                [B] Original lengths of each padded sequence in the batch
        Returns:
            torch.Tensor:
                [B, H] Batch size x hidden size lstm last timestep outputs
                2 x hidden_size if bidirectional
        """
        packed = self.pack(x, lengths)
        output, _ = self.lstm(packed)
        output, lengths = self.unpack(output)
        output = self.drop(output)

        rnn_all_outputs, last_timestep = self._final_output(output, lengths)
        # Use the last_timestep for classification / regression
        # Alternatively rnn_all_outputs can be used with an attention mechanism
        return last_timestep

    def _merge_bi(self, forward, backward):
        """Merge forward and backward states
        Args:
            forward (torch.Tensor): [B, L, H] Forward states
            backward (torch.Tensor): [B, L, H] Backward states
        Returns:
            torch.Tensor: [B, L, 2*H] Merged forward and backward states
        """
        return torch.cat((forward, backward), dim=-1)

    def _final_output(self, out, lengths):
        """Create RNN ouputs
        Collect last hidden state for forward and backward states
        Code adapted from https://stackoverflow.com/a/50950188
        Args:
            out (torch.Tensor): [B, L, num_directions * H] RNN outputs
            lengths (torch.Tensor): [B] Original sequence lengths
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (
                merged forward and backward states [B, L, H] or [B, L, 2*H],
                merged last forward and backward state [B, H] or [B, 2*H]
            )
        """

        if not self.bidirectional:
            return out, self._select_last_unpadded(out, lengths)

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])
        # Last backward corresponds to first token
        last_backward_out = backward[:, 0, :] if self.batch_first else backward[0, ...]
        # Last forward for real length or seq (unpadded tokens)
        last_forward_out = self._select_last_unpadded(forward, lengths)
        out = self._merge_bi(forward, backward)

        return out, self._merge_bi(last_forward_out, last_backward_out)

    def _select_last_unpadded(self, out, lengths):
        """Get the last timestep before padding starts
        Args:
            out (torch.Tensor): [B, L, H] Fprward states
            lengths (torch.Tensor): [B] Original sequence lengths
        Returns:
            torch.Tensor: [B, H] Features for last sequence timestep
        """
        gather_dim = 1  # Batch first
        gather_idx = (
            (lengths - 1)  # -1 to convert to indices
            .unsqueeze(1)  # (B) -> (B, 1)
            .expand((-1, self.hidden_size))  # (B, 1) -> (B, H)
            # (B, 1, H) if batch_first else (1, B, H)
            .unsqueeze(gather_dim)
        )
        # Last forward for real length or seq (unpadded tokens)
        last_out = out.gather(gather_dim, gather_idx).squeeze(gather_dim)

        return last_out
    

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopper:
    def __init__(self, model, save_path, patience=1, min_delta=0):
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(self.model.state_dict(), self.save_path)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_one_epoch(model, train_loader, optimizer, device, regression_flag):
    model.train()
    total_loss = 0
    for x, y, lengths in train_loader:
        if regression_flag:
            loss, logits = model(x.float().to(device), y.float().to(device), lengths.to(device))
        else:      
            loss, logits = model(x.float().to(device), y.to(device), lengths.to(device))
        # prepare
        optimizer.zero_grad()
        # backward
        loss.backward()
        # optimizer step
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)    
    return avg_loss


def validate_one_epoch(model, val_loader, device, regression_flag):    
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y, lengths in val_loader:
            if regression_flag:
                loss, logits = model(x.float().to(device), y.float().to(device), lengths.to(device))
            else:
                loss, logits = model(x.float().to(device), y.to(device), lengths.to(device))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)    
    return avg_loss

def overfit_with_a_couple_of_batches(model, train_loader, optimizer, device, regression_flag, epochs=400):
    print('Training in overfitting mode...')
    # get only the 1st batch
    x_b1, y_b1, lengths_b1 = next(iter(train_loader))
    model.train()
    for epoch in range(epochs):
        if regression_flag:      
            loss, logits = model(x_b1.float().to(device), y_b1.float().to(device), lengths_b1.to(device))
        else:
            loss, logits = model(x_b1.float().to(device), y_b1.to(device), lengths_b1.to(device))
        # prepare
        optimizer.zero_grad()
        # backward
        loss.backward()
        # optimizer step
        optimizer.step()

        if epoch == 0 or (epoch+1)%20 == 0:
            print(f'Epoch {epoch+1}, Loss at training set: {loss.item()}')

def train(model, train_loader, val_loader, optimizer, epochs=400, save_path='checkpoint.pth', device="cuda",
          overfit_batch=False, regression_flag=False, patience=5):
    if overfit_batch:
        overfit_with_a_couple_of_batches(model, train_loader, optimizer, device, regression_flag, epochs=epochs)
        return
    else:
        print(f'Training started for model {save_path.replace(".pth", "")}...')
        early_stopper = EarlyStopper(model, save_path, patience=patience)
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            start_train = time()
            train_loss = train_one_epoch(model, train_loader, optimizer, device, regression_flag)
            end_train = time()
            train_losses.append(train_loss)
            if val_loader is not None:
                start_val = time()
                validation_loss = validate_one_epoch(model, val_loader, device, regression_flag)
                end_val = time()
                val_losses.append(validation_loss)
                print(f'Epoch {epoch+1}/{epochs}\n\tAverage Training Loss: {train_loss} ({end_train - start_train:.2f}s)\n\tAverage Validation Loss: {validation_loss}({end_val - start_val:.2f}s)')          
            
            if early_stopper.early_stop(validation_loss):
                print('Early Stopping was activated.')
                print('Training has been completed.\n')
                break
    return train_losses, val_losses

class Classifier(nn.Module):
    def __init__(self, num_classes, backbone):
        """
        backbone (nn.Module): The nn.Module to use for spectrogram parsing
        num_classes (int): The number of classes
        """
        super(Classifier, self).__init__()
        self.backbone = backbone  # An LSTMBackbone or CNNBackbone
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, num_classes)
        self.criterion = nn.CrossEntropyLoss() if num_classes > 1 else nn.MSELoss()

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        logits = self.output_layer(feats)
        logits = logits.squeeze(-1) if self.criterion.__class__ == nn.modules.loss.MSELoss else logits
        loss = self.criterion(logits, targets)
        return loss, logits

class Regressor(nn.Module):
    def __init__(self, backbone):
        super(Regressor, self).__init__()
        self.backbone = backbone
        self.is_lstm = isinstance(self.backbone, LSTMBackbone)
        self.output_layer = nn.Linear(self.backbone.feature_size, 1)
        self.criterion = nn.MSELoss()  # Loss function for regression

    def forward(self, x, targets, lengths):
        feats = self.backbone(x) if not self.is_lstm else self.backbone(x, lengths)
        out = self.output_layer(feats).squeeze(-1)
        loss = self.criterion(out.float(), targets.float())
        return loss, out


from scipy.stats import spearmanr, pearsonr

def test_model(model, dataloader, device, regression_flag=False):
    model.eval()
    y_true, y_pred, spear_corrs, pear_corrs = [], [], [], []
    with torch.no_grad():
        for x, labels, lengths in dataloader:
            _, logits = model(x.float().to(device), labels.to(device), lengths.to(device))
            if not regression_flag:
                y_pred.append(logits.argmax(dim=-1).cpu())
                y_true.append(labels.cpu())
            else:
                y_pred.append(logits.cpu().numpy())
                y_true.append(labels.cpu().numpy())

                # Compute Spearman and Pearson Correlations for the batch
                batch_labels = labels.cpu().numpy().flatten()
                batch_logits = logits.detach().cpu().numpy().flatten()

                batch_spearman_corr, _ = spearmanr(batch_labels, batch_logits)
                batch_pearson_corr, _ = pearsonr(batch_labels, batch_logits)

                spear_corrs.append(batch_spearman_corr)
                pear_corrs.append(batch_pearson_corr)

    # Return results
    return (y_true, y_pred) if not regression_flag else (y_true, y_pred, spear_corrs, pear_corrs)


def plot_train_val_losses(train_losses, val_losses, save_title):
    fig = plt.figure(figsize=(5, 4))

    plt.plot(train_losses, color="blue", label="Avg Training Loss")
    plt.plot(val_losses, color="red", label="Avg Validation Loss")

    plt.legend()
    plt.title("Average Training/Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.savefig(save_title, dpi=300)
    plt.show()


class CNNBackbone(nn.Module):
    def __init__(self, input_dims, in_channels, filters, feature_size):
        super(CNNBackbone, self).__init__()
        self.input_dims = input_dims
        self.in_channels = in_channels
        self.filters = filters
        self.feature_size = feature_size
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, filters[0], kernel_size=(5,5), stride=1, padding=2),
            nn.BatchNorm2d((self.in_channels**1) * filters[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
                nn.Conv2d(filters[0], filters[1], kernel_size=(5,5), stride=1, padding=2),
                nn.BatchNorm2d((self.in_channels**2) * filters[1]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv3 = nn.Sequential(
                nn.Conv2d(filters[1], filters[2], kernel_size=(3,3), stride=1, padding=1),
                nn.BatchNorm2d((self.in_channels**3) * filters[2]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv4 = nn.Sequential(
                nn.Conv2d(filters[2], filters[3], kernel_size=(3,3), stride=1, padding=1),
                nn.BatchNorm2d((self.in_channels**4) * filters[3]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        
        shape_after_convs = [input_dims[0]//2**(len(filters)), input_dims[1]//2**(len(filters))]
        self.fc1 = nn.Linear(filters[3] * shape_after_convs[0] * shape_after_convs[1], self.feature_size)
        
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, x.shape[1], x.shape[2])
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTBackbone(nn.Module):
    """
    The AST model.
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, model_size='base384', feature_size=1000):

        super(ASTBackbone, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if model_size == 'tiny224':
            self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'small224':
            self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base224':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
        elif model_size == 'base384':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')
        self.feature_size = feature_size
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, self.feature_size))

        # automatcially get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches

        # the linear projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        if imagenet_pretrain == True:
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        # the positional embedding
        if imagenet_pretrain == True:
            # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            # cut (from middle) or interpolate the second dimension of the positional embedding
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
            # cut (from middle) or interpolate the first dimension of the positional embedding
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            # flatten the positional embedding
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
            # concatenate the above positional embedding with the cls token and distillation token of the deit model.
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        else:
            # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    #@autocast('cuda')
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x.float()

    def freeze_layers(self, unfrozen_layers=2):
        """
        Keep the last unfrozen_layers transformer layers trainable and freeze all the rest
        """
        num_layers = len(self.v.blocks)
        for i, blk in enumerate(self.v.blocks):
            if i < num_layers - unfrozen_layers:
                for param in blk.parameters():
                    param.requires_grad = False

def get_classification_report(y_pred, y_true):
    print(classification_report(y_pred=y_pred, y_true=y_true, zero_division=np.nan))
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print(f"Micro-average precision: {micro_precision:.2f}")
    print(f"Micro-average recall: {micro_recall:.2f}")
    print(f"Micro-average F1-score: {micro_f1:.2f}")

def create_folder(folder):
    os.makedirs(folder) if not os.path.exists(os.path.join(os.getcwd(), folder)) else None

def get_regression_report(y_pred, y_true, spear_corrs, pear_corrs):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)

    mean_absolute_percentage_error = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    r2_score = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    explained_variance_score = 1 - (np.var(y_true - y_pred) / np.var(y_true))

    print(f"\tAverage Spearman Correlation: {np.mean(spear_corrs):.4f}")
    print(f"\tAverage Pearson Correlation: {np.mean(pear_corrs):.4f}")
    print(f"\tMSE: {mse:.4f}")
    print(f"\tMAE: {mae:.4f}")
    print(f"\tRMSE: {rmse:.4f}")
    print(f"\tMean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error:.4f}%")
    print(f"\tR^2 Score: {r2_score:.4f}")
    print(f"\tExplained Variance Score: {explained_variance_score:.4f}")

def free_gpu_memory(cleanup=True):
    if cleanup:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    return ((total_memory - torch.cuda.memory_reserved()) / total_memory) * 100

class MultiTaskClassifier(nn.Module):
    def __init__(self, num_tasks, backbone, task_feature_sizes):
        """
        num_tasks (int): The number of tasks
        backbone (nn.Module): The shared backbone
        task_feature_sizes (list of int): Output sizes for each task
        """
        super(MultiTaskClassifier, self).__init__()
        self.backbone = backbone  # Shared backbone
        
        # Separate output layers for each task
        self.output_layers = nn.ModuleList([
            nn.Linear(self.backbone.feature_size, task_feature_sizes[i]) for i in range(num_tasks)
        ])
        
        # Criterion for each task
        self.criterions = [nn.MSELoss() for _ in range(num_tasks)]  # Regression losses

    def forward(self, x, targets):
        """
        x: Input features
        targets: List of target tensors for each task
        lengths: Sequence lengths (for LSTM inputs)
        """
        # shared feature extraction
        feats = self.backbone(x)
        
        # task-specific outputs each element holds the predictions for the corresponding head
        logits = [output_layer(feats) for output_layer in self.output_layers]
        logits = [logit.squeeze(-1) for logit in logits]
        
        # Compute losses for each task
        losses = [criterion(logits[i], targets[:, i]) for i, criterion in enumerate(self.criterions)]
        
        # weighted sum of losses (equal weight for simplicity; can be tuned)
        total_loss = sum(losses)
        
        return total_loss, losses, logits
    
def multi_task_learning_one_epoch(device, train_dl, model, optimizer, scheduler):
    model.to(device)
    model.train()
    curr_total_loss, curr_valence_loss = 0., 0.
    curr_energy_loss, curr_dance_loss = 0., 0.
    for inputs, targets, _ in tqdm(train_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss, losses, logits = model(inputs.float(), targets.float())
        loss.backward()
        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        curr_total_loss += loss.item()
        curr_valence_loss += losses[0].item()
        curr_energy_loss += losses[1].item()
        curr_dance_loss += losses[2].item()

    avg_total_loss = (curr_total_loss / len(train_dl))
    avg_valence_loss = (curr_valence_loss / len(train_dl))
    avg_energy_loss = (curr_energy_loss / len(train_dl))
    avg_dance_loss = (curr_dance_loss / len(train_dl))
    
    return avg_total_loss, avg_valence_loss, avg_energy_loss, avg_dance_loss    

def eval_multi_task(model, val_dl, device):
    model.to(device)
    model.eval()
    total_loss = 0.
    valence_loss = 0.
    energy_loss = 0.
    dance_loss = 0.
    with torch.inference_mode():
        for inputs, targets, _ in (val_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            loss, losses, logits = model(inputs.float(), targets.float())
            total_loss += loss.item()
            valence_loss += losses[0].item()
            energy_loss += losses[1].item()
            dance_loss += losses[2].item()

    avg_total_loss = (total_loss / len(val_dl))
    avg_valence_loss = (valence_loss / len(val_dl))
    avg_energy_loss = (energy_loss / len(val_dl))
    avg_dance_loss = (dance_loss / len(val_dl))
    
    return avg_total_loss, avg_valence_loss, avg_energy_loss, avg_dance_loss

def train_multi_task_learning(epochs, device, train_dl, val_dl, model, optimizer, scheduler, save_path):
    train_total_losses, train_valence_losses = [], []
    train_energy_losses, train_dance_losses = [], []
    val_total_losses, val_valence_losses = [], []
    val_energy_losses, val_dance_losses = [], []
    early_stopper = EarlyStopper(model, save_path, patience=5)
    print(f'Training started for model {save_path.replace(".pth", "")}...')
    for epoch in range(epochs):
        avg_train_total_loss, avg_train_valence_loss, avg_train_energy_loss, avg_train_dance_loss = multi_task_learning_one_epoch(device, train_dl, model, optimizer, scheduler)
        avg_val_total_loss, avg_val_valence_loss, avg_val_energy_loss, avg_val_dance_loss = eval_multi_task(model, val_dl, device)
        
        print(f'Epoch {epoch + 1}')
        print(f"Training Metrics")
        print(f"\t\tAverage Total Loss: {avg_train_total_loss: .2f}")
        print(f"\t\tAverage Valence Loss: {avg_train_valence_loss: .2f}")
        print(f"\t\tAverage Energy Loss: {avg_train_energy_loss: .2f}")
        print(f"\t\tAverage Danceability Loss: {avg_train_dance_loss: .2f}")
        print(f"Evaluation Metrics")
        print(f"\t\tAverage Total Loss: {avg_val_total_loss: .2f}")
        print(f"\t\tAverage Valence Loss: {avg_val_valence_loss: .2f}")
        print(f"\t\tAverage Energy Loss: {avg_val_energy_loss: .2f}")
        print(f"\t\tAverage Danceability Loss: {avg_val_dance_loss: .2f}")

        train_total_losses.append(avg_train_total_loss)
        train_valence_losses.append(avg_train_valence_loss)
        train_energy_losses.append(avg_train_energy_loss)
        train_dance_losses.append(avg_train_dance_loss)

        val_total_losses.append(avg_val_total_loss)
        val_valence_losses.append(avg_val_valence_loss)
        val_energy_losses.append(avg_val_energy_loss)
        val_dance_losses.append(avg_val_dance_loss)

        train_losses = {"Total Loss": train_total_losses, "Valence Loss": train_valence_losses, "Energy Loss": train_energy_losses, "Danceability Loss": train_dance_losses}
        val_losses = {"Total Loss": val_total_losses, "Valence Loss": val_valence_losses, "Energy Loss": val_energy_losses, "Danceability Loss": val_dance_losses}

        if early_stopper.early_stop(avg_val_total_loss):
                print('Early Stopping was activated.')
                print('Training has been completed.\n')
                break
        
    return train_losses, val_losses
import torch
import scipy
import numpy as np

from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from torchvision import transforms


class HyperspectralDataset(Dataset):
    def __init__(self, data, labels, config):
        self.data, self.labels = createImagepatches(data, labels, config)
        self.indices = list(range(len(self.data)))

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        # Extract the patch from the padded data
        patch = self.data[i]

        # Adjust indices for labels to ensure they're within bounds
        label = self.labels[i]

        return patch, label


def indices_process(indices, labels, config):
    """
    Splits the indices into training and validation sets.
    """
    if config.SOLVER["oversample"] and config.SOLVER["mode"] == "train":
        # remove 0 in labels
        labels = labels[labels != 0]

        indices = oversampleWeakClasses(indices, labels)

    return indices


def oversampleWeakClasses(indices, labels):
    uniqueLabels, labelCounts = np.unique(labels, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts

    # Initialize newIndices
    newIndices = []

    # Oversample each label
    for label, labelInverseRatio in zip(uniqueLabels, labelInverseRatios):
        current_indices = [indices[i] for i in range(len(labels)) if labels[i] == label]
        # Repeat the current indices based on the label inverse ratio
        newIndices.extend(current_indices * round(labelInverseRatio))

    # Shuffle the new indices
    np.random.seed(42)
    np.random.shuffle(newIndices)

    return newIndices


def load_data(SOLVER):
    print("Loading data...")
    if SOLVER["mode"] == "train":
        data_hsi = scipy.io.loadmat("./datasets/YanCity/data_hsi.mat")["data"]
        data_msi = scipy.io.loadmat("./datasets/YanCity/data_msi.mat")["data"]
        train_labels = scipy.io.loadmat("./datasets/YanCity/train_label.mat")[
            "train_label"
        ]
        if SOLVER["HM_mixup"]:
            dataX = HMmixup(data_hsi, data_msi, SOLVER["pca_components"])
            labelsY = train_labels
        else:
            dataX = applyPCA(data_hsi, SOLVER["pca_components"])
            labelsY = train_labels

    elif SOLVER["mode"] == "test":
        data_hsi = scipy.io.loadmat("./datasets/YanCity/data_hsi.mat")["data"]
        data_msi = scipy.io.loadmat("./datasets/YanCity/data_msi.mat")["data"]
        test_labels = scipy.io.loadmat("./datasets/YanCity/test_label.mat")[
            "test_label"
        ]
        if SOLVER["HM_mixup"]:
            dataX = HMmixup(data_hsi, data_msi, SOLVER["pca_components"])
            labelsY = test_labels
        else:
            dataX = applyPCA(data_hsi, SOLVER["pca_components"])
            labelsY = test_labels

    elif SOLVER["mode"] == "predict":
        data_hsi = scipy.io.loadmat("./datasets/YanCity/data_hsi.mat")["data"]
        data_msi = scipy.io.loadmat("./datasets/YanCity/data_msi.mat")["data"]
        train_labels = scipy.io.loadmat("./datasets/YanCity/train_label.mat")[
            "train_label"
        ]
        if SOLVER["HM_mixup"]:
            dataX = HMmixup(data_hsi, data_msi, SOLVER["pca_components"])
            labelsY = train_labels
        else:
            dataX = applyPCA(data_hsi, SOLVER["pca_components"])
            labelsY = train_labels

    else:
        raise ValueError("SOLVER-mode should be train or test or predict!")

    return dataX, labelsY


def applyPCA(data, pca_components):
    """
    apply PCA to data
    :param X: input data, shape (H, W, C)
    :param pca_components: number of components
    """
    # newX = np.reshape(X, (-1, X.shape[2]))
    # pca = PCA(n_components=pca_components, whiten=True)
    # newX = pca.fit_transform(newX)
    # newX = np.reshape(newX, (X.shape[0], X.shape[1], pca_components))
    num_samples, num_features, num_bands = data.shape
    flattened_data = data.reshape(num_samples * num_features, num_bands)

    # Apply PCA to the flattened data
    pca = PCA(n_components=pca_components)
    pca.fit(flattened_data)

    # Transform the data back to the original shape
    transformed_data = pca.transform(flattened_data)
    data_pca = transformed_data.reshape(num_samples, num_features, pca_components)
    return data_pca


def HMmixup(data_hsi, data_msi, pca_components):
    data_hsi = applyPCA(data_hsi, pca_components - data_msi.shape[2])
    HM_mix = np.concatenate((data_hsi, data_msi), axis=2)
    return HM_mix


def createImagepatches(data, labels, config):
    print("Creating image patches...")
    # normalize data
    if config.SOLVER["normalize"]:
        data = normalize_data(data)
    else:
        data = data
    labels = labels
    patch_size = config.MODEL["patch_size"]
    removeZeroLabels = config.MODEL["removeZeroLabels"]
    margin = patch_size // 2

    # Pad data with zeros
    padded_data = np.pad(
        data,
        ((margin, margin), (margin, margin), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    padded_data = padded_data.transpose(2, 0, 1)

    # Compute all possible indices for patches
    indices = []
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            label = labels[i, j]
            if removeZeroLabels and label == 0:
                continue
            indices.append((i, j))

    indices = indices_process(indices, labels, config)

    patches = []
    for i, j in indices:
        patch = padded_data[:, i : i + patch_size, j : j + patch_size]
        patches.append(patch)

    patches = np.array(patches)
    patches = torch.tensor(patches, dtype=torch.float32)

    # Augment data
    if config.SOLVER["augmentData"]:
        augment = transforms.RandomChoice(
            [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0),
                transforms.RandomRotation(30),
            ]
        )
        patches = [augment(patch) for patch in patches]

    newLabels = torch.tensor([labels[i, j] for i, j in indices], dtype=torch.long)
    return patches, newLabels


def normalize_data(raw_data):
    max_value = np.max(raw_data)
    min_value = np.min(raw_data)
    normalized_data = (raw_data - min_value) / (max_value - min_value)
    return normalized_data

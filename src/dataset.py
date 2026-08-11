"""
dataset.py
----------
Handles:
1. Loading a medical image dataset from a folder structure (ImageFolder style).
2. Splitting the dataset among N simulated FL clients (hospitals).
3. Supports both IID and Non-IID splits (Non-IID = more realistic, each
   "hospital" sees a skewed class distribution, like in the real world).

Expected folder structure (typical for Chest X-Ray / Skin Lesion datasets):

    data/
      train/
        class_0/  (e.g. NORMAL)
          img1.png
          img2.png
        class_1/  (e.g. PNEUMONIA)
          img1.png
      test/
        class_0/
        class_1/
"""

import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms


IMG_SIZE = 224  # standard input size for ResNet / EfficientNet backbones

# Standard ImageNet normalization since we use pretrained backbones
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True):
    """Return torchvision transform pipeline for train or eval."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])


def load_full_dataset(data_dir: str, train: bool = True):
    """Load dataset using ImageFolder. `data_dir` should point to train/ or test/."""
    return datasets.ImageFolder(root=data_dir, transform=get_transforms(train))


def split_iid(dataset: Dataset, num_clients: int, seed: int = 42):
    """Randomly split dataset into num_clients equal-ish shards (IID)."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    shards = np.array_split(indices, num_clients)
    return [Subset(dataset, shard.tolist()) for shard in shards]


def split_non_iid(dataset: Dataset, num_clients: int, classes_per_client: int = 1, seed: int = 42):
    """
    Split dataset so each client mostly sees a subset of classes
    (simulates real hospitals seeing different disease prevalence).

    classes_per_client: how many dominant classes each client is biased towards.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    num_classes = len(dataset.classes)

    # group indices by class
    class_indices = {c: np.where(targets == c)[0].tolist() for c in range(num_classes)}
    for c in class_indices:
        rng.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]

    # assign each client a set of "dominant" classes round-robin style
    for client_id in range(num_clients):
        dominant_classes = [(client_id * classes_per_client + i) % num_classes
                             for i in range(classes_per_client)]
        for c in dominant_classes:
            chunk_size = len(class_indices[c]) // max(1, (num_clients // num_classes + 1))
            chunk_size = max(chunk_size, 1)
            take = class_indices[c][:chunk_size]
            class_indices[c] = class_indices[c][chunk_size:]
            client_indices[client_id].extend(take)

    # distribute any leftover samples round-robin so no data is wasted
    leftovers = [idx for c in class_indices for idx in class_indices[c]]
    rng.shuffle(leftovers)
    for i, idx in enumerate(leftovers):
        client_indices[i % num_clients].append(idx)

    return [Subset(dataset, idxs) for idxs in client_indices]


def get_client_dataloaders(data_dir: str, num_clients: int, batch_size: int = 16,
                            non_iid: bool = True, classes_per_client: int = 1):
    """
    Main entry point: loads dataset and returns a list of DataLoaders,
    one per simulated FL client.
    """
    full_train = load_full_dataset(data_dir, train=True)

    if non_iid:
        shards = split_non_iid(full_train, num_clients, classes_per_client)
    else:
        shards = split_iid(full_train, num_clients)

    loaders = [DataLoader(shard, batch_size=batch_size, shuffle=True, num_workers=2)
               for shard in shards]
    return loaders, full_train.classes


def get_test_dataloader(data_dir: str, batch_size: int = 32):
    """Central held-out test set to evaluate the global federated model."""
    test_set = load_full_dataset(data_dir, train=False)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2), test_set.classes

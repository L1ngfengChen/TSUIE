from os.path import join
from torchvision.transforms import Compose, ToTensor, Resize,Normalize
from data.dataset import DatasetFromFolder, DatasetFromFolderTest, Pr_DatasetFromFolder


def transform():
    return Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_training_set(label, data, enhance, patch_size, data_augmentation):
    return DatasetFromFolder(label, data, enhance, patch_size, data_augmentation, transform=transform())

def get_eval_set(data_dir):
    return DatasetFromFolderTest(data_dir, transform=transform())

def get_pr_training_set(data_dir, label_dir):
    return Pr_DatasetFromFolder(data_dir, label_dir, transform=transform())
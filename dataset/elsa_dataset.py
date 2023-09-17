import numpy as np

from glob import glob
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import random_split, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ElsaDataset():
    REAL, FAKE = 0, 1
    def __init__(self, path):
        self.elsa_dataset = TrainDataset(path)

    def __len__(self):
        return len(self.elsa_dataset)

    def get_datasets(self, train_ratio = 0.8):
        train_size = int(self.__len__() * train_ratio)
        valid_size = self.__len__() - train_size
        train_dataset, valid_dataset = random_split(self.elsa_dataset, [train_size, valid_size])
        return train_dataset, valid_dataset


class TrainDataset(Dataset):
    def __init__(self, path):
        super(TrainDataset, self).__init__()
        fake = glob(path + 'fake-images/part-00*/**')
        real = glob(path + 'real-images/part-00*/**')
        self.path = np.array(fake + real)
        self.label = np.array([1.] * len(fake) + [0.] * len(real))
        self.totensor = transforms.ToTensor()
        self.resize = transforms.Resize(224, antialias=False)
        self.centercrop = transforms.CenterCrop((224, 224))
        self.norm = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        image = Image.open(self.path[idx]).convert("RGB")
        label = self.label[idx]
        image = self.totensor(image)
        if image.shape[1] < 224 or image.shape[2] < 224:
            image = self.resize(image)
        image = self.centercrop(image)
        image = self.norm(image)
        return image, label


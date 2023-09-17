import torchvision.transforms as transforms

from glob import glob
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TestDataset(Dataset):
    def __init__(self, path, train = False):
        super(TestDataset, self).__init__()
        self.img_list = glob(path + 'test_set/**')
        self.class_list = [_.split('/')[-1] for _ in self.img_list]
        self.totensor = transforms.ToTensor()
        self.resize = transforms.Resize(224, antialias=False)
        self.norm = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
        self.centercrop = transforms.CenterCrop((224, 224))


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx]).convert('RGB')
        file_name = self.class_list[idx].split('.')[0] + '.jpg'
        image = self.totensor(image)
        if image.shape[1] < 224 or image.shape[2] < 224:
            image = self.resize(image)
        image = self.centercrop(image)
        image = self.norm(image)
        return image, file_name
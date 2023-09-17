import requests

from os import exists
from tqdm import tqdm
from datasets import load_dataset


class ElsaDataDownloader():
    def __init__(self, start, end):
        self.start = start
        self.length = end - start
        self.elsa_dataset = load_dataset("elsaEU/ELSA1M_track1", split="train",
                                         streaming=True).skip(start).take(self.length)
        self.__iter = iter(self.elsa_dataset)
    
    def __len__(self):
        return self.length

    def fake_image_download(self, source, save_path):
        try:
            if exists(save_path):
                return
            source.save(save_path)
        except :
            print(f'Error saving fake image: {save_path}')

    def real_image_download(self, source, save_path):
        try:
            response = requests.get(source, timeout = 5)
            if exists(save_path):
                return
            with open(save_path, 'wb') as fo:
                fo.write(response.content)
        except Exception as e:
            print(f'Error saving real image: {save_path} {e}')

    def start_download(self):
        for _ in tqdm(range(self.length)):
            item = next(self.__iter)
            fake_image = item.pop('image')
            url_real_image = item.pop('url_real_image')
            file_path = item.pop('filepath')
            self.fake_image_download(fake_image, f"origin-data/{file_path}")
            self.real_image_download(url_real_image, f"origin-data/real-images/{file_path[12:]}")

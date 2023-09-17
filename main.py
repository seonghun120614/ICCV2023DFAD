EPOCH = 40
SEED = 7777
NUM_WORKERS = 12
BATCH_SIZE = 128
MODEL = 'Grag2021'
LEARNING_RATE = 1e-4
DATA_PATH = 'origin-data/'

import torch
import torch.optim as optim

from tqdm import tqdm
from trainer import Trainer
from torch.utils.data import DataLoader
from networks.DMDetection import resnet50
from dataset.elsa_dataset import ElsaDataset
from dataset.test_dataset import TestDataset

from utils.construct_folder import create_folder
from utils.elsa_downloader import ElsaDataDownloader


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    create_folder()
    ElsaDataDownloader(0,1015000)

    # Call Grag2021 model without pre-training
    model = resnet50(num_classes=1,
                     gap_size=28,
                     stride0=1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    criterion = torch.nn.BCEWithLogitsLoss()
    trainer = Trainer(model=model,
                    optimizer=optimizer,
                    scaler=scaler)


    # Training
    elsa_loader = ElsaDataset(DATA_PATH) # Preparing the Elsa-Dataset
    train_dataset, valid_dataset = elsa_loader.get_datasets() # split the train, valid dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=NUM_WORKERS,
                            drop_last=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=NUM_WORKERS,
                            drop_last=True, pin_memory=True)
    
    for _ in tqdm(range(EPOCH), leave=True, desc='    Epochs'):
        with tqdm(total=len(train_loader), desc='  Training') as bar:
            for image, label in train_loader:
                loss, acc = trainer.learn(image, label, criterion=criterion)
                bar.update(1)

        with tqdm(total=len(valid_loader), desc='Validation') as bar:
            for image, label in valid_loader:
                loss = trainer.evaluate(image, label, criterion=criterion)
                bar.update(1)
            f1 = trainer.getF1()
        trainer.deploy(f"weights/{MODEL}_epoch_{_}.pth")
    trainer.deploy(f"weights/{MODEL}_final.pth")


    # Testing
    trainer.load_state_dict(f"weights/{MODEL}_final.pth")
    test_dataset = TestDataset(DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            drop_last=False, pin_memory=True)
    trainer.test(test_loader, save_name='results/Grag2021.json')

import json
import torch
import torch.nn as nn

from tqdm import tqdm
from collections import OrderedDict


class Trainer():
    def __init__(self, model, optimizer, scaler, device='cuda'):
        self.DEVICE = device
        self.model = nn.DataParallel(model).to(device)
        self.model.train()
        self.optimizer = optimizer
        self.scaler = scaler
        self.valid_pred = None
        self.valid_true = None


    def learn(self, x, y, criterion):
        criterion = criterion.to(self.DEVICE)
        with torch.autocast(device_type = self.DEVICE, dtype = torch.float16):
            image = x.to(self.DEVICE)
            label = y.to(self.DEVICE)
            output = self.model(image).flatten()
            loss = criterion(output, label)
            acc = (((output>=0.5)==label).sum()/len(label)).item()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss.item(), acc


    def evaluate(self, x, y, criterion):
        with torch.no_grad():
            with torch.autocast(device_type = self.DEVICE, dtype = torch.float16):
                image = x.to(self.DEVICE)
                label = y.to(self.DEVICE)
                output = self.model(image).flatten()
                loss = criterion(output, label)
                output = output>=0.5
            
            if self.valid_pred == None: self.valid_pred = output.cpu()
            else: self.valid_pred = torch.cat([self.valid_pred, output.cpu()])

            if self.valid_true == None: self.valid_true = y
            else: self.valid_true = torch.cat([self.valid_true, y])

        return loss.item()


    def test(self, test_loader, save_name):
        file_data = OrderedDict()
        with torch.no_grad():
            self.model.eval()
            for image, file_name in tqdm(test_loader, total=len(test_loader)):
                image = image.to(self.DEVICE)
                output = self.model(image)
                output = torch.sigmoid(output.flatten())
                for i in range(len(file_name)):
                    file_data[file_name[i]] = 1 if output[i].item() > 0.5 else 0
        
        with open(save_name, "w", encoding="utf-8") as f:
            json.dump(file_data, f, ensure_ascii=False, indent="\t")
        print('=> End of Test')


    def getF1(self):
        if self.valid_pred == None:
            raise Exception("There is no valid data")
        tp = (self.valid_true == (self.valid_pred>=0.5)).sum()
        fp = ((~self.valid_true) == (self.valid_pred>=0.5)).sum()
        fn = (self.valid_true == (self.valid_pred<0.5)).sum()
        
        epsilon = 1e-7
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * (precision*recall) / (precision + recall + epsilon)
        f1.requires_grad = False
        self.valid_pred, self.valid_true = None, None
        return f1


    def deploy(self, save_name):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(checkpoint, save_name)
        print("=> Saving Checkpoint")


    def load_state_dict(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("=> Appliance Complete")
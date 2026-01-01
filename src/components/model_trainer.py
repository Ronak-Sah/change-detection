import os
import torch
from src.logger import logger
from src.entity import ModelTrainerConfig
from src.components.models.FcSiam import FCSiamDiff
from torch.utils.data import Dataset,DataLoader
import numpy as np
from PIL import Image
import random

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        probs = probs[:,1,:,:]  # change class
        targets = targets.float()

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class ImageDataset(Dataset):
    def __init__(self,path,patch_size=256,augmentation=True):
        self.sub_a_path=os.path.join(path,"A")
        self.sub_b_path=os.path.join(path,"B")
        self.sub_label_path=os.path.join(path,"label")
        
        self.a_path=os.listdir(self.sub_a_path)
        self.b_path=os.listdir(self.sub_b_path)
        self.label_path=os.listdir(self.sub_label_path)
        self.patch_size=patch_size
        self.augmentation=augmentation

    def __len__(self):
        return len(self.a_path)

    def __getitem__(self, idx):
        after_im_path=os.path.join(self.sub_a_path,self.a_path[idx])
        before_im_path=os.path.join(self.sub_b_path,self.a_path[idx])
        label_im_path=os.path.join(self.sub_label_path,self.a_path[idx])
        with Image.open(before_im_path) as img:
            img = img.convert('L')
            before_image=np.array(img)
        with Image.open(after_im_path) as img:
            img = img.convert('L')
            after_image=np.array(img)
        with Image.open(label_im_path) as img:
            img = img.convert('L')
            label_image=np.array(img)

        H, W = before_image.shape[:2]

        if self.augmentation:
            new_H=random.randint(0, H - self.patch_size)
            new_W=random.randint(0, W - self.patch_size)

        else :
            new_H=0
            new_W=0

        before_image=before_image[new_H:new_H+self.patch_size,new_W:new_W+self.patch_size]
        after_image=after_image[new_H:new_H+self.patch_size,new_W:new_W+self.patch_size]
        label_image=label_image[new_H:new_H+self.patch_size,new_W:new_W+self.patch_size]

        before_image = torch.from_numpy(before_image).float().unsqueeze(0) / 255.0
        after_image = torch.from_numpy(after_image).float().unsqueeze(0) / 255.0
        
        label_image = (label_image > 0).astype(np.uint8)
        label_image = torch.from_numpy(label_image).long()

        return before_image,after_image,label_image



class Model_Trainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config= config
        self.device= "cuda" if torch.cuda.is_available() else "cpu"

        self.model = FCSiamDiff().to(self.device)
              

    def train(self):
        train_data_path=self.config.train_data_path

        model_path = os.path.join(self.config.root_dir, "model.pth")
        checkpoint_path = os.path.join(self.config.root_dir, "checkpoint.pth")
        best_model_path = os.path.join(self.config.root_dir, "best_model.pth")

        dataset = ImageDataset(train_data_path)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,  
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,   
            eta_min=1e-6
        )
        
        best_loss=100.0
        start_epoch = 0
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss=checkpoint["best_loss"]

            print(f"Resuming from epoch {start_epoch}")

        weights = torch.tensor([0.2, 0.8]).to(self.device)
        ce = torch.nn.CrossEntropyLoss(weight=weights)
        dice = DiceLoss()
        for epoch in range(start_epoch, start_epoch + self.config.epochs):
            print("Running epoch no... ",epoch+1)
            self.model.train()
            total_loss = 0.0
            batch_no=0
            for before_image,after_image,label_image in dataloader:
                total_batches = len(dataloader)
                batch_no=batch_no+1
                rem=int(total_batches) - batch_no
                print(f"Batch no : {batch_no}, Total batch : {total_batches}, Remaining batch :{rem}" )
                
                before_image = before_image.to(self.device)
                after_image = after_image.to(self.device)
                label_image = label_image.to(self.device)


                optimizer.zero_grad()
                if torch.isnan(before_image).any() or torch.isnan(after_image).any():
                    print("Skipping batch: NaN detected in input frames")
                    continue
                y_pred=self.model(before_image,after_image)

                loss = ce(y_pred, label_image) + dice(y_pred, label_image)

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                if torch.isnan(grad_norm) or grad_norm > 1000.0:
                    print(f"Skipping batch {batch_no}: Extreme grad norm {grad_norm:.2f}")
                    continue
                if grad_norm > 10.0:  
                    print(f"Warning: Large grad norm {grad_norm:.2f} at batch {batch_no}")

                optimizer.step()

                total_loss += loss.item()

            torch.save({
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss" : best_loss
            }, checkpoint_path)
            print(f"Epoch {epoch+1}/{self.config.epochs+start_epoch}, Loss: {total_loss: .4f}")
            avg_loss = total_loss / len(dataloader)
            scheduler.step()
            print(f"Epoch {epoch+1}/{self.config.epochs+start_epoch}, Avg Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs(self.config.root_dir, exist_ok=True)
                torch.save(self.model.state_dict(),best_model_path)
                logger.info(f"Model saved at: {best_model_path}")
                print(f"*** New best model saved! Avg Loss: {avg_loss:.4f} ***")

        


    
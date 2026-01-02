import os
from src.logger import logger
from src.entity import ModelEvaluationConfig
import torch
from torch.utils.data import  Dataset,DataLoader
from PIL import Image
import numpy as np
import random
from src.components.models.FcSiam import FCSiamDiff

class ImageDataset(Dataset):
    def __init__(self,path,patch_size=256,augmentation=True,limit=60):
        self.sub_a_path=os.path.join(path,"A")
        self.sub_b_path=os.path.join(path,"B")
        self.sub_label_path=os.path.join(path,"label")
        
        self.a_path=os.listdir(self.sub_a_path)[:limit]
        self.b_path=os.listdir(self.sub_b_path)[:limit]
        self.label_path=os.listdir(self.sub_label_path)[:limit]
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

def compute_metrics(pred, target, eps=1e-6):
    """
    pred: (B, H, W) binary {0,1}
    target: (B, H, W) binary {0,1}
    """
    TP = ((pred == 1) & (target == 1)).sum().float()
    TN = ((pred == 0) & (target == 0)).sum().float()
    FP = ((pred == 1) & (target == 0)).sum().float()
    FN = ((pred == 0) & (target == 1)).sum().float()

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = TP / (TP + FP + FN + eps)

    return {
        "IoU": iou.item(),
        "F1": f1.item(),
        "Precision": precision.item(),
        "Recall": recall.item()
    }

def compute_confusion(pred, target):
    TP = ((pred == 1) & (target == 1)).sum().item()
    FP = ((pred == 1) & (target == 0)).sum().item()
    FN = ((pred == 0) & (target == 1)).sum().item()
    TN = ((pred == 0) & (target == 0)).sum().item()
    return TP, FP, FN, TN
class Model_Evaluation:
    def __init__(self,config: ModelEvaluationConfig):
        self.config= config
        self.device= "cuda" if torch.cuda.is_available() else "cpu"

        self.model = FCSiamDiff().to(self.device)

        model_path = config.model_path
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))

        logger.info("Model loaded succesfully for evaluation")

    
    def evaluate(self):
        
        self.model.eval()

        
        test_dataset = ImageDataset(self.config.test_data_path,limit=100)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,  
            num_workers=0,
            pin_memory=True,
        )
        scripted_model = torch.jit.script(self.model)
        scripted_model = torch.jit.optimize_for_inference(scripted_model)
        scripted_model.save(os.path.join(self.config.root_dir,"change_detection.pt"))



        with torch.no_grad():

            TP = FP = FN = TN = 0

            for before_image,after_image,label_image in test_dataloader:
                    
                before_image = before_image.to(self.device)
                after_image = after_image.to(self.device)
                label_image = label_image.to(self.device)

                logits = self.model(before_image, after_image)
                pred = torch.argmax(logits, dim=1)

                metrics = compute_metrics(pred, label_image)
                tp, fp, fn, tn = compute_confusion(pred, label_image)

                TP += tp
                FP += fp
                FN += fn
                TN += tn

        eps = 1e-6

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = TP / (TP + FP + FN + eps)

        metrics = {
            "IoU": iou,
            "F1": f1,
            "Precision": precision,
            "Recall": recall
        }

        print(metrics)


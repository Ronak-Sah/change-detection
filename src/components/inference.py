import torch
from PIL import Image
import numpy as np
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image

# Split image into patches
def image_to_patches(img, patch_size=256):
    H, W = img.shape
    patches = []
    positions = []

    # Slide window over image
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]

            # If patch smaller than patch_size, pad it
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                pad_H = patch_size - patch.shape[0]
                pad_W = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0,pad_H),(0,pad_W)), mode='constant', constant_values=0)

            patches.append(patch)
            positions.append((i,j))
    return patches, positions

# Merge patches into full image
def patches_to_image(patches, positions, full_size, patch_size=256):
    H, W = full_size
    full_mask = np.zeros((H, W), dtype=np.uint8)
    for patch, (i,j) in zip(patches, positions):
        h_end = min(i+patch_size, H)
        w_end = min(j+patch_size, W)
        patch_crop = patch[:h_end-i, :w_end-j]
        full_mask[i:h_end, j:w_end] = patch_crop
    return full_mask



class Prediction():
    def __init__(self,before_image_path,after_image_path):
        with Image.open(before_image_path) as img:
            img = img.convert('L')
            self.before_image=np.array(img)
        with Image.open(after_image_path) as img:
            img = img.convert('L')
            self.after_image=np.array(img)

        model_path=r"artifacts\model_evaluation\change_detection.pt"
        
        self.device= "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(model_path, map_location=self.device)

    def predict(self):

        before_patches, positions = image_to_patches(self.before_image)
        after_patches, _ = image_to_patches(self.after_image)


        self.model.eval()
        preds = []
        with torch.no_grad():
            for b_patch, a_patch in zip(before_patches, after_patches):
                b_tensor = torch.from_numpy(b_patch).float().unsqueeze(0).unsqueeze(0) / 255.0
                a_tensor = torch.from_numpy(a_patch).float().unsqueeze(0).unsqueeze(0) / 255.0
                b_tensor, a_tensor = b_tensor.to(self.device), a_tensor.to(self.device)

                logits = self.model(b_tensor, a_tensor)
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                preds.append(pred.astype(np.uint8))

        full_mask = patches_to_image(preds, positions, self.before_image.shape)
        return full_mask
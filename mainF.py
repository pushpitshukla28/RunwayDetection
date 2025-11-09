# main.py - FINAL TRAINING SCRIPT (improved)
import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp

# --- Config ---
TRAIN_IMG_DIR = r"C:\Users\nandi\Downloads\640x360\640x360\train"
TRAIN_LABELS_JSON_PATH = r"C:\Users\nandi\Downloads\labels\labels\lines\train_labels_640x360.json"
MODEL_SAVE_PATH = "runway_detector_model_final.pth"

IMAGE_H, IMAGE_W = 256, 512
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------

# --- Dataset ---
def create_mask_from_coords(coords, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    try:
        ledg_p1, ledg_p2 = coords[0:2], coords[2:4]
        redg_p1, redg_p2 = coords[4:6], coords[6:8]
        polygon_pts = np.array([ledg_p1, ledg_p2, redg_p2, redg_p1], dtype=np.int32)
        cv2.fillPoly(mask, [polygon_pts], 1)
    except Exception:
        pass
    return mask

class RunwayDataset(Dataset):
    def __init__(self, image_dir, labels_json, transform=None):
        self.image_dir = image_dir
        with open(labels_json, "r") as f:
            self.labels = json.load(f)
        self.image_ids = sorted(list(self.labels.keys()))
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, os.path.splitext(img_name)[0] + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        coords = np.zeros(12, dtype=np.float32)
        annotations = self.labels.get(img_name, [])
        coord_map = {"LEDG": 0, "REDG": 4, "CTL": 8}
        for ann in annotations:
            label = ann.get("label")
            points = np.array(ann.get("points", [])).flatten()
            if label in coord_map and len(points) == 4:
                coords[coord_map[label]:coord_map[label]+4] = points
        mask = create_mask_from_coords(coords, image.shape[0], image.shape[1])

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        mask = mask.unsqueeze(0).float()
        return image, mask

# --- Model ---
class RunwayDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

    def forward(self, x):
        return self.model(x)

# --- Loss ---
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

# --- Train ---
if __name__ == "__main__":
    transform = A.Compose([
        A.Resize(IMAGE_H, IMAGE_W),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    dataset = RunwayDataset(TRAIN_IMG_DIR, TRAIN_LABELS_JSON_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = RunwayDetector().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, masks in progress:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)

            loss_bce = bce(outputs, masks)
            loss_dice = dice_loss(outputs, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice  # balanced loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix({"Loss": loss.item()})

        print(f"Epoch {epoch+1} finished. Avg Loss = {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

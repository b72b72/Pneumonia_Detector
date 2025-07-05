import pandas as pd
import os

# Define base directory
base_dir = r"C:\Users\joshz\Downloads\x-ray_model"
train_img_dir = os.path.join(base_dir, "training_images")
test_img_dir = os.path.join(base_dir, "testing_images")

# Load the CSV
df = pd.read_csv(os.path.join(base_dir, "Data_Entry_2017.csv"))

# Filter only to Pneumonia and No Finding
df = df[df["Finding Labels"].str.contains("Pneumonia") | (df["Finding Labels"] == "No Finding")].copy()
df["Label"] = df["Finding Labels"].apply(lambda x: 1 if "Pneumonia" in x else 0)

# Match only images available in training/testing folders
train_img_names = os.listdir(train_img_dir)
test_img_names = os.listdir(test_img_dir)

train_df = df[df["Image Index"].isin(train_img_names)].copy()
# Filter test_df to only those images in testing folder AND either Pneumonia or No Finding
test_df = df[df["Image Index"].isin(test_img_names)].copy()

# Optional: reduce test set to 200 samples (balanced or not)
n_test_samples = 130

# Balanced subset (optional)
n_per_class_test = n_test_samples // 2
pneumonia_test_subset = test_df[test_df["Label"] == 1].sample(n=n_per_class_test, random_state=42)
no_finding_test_subset = test_df[test_df["Label"] == 0].sample(n=n_per_class_test, random_state=42)

test_df = pd.concat([pneumonia_test_subset, no_finding_test_subset]).sample(frac=1).reset_index(drop=True)



# Set correct image paths
train_df["ImagePath"] = train_df["Image Index"].apply(lambda x: os.path.join(train_img_dir, x))
test_df["ImagePath"] = test_df["Image Index"].apply(lambda x: os.path.join(test_img_dir, x))

# Reset index
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Optional: reduce training data size for faster training (balanced)
n_samples_per_class = 190 # You can adjust this (e.g., 100 total images)

pneumonia_subset = train_df[train_df["Label"] == 1].sample(n=n_samples_per_class, random_state=42)
no_finding_subset = train_df[train_df["Label"] == 0].sample(n=n_samples_per_class, random_state=42)

train_df = pd.concat([pneumonia_subset, no_finding_subset]).sample(frac=1).reset_index(drop=True)

print(f"Training Pneumonia Samples: {train_df[train_df['Label'] == 1].shape[0]}")

print(f"Testing Pneumonia Samples: {test_df[test_df['Label'] == 1].shape[0]}")


from torch.utils.data import Dataset
from PIL import Image

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "ImagePath"]
        label = self.df.loc[idx, "Label"]
        
        image = Image.open(img_path).convert("L")  # grayscale

        if self.transform:
            image = self.transform(image)

        return image, label

from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [C x H x W], pixel range [0, 1]
])

train_dataset = ChestXrayDataset(train_df, transform=transform)
test_dataset = ChestXrayDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch
import torch.nn as nn
import torchvision.models as models

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 1 input channel (grayscale)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):  # increase for better results
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)

        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Save trained model
torch.save(model.state_dict(), "model.pth")

# Define transform for inference (should match training transform)
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image_from_path(image_path):
    image = Image.open(image_path).convert("L")
    image = inference_transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        prediction = (output.item() > 0.5)
        return "Pneumonia" if prediction else "No Finding"


# Evaluation loop 
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = (outputs > 0.5).int()
        correct += (preds.squeeze() == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

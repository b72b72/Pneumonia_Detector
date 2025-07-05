# PneumoNET
PneumoNet is an end-to-end pneumonia detection tool utilising a CNN machine learning model built form NIH Chest X-ray image data.

## Installation
Ensure the below Python packages are installed using pip:

```bash
pip install torch torchvision flask pillow pandas
```
Also ensure images are downloaded from the NIH Chest X-Ray datasets available at: https://www.kaggle.com/datasets/nih-chest-xrays/data?resource=download&select=images_002. Ensure that the following folders are set up to input into the complete end-to-end system:

```bash
- Data_Entry_2017.csv
- training_images/
- testing_images/
- templates/ # contains the index.html file
- static/  # contains the app.js and the style.css files
```

## Training
Train the model utilising a balanced subset of X-ray images which are either No Finding or Pneumonia.

```bash
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model_definition import SimpleCNN, ChestXrayDataset

# Define transforms and datasets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = ChestXrayDataset(train_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Ensure the model is saved and within the current model directory:

```bash
torch.save(model.state_dict(), "model.pth")
```

## Prediction

Utilise the CNN model to predict a new set of X-Ray images from the same site.

```bash
from PIL import Image
from torchvision import transforms
from model_definition import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image_from_path(path):
    image = Image.open(path).convert("L")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        return "Pneumonia" if output.item() > 0.5 else "No Finding"
```

## Web Interface
The flask server can be run to start to interface:

```bash
python app.py
```

Where the http location can be opened within your browser. An examplar is shown below: 

No Finding:
![NoFinding](screenshots/egss.png)

Pneumonia Detected:
![Pneumonia](screenshots/egss.png)

## License

[MIT](https://choosealicense.com/licenses/mit/)

from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import torch
from torchvision import transforms
from model_definition import SimpleCNN, predict_image_from_path  # you'll separate your model + predict func here

import pandas as pd

# Load and process the label CSV once at startup
df = pd.read_csv("Data_Entry_2017.csv")
df = df[df["Finding Labels"].str.contains("Pneumonia") | (df["Finding Labels"] == "No Finding")]
df["Label"] = df["Finding Labels"].apply(lambda x: "Pneumonia" if "Pneumonia" in x else "No Finding")
df = df.set_index("Image Index")  # For fast filename-based lookup


# Init Flask app
app = Flask(__name__)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Define transform (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename  # e.g., "0001326_002.png"
    image = Image.open(file.stream).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = "Pneumonia" if output.item() > 0.5 else "No Finding"

    # Look up the ground truth label from the CSV
    try:
        truth = df.loc[filename]["Label"]
    except KeyError:
        truth = "--"

    return jsonify({
        "prediction": pred,
        "truth": truth
    })


if __name__ == "__main__":
    app.run(debug=True)


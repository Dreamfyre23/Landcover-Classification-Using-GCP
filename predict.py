from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import rasterio
import torch
import torch.nn as nn
import os
import tempfile
from collections import Counter
from einops import rearrange  # Ensure this is installed: pip install einops

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Define SpectralFormer properly
class SpectralFormer(nn.Module):
    def __init__(self, input_channels=6, patch_size=4, num_classes=5, dim=64, depth=6, heads=8, mlp_dim=128):
        super(SpectralFormer, self).__init__()
        
        self.patch_embedding = nn.Conv2d(input_channels, dim, kernel_size=patch_size, stride=patch_size)
        
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim),
            num_layers=depth
        )
        
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)  # Convert patches to feature embeddings
        x = rearrange(x, 'b c h w -> (h w) b c')  # Reshape for transformer input
        x = self.encoder_layers(x)  # Pass through transformer
        x = x.mean(dim=0)  # Global average pooling
        x = self.classifier(x)  # Classify land cover type
        return x

# ✅ Load Model Correctly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpectralFormer(input_channels=6, num_classes=5)  # Ensure the parameters match training
model.load_state_dict(torch.load("spectralformer_weights.pth", map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# Class Labels
class_labels = {
    0: 'Forest',
    1: 'HerbaceousVegetation',
    2: 'Pasture',
    3: 'River',
    4: 'SeaLake'
}

def preprocess_image(image_path):
    """Preprocess the satellite image"""
    with rasterio.open(image_path) as src:
        selected_bands = [3, 4, 6, 7, 8, 11]
        selected_bands = [b for b in range(1, 7) if b <= src.count]
        img = src.read(selected_bands)
    
    img = np.moveaxis(img, 0, -1)  # Convert shape (bands, H, W) → (H, W, bands)
    img = img.astype(np.float32) / 10000.0  # Normalize

    # Ensure 6 bands are present
    if img.shape[-1] < 6:
        diff = 6 - img.shape[-1]
        extra_bands = np.repeat(img[..., -1:], diff, axis=-1)
        img = np.concatenate([img, extra_bands], axis=-1)

    return img

def predict_landcover(image_path, patch_size=64):
    """Predict land cover for an image"""
    image = preprocess_image(image_path)
    h, w, c = image.shape
    
    patches = []
    coords = []
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
            coords.append((i, j))

    patches = np.array(patches)
    
    # Convert patches to PyTorch tensors
    patches_tensor = torch.tensor(patches, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # (batch, bands, H, W)
    
    with torch.no_grad():
        predictions = model(patches_tensor)  # Forward pass
        predicted_classes = torch.argmax(predictions, dim=1).cpu().numpy()

    return coords, predicted_classes

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.tif'):
        return jsonify({'error': 'File must be a .tif'}), 400
    
    try:
        # Save uploaded file to temp location
        temp_dir = tempfile.mkdtemp()
        upload_path = os.path.join(temp_dir, file.filename)
        file.save(upload_path)
        
        # Process the image
        coords, predicted_classes = predict_landcover(upload_path)
        
        # Prepare results
        predictions = [
            {"patch": f"{x},{y}", "class": class_labels[cls]}
            for (x, y), cls in zip(coords[:20], predicted_classes[:20])  # Show first 10 for demo
        ]
        
        # Calculate class distribution
        total = len(predicted_classes)
        class_dist = {
            class_labels[cls]: count/total
            for cls, count in Counter(predicted_classes).items()
        }
        
        # Clean up
        os.remove(upload_path)
        os.rmdir(temp_dir)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'class_distribution': class_dist
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

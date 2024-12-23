import os
from PIL import Image
import numpy as np

data = []
labels = []

DATA_DIR = './data'
characters = os.listdir(DATA_DIR)

for char in characters:
    class_dir = os.path.join(DATA_DIR, char)
    for filename in os.listdir(class_dir):
        filepath = os.path.join(class_dir, filename)
        img = Image.open(filepath).convert('RGB')
        data.append(np.asarray(img))
        labels.append(char)

data = np.asarray(data)  # Should work without errors
print("Dataset shape:", data.shape)
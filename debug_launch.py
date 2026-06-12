import os
import random
import traceback
import torch
from PIL import Image
from torchvision import transforms
from src.finance import calculate_mission_budget, calculate_science_roi
from src.robotics import RoverSimulation
from src.vision import MarsTerrainCNN, LABELS

print('PYTHON', os.sys.executable)
print('TORCH', torch.__version__)

budget = calculate_mission_budget({'compute':'Standard Rad-Hardened','sensors':'Basic Stereo Navcams','power':'Solar Arrays'})
print('budget', budget)

model = MarsTerrainCNN(num_classes=8)
weights_path = os.path.join('outputs','models','mars_terrain_cnn.pth')
print('weights_path', weights_path, os.path.exists(weights_path))
if os.path.exists(weights_path):
    try:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print('loaded weights')
    except Exception as e:
        print('weights load error', type(e).__name__, e)
else:
    print('weights file missing')

base_data_dir = os.path.join('data','raw')
print('base_data_dir', base_data_dir, os.path.isdir(base_data_dir))
image_pool = []
for label in LABELS:
    folder = os.path.join(base_data_dir, label)
    if os.path.exists(folder):
        images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_pool.extend(images[:3])
print('image pool size', len(image_pool))
if image_pool:
    mission_steps = image_pool[:8]
    transform_pipeline = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    rover = RoverSimulation(budget)
    for step, img_path in enumerate(mission_steps, 1):
        print('step', step, 'img', img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)
        raw_img = Image.open(img_path)
        if raw_img.mode != 'RGB':
            raw_img = raw_img.convert('RGB')
        input_tensor = transform_pipeline(raw_img).unsqueeze(0)
        print('tensor', input_tensor.shape)
        with torch.no_grad():
            logits = model(input_tensor)
        _, pred_idx = torch.max(logits, 1)
        prediction_string = LABELS[pred_idx.item()]
        print('prediction', prediction_string)
        action = rover.encounter_terrain(prediction_string)
        print('action', action)
else:
    print('no images available')

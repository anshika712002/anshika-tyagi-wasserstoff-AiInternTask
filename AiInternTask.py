import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import cv2
import numpy as np
import sqlite3
from ultralytics import YOLO
import pytesseract
from transformers import pipeline
import json
import matplotlib.pyplot as plt
import pandas as pd

# Load pre-trained Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Preprocess image
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    return transform(image)

# Perform inference and return prediction
def segment_image(image_path):
    image = preprocess(image_path)
    with torch.no_grad():
        prediction = model([image])
    return prediction

# Example usage
image_path = "input.png"
prediction = segment_image(image_path)

# Visualize the segmentation
def visualize_segmentation(image_path, prediction):
    image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    plt.imshow(prediction[0]['masks'][0, 0].cpu().numpy(), alpha=0.5)
    plt.show()

visualize_segmentation(image_path, prediction)

# Extract text using Tesseract OCR
def extract_text(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

# Extract segmented objects and save them as separate images
def extract_objects(image_path, prediction, output_dir="C:\\Users\\HP\\Desktop\\AiInternTask\\"):
    image = cv2.imread(image_path)
    masks = prediction[0]['masks']
    for i in range(len(masks)):
        mask = masks[i, 0].cpu().numpy()
        object_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        output_file = f"{output_dir}object_{i}.jpg"
        result = cv2.imwrite(output_file, object_image)
        if result:
            print(f"object_{i}.jpg saved successfully at {output_file}")
        else:
            print(f"Failed to save object_{i}.jpg")

extract_objects(image_path, prediction)

# Identify objects using YOLOv5
def identify_objects(image_path):
    model = YOLO('yolov5s.pt')  # Load pre-trained YOLOv5 model
    results = model(image_path)  # Run inference
    
    # Display the results for each image in the batch
    for result in results:
        result.show()  # This will display the image with detections
    
    # Extract bounding box data and convert it to a pandas DataFrame
    if results[0].boxes is not None:
        boxes = results[0].boxes.cpu().numpy()  # Move boxes to CPU and convert to numpy array
        data = {'xmin': boxes[:, 0], 'ymin': boxes[:, 1], 'xmax': boxes[:, 2], 'ymax': boxes[:, 3], 'confidence': boxes[:, 4], 'class': boxes[:, 5]}
        df = pd.DataFrame(data)  # Create a pandas DataFrame
        return df  # Return the DataFrame with bounding box details
    else:
        return None  # If no boxes are found, return None


# Example usage
objects = identify_objects(image_path)
if objects is not None:
    print(objects)
else:
    print("No objects detected.")

# Summarize text for each object
def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text)[0]['summary_text']  # Extracting summary text from the output
    return summary

# Summarize extracted text
text_data = extract_text("C:\\Users\\HP\\Desktop\\AiInternTask\\object_0.jpg")
if text_data:
    summary = summarize_text(text_data)
    print(summary)

# Assuming you have multiple objects and their summaries
summaries = [summarize_text(text_data)] if text_data else []  # Replace with summaries for multiple objects

# Map objects with summaries and bounding boxes
def map_data(objects, summaries):
    data = {}
    for obj, summary in zip(objects.iterrows(), summaries):
        obj_data = obj[1]  # Get the data from the Pandas row
        data[obj_data['name']] = {
            'box': [obj_data['xmin'], obj_data['ymin'], obj_data['xmax'], obj_data['ymax']],
            'summary': summary
            }
    
    with open('object_data.json', 'w') as f:
        json.dump(data, f)

map_data(objects, summaries)

# Generate the output image with annotations
def generate_output(image_path, objects):
    image = cv2.imread(image_path)
    for i, row in objects.iterrows():
        label = row['name']
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.putText(image, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

generate_output(image_path, objects)

# Store object metadata in SQLite database
def store_metadata(image_id, objects):
    conn = sqlite3.connect('objects.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS object_data
                 (object_id INTEGER PRIMARY KEY, image_id TEXT, object_image TEXT)''')
    
    for obj_id, obj_image in objects.iterrows():
        c.execute("INSERT INTO object_data (image_id, object_image) VALUES (?, ?)", (image_id, obj_image))
    
    conn.commit()
    conn.close()

import os
import cv2
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_img = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)

    # Convert the image to HSV color space for easier color detection
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    
    # Define the color range for yellow
    lower_orange = np.array([15, 100, 200], dtype=np.uint8)
    upper_orange = np.array([25, 255, 255], dtype=np.uint8)
    lower_yellowish_orange = np.array([10, 100, 180], dtype=np.uint8)
    upper_yellowish_orange = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask for yellow areas
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_yellowish_orange = cv2.inRange(hsv, lower_yellowish_orange, upper_yellowish_orange)
    combined_mask = cv2.bitwise_or(mask_orange, mask_yellowish_orange)
    
    processed_img = enhanced_img.copy()
    processed_img[combined_mask > 0] = [0, 0, 0]
    processed_img[combined_mask == 0] = [255, 255, 255]

    # Convert to grayscale for OCR
    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Apply thresholding for binarization
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

# Function to extract text from image
def extract_text_from_image(image_path):
    # Preprocess image and get it directly in memory
    processed_image = preprocess_image(image_path)
    
    # Convert processed image to PIL format for pytesseract
    pil_img = Image.fromarray(processed_image)
    
    # Extract text using pytesseract
    text = pytesseract.image_to_string(pil_img, lang="eng", config='--psm 6')
    return text

# Function to save text from multiple images to Excel
def save_multiple_images_to_excel(image_paths, output_file):
    all_data = []
    for image_path in image_paths:
        text = extract_text_from_image(image_path)
        if text:
            lines = [line for line in text.split('\n') if line.strip()]
            if len(lines) < 4:
                continue

            data_dict = {}
            day = f"{lines[0].strip()} {lines[1].strip()}"
            data_dict['Day'] = day

            for line in lines[2:]:
                if line.strip():
                    key_value = line.split(':', 1)
                    if len(key_value) == 2:
                        key, value = key_value
                        data_dict[key.strip()] = value.strip()

            all_data.append(data_dict)

    df = pd.DataFrame(all_data)
    df.to_excel(output_file, index=False)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading images
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return redirect(request.url)
    files = request.files.getlist('files[]')
    image_paths = []
    extracted_texts = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_paths.append(filepath)
            text = extract_text_from_image(filepath)
            extracted_texts.append(text)
    
    # Ensure the output path is correct and does not repeat `uploads`
    output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'output.xlsx')
    save_multiple_images_to_excel(image_paths, output_file)

    # Pass only the filename (not full path) to the template for download
    return render_template('index.html', extracted_texts=extracted_texts, output_file='output.xlsx')

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        return send_file(path, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)

import os
import cv2
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify
import time
import re
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
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

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

    # Keep only letters (a-z, A-Z) and numbers (0-9) using regex
    filtered_text = re.sub(r'[^a-zA-Z0-9\s . :]', '', text)  # Removes anything except letters and numbers
    
    return filtered_text

# Function to save text from multiple images to Excel
def save_multiple_images_to_excel(image_paths, output_file):
    all_data = []
    fields = ["Day", "Distance", "LFE index"]

    # Dictionary mapping standardized fields to lists of possible variations
    field_aliases = {
        "Day": ["Day"],
        "Distance": ["Distance", "Dist", "Ontagce", "Distance.", "Dis", "stan", "sta"],
        "LFE index": ["LFE index", "lSi index", "L Index", "Indes", "index", "ind", "dex", "des", "de", "FE"]
    }

    for image_path in image_paths:
        text = extract_text_from_image(image_path)
        if text:
            lines = [line for line in text.split('\n') if line.strip()]
            if len(lines) < 4:
                continue

            # Initialize a dictionary with all fields as empty strings
            data_dict = {field: "" for field in fields}
            
            # Set 'Day' field
            day = f"{lines[0].strip()} {lines[1].strip()}"
            data_dict['Day'] = day

            # Fill in data for other fields
            for line in lines[2:]:
                if line.strip():
                    # Try splitting by ':', ',', or space (in that order)
                    key_value = line.split(':', 1)  # First try splitting by ':'
                    if len(key_value) != 2:
                        key_value = line.split(',', 1)  # If no ':', try splitting by ','
                    if len(key_value) != 2:
                        key_value = line.split(None, 1)  # If no ':' or ',', try splitting by any whitespace

                    if len(key_value) == 2:
                        key, value = key_value
                        key = key.strip()
                        value = value.strip()

                        # Match the extracted key with possible aliases in field_aliases
                        for field, aliases in field_aliases.items():
                            if any(alias.lower() in key.lower() for alias in aliases):
                                # For "Distance" field, only extract the number (ignoring letters)
                                if field == "Distance":
                                    number_only = extract_number_only(value)
                                    data_dict[field] = number_only if number_only is not None else ""
                                # For "LFE index" field, extract the first digit of numbers like 53 or 45
                                elif field == "LFE index":
                                    first_digit = extract_first_digit(value)
                                    data_dict[field] = first_digit if first_digit is not None else ""
                                else:
                                    data_dict[field] = value
                                break

            all_data.append(data_dict)

    # Create DataFrame and write to Excel
    df = pd.DataFrame(all_data, columns=fields)
    df.to_excel(output_file, index=False)

def extract_first_number(value):
    """Extracts the first number (integer or decimal) from the string."""
    match = re.search(r"\d+(\.\d+)?", value)  # Match integers or decimals
    return match.group(0) if match else None

def extract_number_only(value):
    """Extracts only the number from the string, removing any letters or other characters."""
    match = re.search(r"\d+(\.\d+)?", value)  # Match only integers (ignores letters)
    return match.group(0) if match else None

def extract_first_digit(value):
    """Extracts the first digit of the number, if the number is 53 or 45 (just return 5 or 4)."""
    match = re.match(r"(\d)", value)  # Match the first digit in the string
    return match.group(0) if match else None



# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading images
@app.route('/upload', methods=['POST'])
def upload_file():
    # Xóa toàn bộ nội dung trong thư mục 'uploads' trước khi thêm file mới
    upload_folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(upload_folder):
        for file_name in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, file_name)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # Xóa file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Xóa thư mục con
            except Exception as e:
                print(f"Lỗi khi xóa {file_path}: {e}")

    # Kiểm tra xem có file trong request không
    if 'files[]' not in request.files:
        return redirect(request.url)
    files = request.files.getlist('files[]')
    image_paths = []
    extracted_texts = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            image_paths.append(filepath)
            text = extract_text_from_image(filepath)
            extracted_texts.append(text)

    # Tạo file Excel mới từ dữ liệu đã xử lý
    output_filename = f'output_{int(time.time())}.xlsx'
    output_file = os.path.join(upload_folder, output_filename)
    save_multiple_images_to_excel(image_paths, output_file)

    # Trả về giao diện hiển thị kết quả
    return render_template('index.html', extracted_texts=extracted_texts, output_file=output_filename)

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        return send_file(path, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)

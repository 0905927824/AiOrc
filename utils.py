import os
import re
import cv2
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd

# Tiền xử lý ảnh
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Phóng to ảnh
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

    # Tăng cường chất lượng hình ảnh
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_img = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    
    # Phát hiện màu cam và vàng
    lower_orange = np.array([15, 100, 200], dtype=np.uint8)
    upper_orange = np.array([25, 255, 255], dtype=np.uint8)
    lower_yellowish_orange = np.array([10, 100, 180], dtype=np.uint8)
    upper_yellowish_orange = np.array([20, 255, 255], dtype=np.uint8)
    
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask_yellowish_orange = cv2.inRange(hsv, lower_yellowish_orange, upper_yellowish_orange)
    combined_mask = cv2.bitwise_or(mask_orange, mask_yellowish_orange)
    
    processed_img = enhanced_img.copy()
    # Tô đen các vùng phát hiện
    processed_img[combined_mask > 0] = [0, 0, 0]
    processed_img[combined_mask == 0] = [255, 255, 255]

    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

# Trích xuất văn bản từ ảnh sử dụng OCR
def extract_text_from_image(image_path):
    processed_image = preprocess_image(image_path)
    pil_img = Image.fromarray(processed_image)
    text = pytesseract.image_to_string(pil_img, lang="eng", config='--psm 6')
    filtered_text = re.sub(r'[^a-zA-Z0-9\s . :]', '', text)  # Lọc các ký tự không mong muốn
    return filtered_text

# Lưu nhiều ảnh vào file Excel
def save_multiple_images_to_excel(image_paths, output_file):
    all_data = []
    fields = ["Day", "Distance", "LFE index"]

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
            data_dict = {field: "" for field in fields}
            day = f"{lines[0].strip()} {lines[1].strip()}"
            data_dict['Day'] = day

            for line in lines[2:]:
                if line.strip():
                    key_value = line.split(':', 1)
                    if len(key_value) != 2:
                        key_value = line.split(',', 1)
                    if len(key_value) != 2:
                        key_value = line.split(None, 1)
                    if len(key_value) == 2:
                        key, value = key_value
                        key = key.strip()
                        value = value.strip()

                        for field, aliases in field_aliases.items():
                            if any(alias.lower() in key.lower() for alias in aliases):
                                if field == "Distance":
                                    number_only = extract_number_only(value)
                                    data_dict[field] = number_only if number_only else ""
                                elif field == "LFE index":
                                    first_digit = extract_first_digit(value)
                                    data_dict[field] = first_digit if first_digit else ""
                                else:
                                    data_dict[field] = value
                                break

            all_data.append(data_dict)

    df = pd.DataFrame(all_data, columns=fields)
    df.to_excel(output_file, index=False)

def extract_first_number(value):
    match = re.search(r"\d+(\.\d+)?", value)
    return match.group(0) if match else None

def extract_number_only(value):
    match = re.search(r"\d+(\.\d+)?", value)
    return match.group(0) if match else None

def extract_first_digit(value):
    match = re.match(r"(\d)", value)
    return match.group(0) if match else None

import os
import re
import cv2
from google.cloud import vision
from PIL import Image
import numpy as np
import pandas as pd

# Trích xuất văn bản từ ảnh sử dụng OCR
def extract_text_from_image(image_path):
    # Tạo client cho Vision API
    client = vision.ImageAnnotatorClient()

    # Đọc file và mã hóa Base64
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Gửi yêu cầu OCR
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description  # Trả về văn bản đầu tiên (toàn bộ văn bản)
    return ""

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

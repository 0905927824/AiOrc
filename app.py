from flask import Flask, request, render_template, send_file, redirect, url_for
import os
import time
import shutil
from utils import preprocess_image, extract_text_from_image, save_multiple_images_to_excel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Kiểm tra định dạng file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
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

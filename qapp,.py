from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
from PIL import Image
import numpy as np
import dlib
from werkzeug.utils import secure_filename
import requests
import io
import math

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join('static', 'processed')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['CROP_API_KEY'] = ''
app.config['ENHANCEMENT_API_KEY'] = ''

predictor_path = 'C:/Users/tayla/Downloads/shape_predictor_81_face_landmarks (2).dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_face_rotate_and_crop(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector(gray, 1)

    if len(detections) == 0:
        print("No faces detected.")
        return None

    face = sorted(detections, key=lambda rect: rect.width() * rect.height(), reverse=True)[0]
    landmarks = predictor(gray, face)

    forehead_y = landmarks.part(21).y
    chin_y = landmarks.part(8).y
    left_ear_x = landmarks.part(0).x
    left_ear_y = landmarks.part(0).y
    right_ear_x = landmarks.part(16).x
    right_ear_y = landmarks.part(16).y

    delta_x = right_ear_x - left_ear_x
    delta_y = right_ear_y - left_ear_y
    angle = math.degrees(math.atan2(delta_y, delta_x))

    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    gray_rotated = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    detections_rotated = detector(gray_rotated, 1)

    if len(detections_rotated) == 0:
        print("No faces detected after rotation.")
        return None

    face_rotated = sorted(detections_rotated, key=lambda rect: rect.width() * rect.height(), reverse=True)[0]
    landmarks_rotated = predictor(gray_rotated, face_rotated)

    forehead_y_rotated = landmarks_rotated.part(21).y
    chin_y_rotated = landmarks_rotated.part(8).y
    left_ear_x_rotated = landmarks_rotated.part(0).x
    right_ear_x_rotated = landmarks_rotated.part(16).x

    face_height = chin_y_rotated - forehead_y_rotated
    desired_ratio = 5 / 6
    new_height = face_height / 0.35  
    new_width = new_height * desired_ratio

    center_x_rotated = (left_ear_x_rotated + right_ear_x_rotated) // 2
    crop_center_y = ((forehead_y_rotated + chin_y_rotated) // 2 - face_height * 0.3)

    crop_box = (
        int(center_x_rotated - new_width // 2),
        int(crop_center_y - new_height // 2),
        int(center_x_rotated + new_width // 2),
        int(crop_center_y + new_height // 2)
    )

    crop_box = (
        max(0, crop_box[0]),
        max(0, crop_box[1]),
        min(rotated_image.shape[1], crop_box[2]),
        min(rotated_image.shape[0], crop_box[3])
    )

    cropped_image = rotated_image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

    cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    return cropped_image_pil

def remove_background(crop_api_key, image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = buffered.getvalue()
    url = 'https://api.remove.bg/v1.0/removebg'
    response = requests.post(
        url,
        files={'image_file': ('image.png', img_str, 'image/png')},
        data={'size': 'auto'},
        headers={'X-Api-Key': crop_api_key}
    )

    if response.status_code == requests.codes.ok:
        image_data = io.BytesIO(response.content)
        image = Image.open(image_data)
        return image
    else:
        print("Error:", response.status_code, response.text)
        return None

def add_white_background(image):
    white_bg = Image.new("RGB", image.size, (255, 255, 255))
    white_bg.paste(image, (0, 0), image)
    return white_bg

def enhance_image(enhancement_api_key, image_path):
    url = 'https://www.cutout.pro/api/v1/photoEnhance'
    with open(image_path, 'rb') as file:
        response = requests.post(
            url,
            files={'file': file},
            headers={'APIKEY': enhancement_api_key}
        )

    if response.status_code == requests.codes.ok:
        enhanced_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'enhanced_' + os.path.basename(image_path))
        with open(enhanced_image_path, 'wb') as out:
            out.write(response.content)
        return enhanced_image_path
    else:
        print("Error:", response.status_code, response.text)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"Image saved to {filepath}")
        return jsonify({'filepath': filepath})

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    image_path = data['filepath']

    print(f"Processing image: {image_path}")

    cropped_image = detect_face_rotate_and_crop(image_path)
    if cropped_image is None:
        return jsonify({'error': 'No faces detected in the uploaded image.'}), 400
    print("Face detected, rotated, and cropped")

    cropped_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'cropped_' + os.path.basename(image_path))
    cropped_image.save(cropped_image_path)

    image_no_bg = remove_background(app.config['CROP_API_KEY'], cropped_image)
    if image_no_bg is None:
        return jsonify({'error': 'Error in background removal.'}), 500
    print("Background removed")

    final_image = add_white_background(image_no_bg)
    processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + os.path.basename(image_path))
    final_image.save(processed_image_path)
    print(f"Processed image saved to {processed_image_path}")

    enhanced_image_path = enhance_image(app.config['ENHANCEMENT_API_KEY'], processed_image_path)
    if enhanced_image_path is None:
        return jsonify({'error': 'Error in image enhancement.'}), 500
    print(f"Enhanced image saved to {enhanced_image_path}")

    return jsonify({'processed_image_url': processed_image_path, 'enhanced_image_url': enhanced_image_path})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/thank-you')
def thank_you():
    return "Thank you for using our service! Your processed photo is ready for download."

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)

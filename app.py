from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import requests
import io
import ctypes
from PIL import ImageFilter, ImageDraw, ImageFont

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join('static', 'processed')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['CROP_API_KEY'] = '13S43wEzX9BaNJ37rSWfx1ur'
app.config['ENHANCEMENT_API_KEY'] = 'c9b248213db94f029b6bfedc75d0da8a'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_short_path(path):
    """Convert to Windows short (8.3) path to avoid Unicode path issues."""
    if os.name != "nt":
        return path
    try:
        buf = ctypes.create_unicode_buffer(260)
        res = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 260)
        if res > 0 and res < 260:
            return buf.value
    except Exception as exc:
        print(f"GetShortPathName failed: {exc}")
    return path


def detect_face_and_crop(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image from {image_path}")
            return None

        # Get cascade file path - prioritize project directory to avoid Unicode path issues
        cascade_file = None
        project_dir = os.path.dirname(os.path.abspath(__file__))
        project_cascade = os.path.join(project_dir, 'haarcascade_frontalface_default.xml')
        project_cascade = os.path.normpath(project_cascade)
        
        # First priority: Use cascade from project directory (avoids Unicode path issues)
        if os.path.exists(project_cascade):
            cascade_file = project_cascade
            print(f"Using cascade from project directory: {cascade_file}")
        else:
            # Try to copy from OpenCV installation to project directory
            import shutil
            import sys
            source_paths = []
            
            # Try cv2.data.haarcascades
            try:
                if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                    path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
                    source_paths.append(os.path.normpath(path))
            except:
                pass
            
            # Try cv2 module directory
            try:
                cv2_dir = os.path.dirname(cv2.__file__)
                path = os.path.join(cv2_dir, 'data', 'haarcascade_frontalface_default.xml')
                source_paths.append(os.path.normpath(path))
            except:
                pass
            
            # Try site-packages
            try:
                path = os.path.join(sys.prefix, 'lib', 'site-packages', 'cv2', 'data', 'haarcascade_frontalface_default.xml')
                source_paths.append(os.path.normpath(path))
            except:
                pass
            
            # Try to copy from first existing source
            copied = False
            for source in source_paths:
                if source and os.path.exists(source):
                    try:
                        shutil.copy2(source, project_cascade)
                        cascade_file = project_cascade
                        print(f"Copied cascade to project directory from: {source}")
                        copied = True
                        break
                    except Exception as e:
                        print(f"Could not copy cascade: {e}")
                        continue
            
            # If copy failed, try to use source directly (may have Unicode issues on Windows)
            if not copied:
                for source in source_paths:
                    if source and os.path.exists(source):
                        cascade_file = source
                        print(f"Using cascade from OpenCV installation (may have Unicode issues): {source}")
                        break
        
        if not cascade_file or not os.path.exists(cascade_file):
            print(f"ERROR: Cascade file not found anywhere")
            return None
        
        # Load cascade - use absolute path
        cascade_file_abs = os.path.abspath(cascade_file)
        cascade_file_short = get_short_path(cascade_file_abs)
        print(f"Loading cascade from: {cascade_file_short}")
        
        try:
            face_cascade = cv2.CascadeClassifier(cascade_file_short)
        except Exception as e:
            print(f"Error loading cascade with abs path: {e}")
            # Try with original path
            try:
                face_cascade = cv2.CascadeClassifier(cascade_file_abs)
            except Exception as e2:
                print(f"Error loading cascade: {e2}")
                return None
        
        if face_cascade.empty():
            print("ERROR: Cascade classifier is empty after loading")
            return None
        
        # Convert to grayscale and enhance for better detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance image for better face detection
        # Apply histogram equalization to improve contrast
        gray_enhanced = cv2.equalizeHist(gray)
        
        # Try multiple detection parameters for better face detection
        faces = []
        
        # Strategy 1: Original grayscale with default params
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"Detection attempt 1 (default): {len(faces)} faces")
        
        # Strategy 2: Enhanced grayscale with default params
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray_enhanced, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            print(f"Detection attempt 2 (enhanced): {len(faces)} faces")
        
        # Strategy 3: More lenient parameters
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(20, 20))
            print(f"Detection attempt 3 (lenient): {len(faces)} faces")
        
        # Strategy 4: Enhanced with lenient parameters
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray_enhanced, scaleFactor=1.2, minNeighbors=3, minSize=(20, 20))
            print(f"Detection attempt 4 (enhanced+lenient): {len(faces)} faces")
        
        # Strategy 5: Very lenient parameters
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=2, minSize=(10, 10))
            print(f"Detection attempt 5 (very lenient): {len(faces)} faces")
        
        # Strategy 6: Enhanced with very lenient parameters
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray_enhanced, scaleFactor=1.3, minNeighbors=2, minSize=(10, 10))
            print(f"Detection attempt 6 (enhanced+very lenient): {len(faces)} faces")
        
        # Debug info
        print(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
        print(f"Final result: {len(faces)} face(s) detected")
        
        if len(faces) == 0:
            print("No faces detected with any parameters.")
            # Save debug images
            os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
            debug_gray = os.path.join(app.config['PROCESSED_FOLDER'], 'debug_gray.jpg')
            debug_enhanced = os.path.join(app.config['PROCESSED_FOLDER'], 'debug_enhanced.jpg')
            cv2.imwrite(debug_gray, gray)
            cv2.imwrite(debug_enhanced, gray_enhanced)
            print(f"Debug images saved: {debug_gray}, {debug_enhanced}")
            return None

        x, y, w, h = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)[0]
        crop_box = (x - w // 4, y - h // 4, x + 5 * w // 4, y + 5 * h // 4)
        crop_box = (max(0, crop_box[0]), max(0, crop_box[1]), min(image.shape[1], crop_box[2]), min(image.shape[0], crop_box[3]))
        cropped_image = image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        return cropped_image_pil
    except Exception as e:
        import traceback
        print(f"Error in detect_face_and_crop: {e}")
        traceback.print_exc()
        return None

def remove_background(crop_api_key, image):
    try:
        if not crop_api_key:
            print("Warning: CROP_API_KEY is empty")
            return None
            
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = buffered.getvalue()
        url = 'https://api.remove.bg/v1.0/removebg'
        
        print(f"Calling remove.bg API with key: {crop_api_key[:10]}...")
        response = requests.post(
            url,
            files={'image_file': ('image.png', img_str, 'image/png')},
            data={'size': 'auto'},
            headers={'X-Api-Key': crop_api_key},
            timeout=30
        )

        if response.status_code == requests.codes.ok:
            image_data = io.BytesIO(response.content)
            image = Image.open(image_data)
            return image
        else:
            print(f"remove.bg API Error: Status {response.status_code}, Response: {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        print("remove.bg API timeout")
        return None
    except Exception as exc:
        import traceback
        print(f"remove_background exception: {exc}")
        traceback.print_exc()
        return None

def add_white_background(image):
    white_bg = Image.new("RGB", image.size, (255, 255, 255))
    white_bg.paste(image, (0, 0), image)
    return white_bg


def blur_and_watermark(image_path, output_path, text="", blur_radius=6, opacity=90):
    """Create a blurred (optionally watermarked) version of the image for preview."""
    try:
        with Image.open(image_path).convert("RGBA") as im:
            blurred = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))

            # If text is provided, add watermark; otherwise only blur
            if text.strip():
                watermark = Image.new("RGBA", blurred.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(watermark)
                try:
                    font = ImageFont.truetype("arial.ttf", size=max(18, blurred.size[0] // 22))
                except Exception:
                    font = ImageFont.load_default()

                # textbbox is preferred; fallback to textsize if unavailable
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except Exception:
                    text_width, text_height = draw.textsize(text, font=font)
                padding = 20
                x = blurred.size[0] - text_width - padding
                y = blurred.size[1] - text_height - padding
                draw.text((x, y), text, font=font, fill=(255, 255, 255, opacity))

                combined = Image.alpha_composite(blurred, watermark)
            else:
                combined = blurred

            combined.convert("RGB").save(output_path)
            return output_path
    except Exception as exc:
        print(f"blur_and_watermark failed: {exc}")
        return None

def enhance_image(enhancement_api_key, image_path):
    url = 'https://www.cutout.pro/api/v1/photoEnhance'
    try:
        if not enhancement_api_key:
            print("Warning: ENHANCEMENT_API_KEY is empty")
            return None
            
        if not os.path.exists(image_path):
            print(f"Error: Image file not found for enhancement: {image_path}")
            return None
            
        print(f"Calling cutout.pro API with key: {enhancement_api_key[:10]}...")
        with open(image_path, 'rb') as file:
            response = requests.post(
                url,
                files={'file': file},
                headers={'APIKEY': enhancement_api_key},
                timeout=60
            )

        if response.status_code == requests.codes.ok:
            enhanced_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'enhanced_' + os.path.basename(image_path))
            os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
            with open(enhanced_image_path, 'wb') as out:
                out.write(response.content)
            return enhanced_image_path
        else:
            print(f"cutout.pro API Error: Status {response.status_code}, Response: {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        print("cutout.pro API timeout")
        return None
    except Exception as exc:
        import traceback
        print(f"enhance_image exception: {exc}")
        traceback.print_exc()
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
    try:
        import traceback
        data = request.json or {}
        image_path = data.get('filepath')
        if not image_path:
            return jsonify({'error': 'Missing filepath'}), 400

        # Check if file exists
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image file not found: {image_path}'}), 400

        print(f"Processing image: {image_path}")

        # Step 1: Face detection and cropping
        try:
            cropped_image = detect_face_and_crop(image_path)
            if cropped_image is None:
                return jsonify({'error': 'No faces detected or invalid image.'}), 400
            print("Face detected and cropped")
        except Exception as e:
            print(f"Error in face detection: {e}")
            traceback.print_exc()
            return jsonify({'error': f'Face detection failed: {str(e)}'}), 500

        # Step 2: Save cropped image
        try:
            cropped_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'cropped_' + os.path.basename(image_path))
            os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
            cropped_image.save(cropped_image_path)
            print(f"Cropped image saved to {cropped_image_path}")
        except Exception as e:
            print(f"Error saving cropped image: {e}")
            traceback.print_exc()
            return jsonify({'error': f'Failed to save cropped image: {str(e)}'}), 500

        # Step 3: Remove background
        try:
            image_no_bg = remove_background(app.config['CROP_API_KEY'], cropped_image)
            if image_no_bg is None:
                return jsonify({'error': 'Error in background removal. API may be unavailable or invalid key.'}), 500
            print("Background removed")
        except Exception as e:
            print(f"Error in background removal: {e}")
            traceback.print_exc()
            return jsonify({'error': f'Background removal failed: {str(e)}'}), 500

        # Step 4: Add white background
        try:
            final_image = add_white_background(image_no_bg)
            processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + os.path.basename(image_path))
            final_image.save(processed_image_path)
            print(f"Processed image saved to {processed_image_path}")
        except Exception as e:
            print(f"Error adding white background: {e}")
            traceback.print_exc()
            return jsonify({'error': f'Failed to add white background: {str(e)}'}), 500

        # Step 5: Enhance image
        try:
            enhanced_image_path = enhance_image(app.config['ENHANCEMENT_API_KEY'], processed_image_path)
            if enhanced_image_path is None:
                return jsonify({'error': 'Error in image enhancement. API may be unavailable or invalid key.'}), 500
            print(f"Enhanced image saved to {enhanced_image_path}")
        except Exception as e:
            print(f"Error in image enhancement: {e}")
            traceback.print_exc()
            return jsonify({'error': f'Image enhancement failed: {str(e)}'}), 500

        # Step 6: Create blurred + watermarked preview (for UI display/paywall)
        try:
            preview_path = os.path.join(app.config['PROCESSED_FOLDER'], 'preview_' + os.path.basename(image_path))
            wm_result = blur_and_watermark(processed_image_path, preview_path, text="")
            if wm_result:
                print(f"Watermarked preview saved to {wm_result}")
            else:
                wm_result = processed_image_path
                print("Watermark/blur failed; falling back to processed image")
        except Exception as e:
            print(f"Error creating watermarked preview: {e}")
            traceback.print_exc()
            wm_result = processed_image_path

        return jsonify({
            'processed_image_url': processed_image_path,
            'enhanced_image_url': enhanced_image_path,
            'watermarked_image_url': wm_result
        })
    except Exception as exc:
        import traceback
        error_trace = traceback.format_exc()
        print("Unexpected error in /process:")
        print(error_trace)
        return jsonify({'error': f'Unexpected server error: {str(exc)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/payment')
def payment():
    return render_template('payment.html')

@app.route('/thank-you')
def thank_you():
    return render_template('thank_you.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)

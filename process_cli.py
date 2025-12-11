"""
Process a local image to biometric version using app.py functions and save output in pics folder.
"""
import os
import shutil
import app


def process_image(src_path: str) -> str:
    """Run face detect -> background remove -> white bg -> enhance. Returns output path."""
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source image not found: {src_path}")

    # Ensure processed folder exists
    os.makedirs(app.app.config['PROCESSED_FOLDER'], exist_ok=True)

    # Step 1: detect and crop
    cropped = app.detect_face_and_crop(src_path)
    if cropped is None:
        raise RuntimeError("No face detected or cascade load failed.")
    cropped_path = os.path.join(app.app.config['PROCESSED_FOLDER'], 'cropped_cli.jpg')
    cropped.save(cropped_path)

    # Step 2: background removal
    no_bg = app.remove_background(app.app.config['CROP_API_KEY'], cropped)
    if no_bg is None:
        raise RuntimeError("Background removal failed (check API key/network).")

    # Step 3: add white background
    final_img = app.add_white_background(no_bg)
    processed_path = os.path.join(app.app.config['PROCESSED_FOLDER'], 'processed_cli.jpg')
    final_img.save(processed_path)

    # Step 4: enhance
    enhanced_path = app.enhance_image(app.app.config['ENHANCEMENT_API_KEY'], processed_path)
    if enhanced_path is None:
        enhanced_path = processed_path  # fallback

    return enhanced_path


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    pics_dir = os.path.join(project_dir, "pics")

    # Default source file
    default_src = os.path.join(pics_dir, "istockphoto-1289220545-612x612.jpg")

    src = default_src
    if not os.path.exists(src):
        # Pick first jpg/png if default missing
        candidates = [f for f in os.listdir(pics_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not candidates:
            raise FileNotFoundError("No image found in pics folder.")
        src = os.path.join(pics_dir, candidates[0])

    print(f"Processing source: {src}")
    out_path = process_image(src)

    dest_name = f"biometric_{os.path.basename(src)}"
    dest_path = os.path.join(pics_dir, dest_name)
    shutil.copy2(out_path, dest_path)

    print(f"Saved biometric image to: {dest_path}")


if __name__ == "__main__":
    main()


"""Script to copy Haar Cascade file to project directory to avoid Unicode path issues"""
import os
import shutil
import cv2
import sys

def copy_cascade_to_project():
    """Copy haarcascade_frontalface_default.xml to project directory"""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(project_dir, 'haarcascade_frontalface_default.xml')
    
    # Try to find source file
    source_paths = []
    
    # Try cv2.data.haarcascades
    try:
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            source_paths.append(path)
    except:
        pass
    
    # Try cv2 module directory
    try:
        cv2_dir = os.path.dirname(cv2.__file__)
        path = os.path.join(cv2_dir, 'data', 'haarcascade_frontalface_default.xml')
        source_paths.append(path)
    except:
        pass
    
    # Try site-packages
    try:
        path = os.path.join(sys.prefix, 'lib', 'site-packages', 'cv2', 'data', 'haarcascade_frontalface_default.xml')
        source_paths.append(path)
    except:
        pass
    
    # Try to copy from first existing source
    for source in source_paths:
        if source and os.path.exists(source):
            try:
                shutil.copy2(source, target_file)
                print(f"Successfully copied cascade file from:")
                print(f"  Source: {source}")
                print(f"  Target: {target_file}")
                return True
            except Exception as e:
                print(f"Failed to copy from {source}: {e}")
                continue
    
    print("Could not find or copy cascade file")
    return False

if __name__ == '__main__':
    copy_cascade_to_project()


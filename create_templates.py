import os

# Define the path to the templates directory
templates_path = 'C:/Users/tayla/Downloads//biometric_photo_maker/templates/'

# Ensure the directory exists
os.makedirs(templates_path, exist_ok=True)

# Define the contents of the upload.html file
upload_html_content = """
<!doctype html>
<html>
<head>
    <title>Upload Photo</title>
</head>
<body>
    <h1>Upload a Photo</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
</body>
</html>
"""

# Define the contents of the uploaded.html file
uploaded_html_content = """
<!doctype html>
<html>
<head>
    <title>Uploaded Photo</title>
</head>
<body>
    <h1>Photo Uploaded Successfully</h1>
    <img src="{{ url_for('static', filename='uploads/' ~ filename) }}" alt="Uploaded Image">
    <p>Filename: {{ filename }}</p>
</body>
</html>
"""

# Save the upload.html file
with open(os.path.join(templates_path, 'upload.html'), 'w') as file:
    file.write(upload_html_content)

# Save the uploaded.html file
with open(os.path.join(templates_path, 'uploaded.html'), 'w') as file:
    file.write(uploaded_html_content)

print("Templates created successfully.")

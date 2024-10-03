import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import cv2
import tempfile

# Define ASCII characters from dark to light
ASCII_CHARS = [
    '$', '@', 'B', '%', '8', '&', 'W', 'M', '#', '*', 'o', 'a', 'h', 'k',
    'b', 'd', 'p', 'q', 'w', 'm', 'Z', 'O', '0', 'Q', 'L', 'C', 'J', 'U',
    'Y', 'X', 'z', 'c', 'v', 'u', 'n', 'x', 'r', 'j', 'f', 't', '/', '\\',
    '|', '(', ')', '1', '{', '}', '[', ']', '?', '-', '_', '+', '~', '<',
    '>', 'i', '!', 'l', 'I', ';', ':', '"', '^', '`', "'", '.', ' '
]


def image_to_ascii(image, new_width=100, edge_detection=False):
    # Resize image while maintaining aspect ratio
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.55)  # Adjust for font aspect ratio
    image = image.resize((new_width, new_height))

    # Convert image to grayscale
    image = image.convert('L')

    # Apply edge detection if selected
    if edge_detection:
        # Convert PIL image to OpenCV format
        img_cv = np.array(image)
        # Apply Canny edge detection
        edges = cv2.Canny(img_cv, threshold1=100, threshold2=200)
        # Invert colors: edges are white on black background
        edges = 255 - edges
        # Convert back to PIL image
        image = Image.fromarray(edges)
    else:
        # Enhance contrast
        image = ImageOps.autocontrast(image)

    # Convert image to numpy array
    pixels = np.array(image)

    # Normalize pixel values to match the ASCII character list
    pixels_normalized = (pixels / 255) * (len(ASCII_CHARS) - 1)
    pixels_normalized = pixels_normalized.astype(int)

    # Map pixels to ASCII characters
    ascii_chars = np.array(ASCII_CHARS)
    ascii_image = ascii_chars[pixels_normalized]

    # Convert to a string
    ascii_str = '\n'.join(''.join(row) for row in ascii_image)

    # Prepare HTML output
    ascii_html = f'<pre style="font-family: monospace; font-size:6px; line-height:6px">{ascii_str}</pre>'

    # Save ASCII art to a temporary text file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as f:
        f.write(ascii_str)
        temp_file_name = f.name

    return ascii_html, temp_file_name


# Define Gradio interface components
inputs = [
    gr.Image(type='pil', label='Upload Image'),
    gr.Slider(minimum=50, maximum=200, value=100, label='Width'),
    gr.Checkbox(label='Apply Edge Detection')
]

outputs = [
    gr.HTML(label='ASCII Art'),
    gr.File(label='Download ASCII Art')
]

title = "ASCII Art Generator"
description = "Upload an image and convert it to ASCII art. Adjust the width and optionally apply edge detection."

# Create Gradio Interface
iface = gr.Interface(
    fn=image_to_ascii,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    allow_flagging=False
)

iface.launch()

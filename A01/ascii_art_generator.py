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


def image_to_ascii(image, new_width=100, edge_method='None'):
    if image is None:
        return None, None  # Return None values if no image is provided

    # Resize image while maintaining aspect ratio
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width * 0.55)  # Adjust for font aspect ratio
    image = image.resize((new_width, new_height))

    # Convert image to grayscale
    image = image.convert('L')
    img_cv = np.array(image)

    # Apply edge detection based on user selection
    if edge_method != 'None':
        if edge_method == 'Canny':
            # Canny edge detection
            edges = cv2.Canny(img_cv, threshold1=100, threshold2=200)
        elif edge_method == 'Sobel':
            # Sobel edge detection
            sobel_x = cv2.Sobel(img_cv, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(img_cv, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.magnitude(sobel_x, sobel_y)
            edges = np.uint8(edges / np.max(edges) * 255) if np.max(edges) != 0 else np.uint8(edges)
        elif edge_method == 'Laplacian':
            # Laplacian edge detection
            edges = cv2.Laplacian(img_cv, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges) / np.max(np.absolute(edges)) * 255) if np.max(edges) != 0 else np.uint8(
                edges)
        else:
            edges = img_cv  # No edge detection
        # Invert colors: edges are white on black background
        edges = 255 - edges
        # Convert back to PIL image
        image = Image.fromarray(edges)

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

    # Save ASCII art to a temporary text file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as f:
        f.write(ascii_str)
        temp_file_name = f.name

    # Return the temporary file name and the ASCII art string
    return temp_file_name, ascii_str


title = "ASCII Art Generator by Tianhao Zhang"
description = ("Upload an image and convert it to ASCII art. You can adjust the art's width and choose different edge detection "
               "methods.")

# Create Gradio Interface using Blocks
with gr.Blocks(title=title) as demo:
    gr.Markdown(f"## {title}")
    gr.Markdown(description)

    # Outputs at the top, initially hidden
    with gr.Column():
        full_ascii_code = gr.Code(label='Full ASCII Art', visible=False)
        download_file = gr.File(label='Download ASCII Art', visible=False)

    # Inputs
    with gr.Row():
        # Left column: Upload Image
        with gr.Column():
            image_input = gr.Image(type='pil', label='Upload Image')
        # Right column: Width and Edge Detection Method
        with gr.Column():
            width_slider = gr.Slider(minimum=50, maximum=200, value=100, label='Width')
            edge_method = gr.Radio(
                choices=['None', 'Canny', 'Sobel', 'Laplacian'],
                value='None',
                label='Edge Detection Method'
            )
            generate_button = gr.Button("Generate ASCII Art")

    # Function to generate ASCII art and update outputs
    def generate_ascii(image, new_width, edge_method):
        result = image_to_ascii(image, new_width, edge_method)
        if result is None:
            # No image provided or an error occurred
            return [
                gr.update(visible=False),
                gr.update(visible=False)
            ]
        else:
            temp_file_name, ascii_str = result
            return [
                gr.update(value=ascii_str, visible=True),
                gr.update(value=temp_file_name, visible=True)
            ]


    # Link the button click to the function
    generate_button.click(
        fn=generate_ascii,
        inputs=[image_input, width_slider, edge_method],
        outputs=[full_ascii_code, download_file],
    )

demo.launch()

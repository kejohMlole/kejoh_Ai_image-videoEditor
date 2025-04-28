from flask import Flask, request, render_template, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from difflib import get_close_matches
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Supported actions for command suggestions
SUPPORTED_ACTIONS = [
    "grayscale", "mirror", "rotate", "enhance", "invert",
    "crop", "blur", "add border", "add text", "resize",
    "brightness", "contrast", "hue", "saturation", "sharpen",
    "flip", "edge enhance", "emboss", "find edges"
]

# Global variables to track state
last_suggested_action = None
edit_history = []
current_step = -1

# Home route
@app.route('/')
def index():
    global last_suggested_action, edit_history, current_step
    last_suggested_action = None
    edit_history = []
    current_step = -1
    return render_template('index.html')

# File upload endpoint
@app.route('/upload', methods=['POST'])
def upload_file():
    global last_suggested_action, edit_history, current_step
    last_suggested_action = None
    edit_history = []
    current_step = -1
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize edit history with the original file
        edit_history = [filepath]
        current_step = 0
        
        return jsonify({'success': True, 'filepath': filepath})

# Chat endpoint for image processing
@app.route('/chat', methods=['POST'])
def chat():
    global last_suggested_action, edit_history, current_step
    try:
        data = request.json
        user_input = data.get('user_input', '').lower()
        action = data.get('action', '').lower()
        filepath = edit_history[current_step] if edit_history else None
        param1 = data.get('param1')
        param2 = data.get('param2')
        
        # Check if user is confirming a previous suggestion
        if last_suggested_action and user_input in ['yes', 'yeah', 'y', 'confirm']:
            action = last_suggested_action
            last_suggested_action = None
            
        # Check if action is supported
        if action not in SUPPORTED_ACTIONS:
            matches = get_close_matches(action, SUPPORTED_ACTIONS, n=1, cutoff=0.6)
            if matches:
                last_suggested_action = matches[0]
                return jsonify({'success': False, 'error': f'Did you mean "{matches[0]}"? Type "yes" to confirm.'})
            else:
                last_suggested_action = None
                return jsonify({'success': False, 'error': 'I didn\'t understand that. Type "help" to see what I can do.'})
        
        kwargs = {}
        
        if action == 'rotate' and param1:
            try:
                kwargs['angle'] = int(param1)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid angle value'})
        
        elif action == 'add text' and param1:
            kwargs['text'] = param1
        
        elif action == 'resize' and param1 and param2:
            try:
                kwargs['width'] = int(param1)
                kwargs['height'] = int(param2)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid dimensions'})
        
        elif action == 'brightness' and param1:
            try:
                kwargs['value'] = float(param1)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid brightness value'})
        
        elif action == 'contrast' and param1:
            try:
                kwargs['value'] = float(param1)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid contrast value'})
        
        elif action == 'hue' and param1:
            try:
                kwargs['value'] = float(param1)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid hue value'})
        
        elif action == 'saturation' and param1:
            try:
                kwargs['value'] = float(param1)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid saturation value'})
        
        elif action == 'crop' and param1 and param2 and param1 and param2:
            try:
                kwargs['x'] = int(param1)
                kwargs['y'] = int(param2)
                kwargs['width'] = int(param1)
                kwargs['height'] = int(param2)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid crop dimensions'})
        
        # Process the image
        if filepath.endswith(('.jpg', '.png', '.jpeg')):
            result_path = edit_image(filepath, action, **kwargs)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file type'})
        
        # Update edit history
        if current_step < len(edit_history) - 1:
            edit_history = edit_history[:current_step + 1]
        
        edit_history.append(result_path)
        current_step = len(edit_history) - 1
        
        last_suggested_action = None
        return jsonify({
            'success': True,
            'result_path': result_path,
            'message': f'Applied: {action.replace("_", " ").title()} ✅',
            'download_link': f'/download?path={result_path}'
        })
    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        last_suggested_action = None
        return jsonify({'success': False, 'error': 'An error occurred while processing your request'})

# Help endpoint
@app.route('/help', methods=['GET'])
def help():
    global last_suggested_action
    last_suggested_action = None
    supported_actions = [
        "grayscale - Convert to grayscale",
        "mirror - Mirror the image",
        "rotate [angle] - Rotate the image (e.g., rotate 90)",
        "enhance - Enhance image contrast",
        "invert - Invert colors",
        "crop [x] [y] [width] [height] - Crop the image",
        "blur - Add blur effect",
        "add border - Add a black border",
        "add text [text] - Add text to the image",
        "resize [width] [height] - Resize the image",
        "brightness [value] - Adjust brightness",
        "contrast [value] - Adjust contrast",
        "hue [value] - Adjust hue",
        "saturation [value] - Adjust saturation",
        "sharpen - Apply sharpening",
        "flip [horizontal/vertical] - Flip image",
        "edge enhance - Enhance edges",
        "emboss - Apply emboss effect",
        "find edges - Detect edges in image"
    ]
    help_message = "Here's what I can do:\n" + "\n".join(supported_actions)
    return jsonify({'message': help_message})

# Undo endpoint
@app.route('/undo', methods=['POST'])
def undo():
    global current_step
    if current_step > 0:
        current_step -= 1
        return jsonify({
            'success': True,
            'message': 'Reverted to previous edit.',
            'result_path': edit_history[current_step]
        })
    else:
        return jsonify({'success': False, 'error': 'No previous edit to revert to.'})

# Redo endpoint
@app.route('/redo', methods=['POST'])
def redo():
    global current_step
    if current_step < len(edit_history) - 1:
        current_step += 1
        return jsonify({
            'success': True,
            'message': 'Redid the last undone edit.',
            'result_path': edit_history[current_step]
        })
    else:
        return jsonify({'success': False, 'error': 'No undone edit to redo.'})

# Download endpoint
@app.route('/download', methods=['GET'])
def download():
    path = request.args.get('path')
    if not path:
        return jsonify({'error': 'No file path provided'})
    
    try:
        return send_file(path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)})

# Function to edit images using PIL
def edit_image(filepath, action, **kwargs):
    img = Image.open(filepath)
    
    if action == 'grayscale':
        img = img.convert('L')
    
    elif action == 'mirror':
        img = ImageOps.mirror(img)
    
    elif action == 'rotate':
        angle = kwargs.get('angle', 90)
        img = img.rotate(angle, expand=True)
    
    elif action == 'enhance':
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
    
    elif action == 'invert':
        img = ImageOps.invert(img)
    
    elif action == 'blur':
        img = img.filter(ImageFilter.BLUR)
    
    elif action == 'add border':
        img = ImageOps.expand(img, border=10, fill='black')
    
    elif action == 'brightness':
        value = kwargs.get('value', 1.0)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(value)
    
    elif action == 'contrast':
        value = kwargs.get('value', 1.0)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(value)
    
    elif action == 'hue':
        # Convert to HSV, adjust hue, convert back
        hsv = img.convert('HSV')
        h, s, v = hsv.split()
        h = h.point(lambda x: (x + int(kwargs.get('value', 0) * 255) % 255)
        img = Image.merge('HSV', (h, s, v)).convert('RGB')
    
    elif action == 'saturation':
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(kwargs.get('value', 1.0))
    
    elif action == 'sharpen':
        img = img.filter(ImageFilter.SHARPEN)
    
    elif action == 'flip':
        mode = kwargs.get('mode', 'horizontal')
        if mode == 'horizontal':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif mode == 'vertical':
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    elif action == 'edge enhance':
        img = img.filter(ImageFilter.EDGE_ENHANCE)
    
    elif action == 'emboss':
        img = img.filter(ImageFilter.EMBOSS)
    
    elif action == 'find edges':
        img = img.filter(ImageFilter.FIND_EDGES)
    
    elif action == 'crop':
        x = kwargs.get('x', 0)
        y = kwargs.get('y', 0)
        width = kwargs.get('width', img.width)
        height = kwargs.get('height', img.height)
        img = img.crop((x, y, x + width, y + height))
    
    # Save the modified image
    result_path = f"static/edited_{os.path.basename(filepath)}"
    img.save(result_path)
    return result_path

if __name__ == '__main__':
    app.run(debug=True)

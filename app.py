from flask import Flask, request, render_template, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from difflib import get_close_matches
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

# Supported actions for command suggestions
SUPPORTED_ACTIONS = [
    "grayscale", "mirror", "rotate", "enhance", "invert",
    "crop", "remove background", "blur", "add border",
    "add text", "resize", "trim", "speed", "change color",
    "brightness", "contrast", "exposure", "white balance",
    "hue", "saturation", "luminance", "sharpen", "noise reduction",
    "red-eye removal", "flip", "panorama", "hdr", "layers", "clone stamp",
    "healing brush", "gradient", "selective adjustments", "filters",
    "effects", "text overlay", "background removal", "batch processing",
    "focus stacking", "content-aware fill", "blending modes", "histogram",
    "channel mixing", "smart objects", "vector editing", "3d editing",
    "animation frames", "scripting", "automation"
]

# Load a pre-trained YOLO model for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

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

# Chat endpoint for image/video processing
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
        param3 = data.get('param3')
        param4 = data.get('param4')
        
        # Check if user is confirming a previous suggestion
        if last_suggested_action and user_input in ['yes', 'yeah', 'y', 'confirm']:
            action = last_suggested_action
            last_suggested_action = None  # Reset after using
            
        # Check if action is supported
        if action not in SUPPORTED_ACTIONS:
            # Find close matches
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
        
        elif action == 'trim' and param1 and param2:  # For video trimming
            try:
                kwargs['start'] = int(param1)
                kwargs['end'] = int(param2)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid time values'})
        
        elif action == 'speed' and param1:  # For video speed adjustment
            try:
                kwargs['speed'] = float(param1)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid speed value'})
        
        elif action == 'change color' and param1:
            kwargs['color'] = param1
        
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
        
        elif action == 'exposure' and param1:
            try:
                kwargs['value'] = float(param1)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid exposure value'})
        
        elif action == 'white balance' and param1:
            kwargs['mode'] = param1
        
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
        
        elif action == 'luminance' and param1:
            try:
                kwargs['value'] = float(param1)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid luminance value'})
        
        elif action == 'sharpen' and param1:
            try:
                kwargs['value'] = float(param1)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid sharpen value'})
        
        elif action == 'noise reduction' and param1:
            try:
                kwargs['value'] = float(param1)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid noise reduction value'})
        
        elif action == 'red-eye removal':
            kwargs['apply'] = True
        
        elif action == 'flip' and param1:
            kwargs['mode'] = param1
        
        elif action == 'crop' and param1 and param2 and param3 and param4:
            try:
                kwargs['x'] = int(param1)
                kwargs['y'] = int(param2)
                kwargs['width'] = int(param3)
                kwargs['height'] = int(param4)
            except ValueError:
                return jsonify({'success': False, 'error': 'Invalid crop dimensions'})
        
        # Determine if the file is an image or video
        if filepath.endswith(('.jpg', '.png', '.jpeg')):
            result_path = edit_image(filepath, action, **kwargs)
        elif filepath.endswith(('.mp4', '.mov', '.avi')):
            result_path = edit_video(filepath, action, **kwargs)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file type'})
        
        # Discard any future history if not at the latest step
        if current_step < len(edit_history) - 1:
            edit_history = edit_history[:current_step + 1]
        
        edit_history.append(result_path)
        current_step = len(edit_history) - 1
        
        last_suggested_action = None  # Reset after successful action
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
        "crop [x] [y] [width] [height] - Crop the image (e.g., crop 10 10 100 100)",
        "remove background - Remove background",
        "blur - Add blur effect",
        "add border - Add a black border",
        "add text [text] - Add text to the image (e.g., add text Hello)",
        "resize [width] [height] - Resize the image (e.g., resize 300 300)",
        "trim [start] [end] - Trim a video (e.g., trim 10 20)",
        "speed [value] - Adjust video speed (e.g., speed 2.0)",
        "change color [color] - Change the color of detected objects (e.g., change color red)",
        "brightness [value] - Adjust brightness (e.g., brightness 1.5)",
        "contrast [value] - Adjust contrast (e.g., contrast 1.5)",
        "exposure [value] - Adjust exposure (e.g., exposure 1.5)",
        "white balance [mode] - Adjust white balance (e.g., white balance auto)",
        "hue [value] - Adjust hue (e.g., hue 0.5)",
        "saturation [value] - Adjust saturation (e.g., saturation 1.5)",
        "luminance [value] - Adjust luminance (e.g., luminance 1.5)",
        "sharpen [value] - Apply sharpening (e.g., sharpen 1.5)",
        "noise reduction [value] - Apply noise reduction (e.g., noise reduction 1.5)",
        "red-eye removal - Remove red-eye effect",
        "flip [mode] - Flip image (e.g., flip horizontal)",
        "undo - Revert to the previous edit",
        "redo - Redo the last undone edit",
        "download - Download the edited file"
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

# Function to edit images
def edit_image(filepath, action, **kwargs):
    # Load the image
    img = cv2.imread(filepath)
    
    if action == 'grayscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    elif action == 'mirror':
        img = cv2.flip(img, 1)
    
    elif action == 'rotate':
        angle = kwargs.get('angle', 90)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
    
    elif action == 'enhance':
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    
    elif action == 'invert':
        img = cv2.bitwise_not(img)
    
    elif action == 'blur':
        img = cv2.GaussianBlur(img, (15, 15), 0)
    
    elif action == 'add border':
        img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    elif action == 'change color':
        color = kwargs.get('color', 'red')
        color_map = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255)
        }
        target_color = color_map.get(color.lower(), (0, 0, 255))  # Default to red
        
        # Convert the image to RGB for YOLO
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Perform object detection
        results = model(img_rgb)
        
        # Get the bounding boxes of detected objects
        boxes = results.pandas().xyxy[0]
        
        # Iterate over each detected object and change its color
        for _, box in boxes.iterrows():
            xmin, ymin, xmax, ymax = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])
            img[ymin:ymax, xmin:xmax] = target_color
    
    elif action == 'brightness':
        value = kwargs.get('value', 1.0)
        img = cv2.convertScaleAbs(img, alpha=value, beta=0)
    
    elif action == 'contrast':
        value = kwargs.get('value', 1.0)
        img = cv2.convertScaleAbs(img, alpha=value, beta=0)
    
    elif action == 'exposure':
        value = kwargs.get('value', 1.0)
        img = cv2.convertScaleAbs(img, alpha=value, beta=0)
    
    elif action == 'white balance':
        mode = kwargs.get('mode', 'auto')
        if mode == 'auto':
            # Placeholder for auto white balance
            pass
    
    elif action == 'hue':
        value = kwargs.get('value', 0.0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] += int(value * 180)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif action == 'saturation':
        value = kwargs.get('value', 1.0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * value, 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif action == 'luminance':
        value = kwargs.get('value', 1.0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value, 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif action == 'sharpen':
        value = kwargs.get('value', 1.0)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel * value)
    
    elif action == 'noise reduction':
        value = kwargs.get('value', 1.0)
        img = cv2.fastNlMeansDenoisingColored(img, None, value, value, 7, 21)
    
    elif action == 'red-eye removal':
        # Placeholder for red-eye removal
        pass
    
    elif action == 'flip':
        mode = kwargs.get('mode', 'horizontal')
        if mode == 'horizontal':
            img = cv2.flip(img, 1)
        elif mode == 'vertical':
            img = cv2.flip(img, 0)
    
    elif action == 'crop':
        x = kwargs.get('x', 0)
        y = kwargs.get('y', 0)
        width = kwargs.get('width', img.shape[1])
        height = kwargs.get('height', img.shape[0])
        img = img[y:y+height, x:x+width]
    
    # Save the modified image
    result_path = f"static/edited_{os.path.basename(filepath)}"
    cv2.imwrite(result_path, img)
    return result_path

# Function to edit videos (simplified example)
def edit_video(filepath, action, **kwargs):
    # This is a placeholder for video editing logic
    # In a real application, you would use libraries like OpenCV or MoviePy
    result_path = f"static/edited_{os.path.basename(filepath)}"
    return result_path

if __name__ == '__main__':
    app.run(debug=True)
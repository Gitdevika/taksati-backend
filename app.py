from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import mediapipe as mp
import io

app = Flask(__name__)
# Replace the simple CORS(app) with a more explicit setup
cors = CORS(app, resources={r"/*": {"origins": "*"}})



UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/try-on', methods=['POST'])
def try_on():
    try:
        user_img = None
        shirt_img = None
        debug_info = {}  # For tracking what's happening

        # Process user image
        if 'user_image' in request.files:
            user_image = request.files['user_image']
            debug_info['user_source'] = 'file'
            debug_info['user_filename'] = user_image.filename
            
            user_path = os.path.join(UPLOAD_FOLDER, secure_filename(user_image.filename))
            user_image.save(user_path)
            user_img = cv2.imread(user_path)
            if user_img is not None:
                user_img = cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB)
                debug_info['user_image_loaded'] = True
            else:
                debug_info['user_image_loaded'] = False

        # Case 2: User image as bytes in the request body
        elif 'user_image_bytes' in request.form:
            user_bytes = request.form['user_image_bytes']
            debug_info['user_source'] = 'bytes'
            debug_info['user_bytes_length'] = len(user_bytes)
            
            # Convert base64 to image
            user_img = bytes_to_cv_image(user_bytes)
            debug_info['user_image_loaded'] = user_img is not None
            
        else:
            return jsonify({"error": "No user image provided"}), 400

        # Process shirt image
        if 'shirt_image' in request.files:
            shirt_image = request.files['shirt_image']
            debug_info['shirt_source'] = 'file'
            debug_info['shirt_filename'] = shirt_image.filename
            
            shirt_path = os.path.join(UPLOAD_FOLDER, secure_filename(shirt_image.filename))
            shirt_image.save(shirt_path)
            shirt_img = cv2.imread(shirt_path)
            if shirt_img is not None:
                shirt_img = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2RGB)
                debug_info['shirt_image_loaded'] = True
            else:
                debug_info['shirt_image_loaded'] = False
            
        elif 'shirt_image_bytes' in request.form:
            shirt_bytes = request.form['shirt_image_bytes']
            debug_info['shirt_source'] = 'bytes'
            debug_info['shirt_bytes_length'] = len(shirt_bytes)
            
            # Convert base64 to image
            shirt_img = bytes_to_cv_image(shirt_bytes)
            debug_info['shirt_image_loaded'] = shirt_img is not None
            
            # Save the image for debugging
            if shirt_img is not None:
                debug_path = os.path.join(UPLOAD_FOLDER, "debug_shirt.png")
                cv2.imwrite(debug_path, cv2.cvtColor(shirt_img, cv2.COLOR_RGB2BGR))
                debug_info['debug_shirt_saved'] = True
        else:
            # Handle the case where shirt image is passed directly as bytes in multipart/form-data
            # This handles the case from TryOnPage where shirtImage = widget.imageBytes
            shirt_image = request.files.get('shirt_image')
            if shirt_image:
                debug_info['shirt_source'] = 'multipart_bytes'
                
                # Read the raw bytes
                shirt_bytes = shirt_image.read()
                debug_info['shirt_bytes_length'] = len(shirt_bytes)
                
                # Convert to numpy array and decode
                nparr = np.frombuffer(shirt_bytes, np.uint8)
                try:
                    decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    if decoded is not None:
                        if decoded.shape[2] == 4:  # Has alpha channel
                            # Extract RGB and store alpha separately
                            shirt_img = cv2.cvtColor(decoded[:,:,:3], cv2.COLOR_BGR2RGB)
                            shirt_img.alpha_channel = decoded[:,:,3]
                        else:
                            shirt_img = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                        
                        debug_info['shirt_image_loaded'] = True
                        
                        # Save for debugging
                        debug_path = os.path.join(UPLOAD_FOLDER, "debug_multipart_shirt.png")
                        cv2.imwrite(debug_path, cv2.cvtColor(shirt_img, cv2.COLOR_RGB2BGR))
                        debug_info['debug_shirt_saved'] = True
                    else:
                        debug_info['shirt_image_decode_failed'] = True
                except Exception as e:
                    debug_info['shirt_decode_error'] = str(e)
            else:
                return jsonify({"error": "No shirt image provided"}), 400

        if user_img is None or shirt_img is None:
            raise ValueError(f"Failed to load images: {debug_info}")

        # Process the images directly
        output_path = overlay_images_direct(user_img, shirt_img)
        debug_info['output_path'] = output_path
        print(f"Debug info: {debug_info}")

        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        print(f"Error in /tryon: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def bytes_to_cv_image(image_bytes):
    """Convert image bytes (base64) to OpenCV image with alpha channel support"""
    import base64
    
    # Decode the base64 string
    if isinstance(image_bytes, str) and image_bytes.startswith('data:image'):
        # Handle data URL format
        image_bytes = image_bytes.split(',')[1]
    
    if isinstance(image_bytes, str):
        image_bytes = base64.b64decode(image_bytes)
    
    # Convert to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image with transparency support
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)  # Changed to UNCHANGED to preserve alpha
    
    # Check if we got an image
    if img is None:
        print("Failed to decode image bytes")
        return None
        
    # Print shape to debug
    print(f"Decoded image shape: {img.shape}")
    
    # Handle transparency if present
    if img.shape[2] == 4:  # Image has alpha channel
        # Extract RGB and alpha
        rgb = img[:, :, :3]
        alpha = img[:, :, 3]
        
        # Create a white background
        white_bg = np.ones_like(rgb) * 255
        
        # Alpha blend with white background
        alpha_normalized = alpha[:, :, np.newaxis] / 255.0
        rgb_with_bg = (rgb * alpha_normalized + white_bg * (1 - alpha_normalized)).astype(np.uint8)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(rgb_with_bg, cv2.COLOR_BGR2RGB)
        
        # Store alpha channel for later use
        img.alpha_channel = alpha
    else:
        # For regular RGB images, just convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.alpha_channel = None
    
    return img

def overlay_images_direct(user_img, shirt_img):
    """
    Same functionality as overlay_images but works directly with image arrays
    instead of file paths
    """
    # Initialize result_img
    result_img = user_img.copy()
    
    # Get user body landmarks using MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    
    # Detect pose landmarks
    results = pose.process(user_img)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Get image dimensions
        h, w = user_img.shape[:2]
        
        # Extract key points for shirt placement
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        
        # Calculate shoulder width
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        
        # Find hip points to determine torso height
        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
        right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
        
        # Calculate torso height
        torso_height = abs((left_hip[1] + right_hip[1])//2 - (left_shoulder[1] + right_shoulder[1])//2)
        
        # Calculate neck position
        neck_y = min(left_shoulder[1], right_shoulder[1]) - int(shoulder_width * 0.25)
        
        # Use neck position for top alignment
        top_y = neck_y - int(shoulder_width * 0.1)
        top_x = left_shoulder[0] - int(shoulder_width * 0.2)
        
        # Process shirt image - extract it more carefully
        shirt_mask = extract_shirt_improved(shirt_img)
        
        # Debug: Check if shirt_mask is valid
        print(f"Shirt mask shape: {shirt_mask.shape if shirt_mask is not None else 'None'}")
        
        # Find shirt contours
        shirt_contours, _ = cv2.findContours(shirt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(shirt_contours) > 0:
            largest_contour = max(shirt_contours, key=cv2.contourArea)
            x_shirt, y_shirt, w_shirt, h_shirt = cv2.boundingRect(largest_contour)
            
            # Extract just the shirt portion with padding
            y_top_padded = max(0, y_shirt - int(h_shirt * 0.1))
            h_padded = min(shirt_img.shape[0] - y_top_padded, h_shirt + int(h_shirt * 0.1))
            
            shirt_only = shirt_img[y_top_padded:y_top_padded+h_padded, x_shirt:x_shirt+w_shirt]
            shirt_mask_only = shirt_mask[y_top_padded:y_top_padded+h_padded, x_shirt:x_shirt+w_shirt]
            
            # Debug: Check shirt_only and shirt_mask_only
            print(f"Shirt only shape: {shirt_only.shape}")
            print(f"Shirt mask only shape: {shirt_mask_only.shape}")
            
            # Calculate new width based on shoulders with some extra for sleeves
            new_width = int(shoulder_width * 1.4)
            
            # Maintain aspect ratio
            aspect_ratio = w_shirt / h_padded
            new_height = int(new_width / aspect_ratio)
            
            # Make sure height isn't too short or too long
            if new_height < torso_height * 0.8:
                new_height = int(torso_height * 0.8)
                new_width = int(new_height * aspect_ratio)
            elif new_height > torso_height * 1.2:
                new_height = int(torso_height * 1.2)
                new_width = int(new_height * aspect_ratio)
            
            # Resize shirt and mask
            resized_shirt = cv2.resize(shirt_only, (new_width, new_height))
            resized_mask = cv2.resize(shirt_mask_only, (new_width, new_height))
            
            # Debug: Check resized_shirt and resized_mask
            print(f"Resized shirt shape: {resized_shirt.shape}")
            print(f"Resized mask shape: {resized_mask.shape}")
            
            # Center the shirt horizontally
            center_x = (left_shoulder[0] + right_shoulder[0]) // 2
            start_x = center_x - new_width // 2
            
            # Overlay the shirt
            for i in range(new_height):
                for j in range(new_width):
                    y_pos = top_y + i
                    x_pos = start_x + j
                    
                    # Make sure we're within image bounds and mask is valid
                    if (0 <= y_pos < h and 0 <= x_pos < w and 
                        i < resized_mask.shape[0] and j < resized_mask.shape[1] and
                        resized_mask[i, j] > 10):
                        
                        # Get the shirt pixel color
                        shirt_pixel = resized_shirt[i, j]
                        
                        # Skip very bright/white pixels
                        is_white_or_very_light = (shirt_pixel[0] > 240 and 
                                                shirt_pixel[1] > 240 and 
                                                shirt_pixel[2] > 240)
                        
                        if not is_white_or_very_light:
                            # Apply alpha blending
                            alpha = min(1.0, resized_mask[i, j] / 255.0)
                            
                            # Skip pixels with low alpha
                            if alpha > 0.2:
                                result_img[y_pos, x_pos] = (
                                    alpha * shirt_pixel + 
                                    (1 - alpha) * result_img[y_pos, x_pos]
                                ).astype(np.uint8)
    else:
        print("Pose detection failed, using original image as result")
    
    # Save the output
    output_path = os.path.join(RESULT_FOLDER, "tryon_output.png")
    cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    
    return output_path


def extract_shirt_improved(image):
    """
    Improved shirt extraction that works with both regular and transparent images
    """
    # Check if the image has an alpha channel from previous processing
    has_alpha = hasattr(image, 'alpha_channel') and image.alpha_channel is not None
    
    if has_alpha:
        print("Using alpha channel for shirt mask")
        # Use the alpha channel directly as the mask
        shirt_mask = image.alpha_channel
        # Clean up the mask
        kernel = np.ones((5,5), np.uint8)
        shirt_mask = cv2.morphologyEx(shirt_mask, cv2.MORPH_CLOSE, kernel)
        return shirt_mask
    
    # Otherwise use the original method for non-transparent images
    # Convert to grayscale for better background detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create a mask for the likely background (black or very dark colors)
    _, background_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    
    # Invert to get the foreground (shirt)
    shirt_mask = cv2.bitwise_not(background_mask)
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    shirt_mask = cv2.morphologyEx(shirt_mask, cv2.MORPH_CLOSE, kernel)
    shirt_mask = cv2.morphologyEx(shirt_mask, cv2.MORPH_OPEN, kernel)
    
    # Fill holes in the mask
    contours, _ = cv2.findContours(shirt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour (should be the shirt)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a filled mask from the largest contour
        filled_mask = np.zeros_like(shirt_mask)
        cv2.drawContours(filled_mask, [largest_contour], 0, 255, -1)
        
        # Use the filled mask
        shirt_mask = filled_mask
    
    return shirt_mask

if __name__ == '__main__':
    app.run(debug=True)
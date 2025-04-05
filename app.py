from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS # Enables Cross-Origin Resource Sharing to allow requests from different domains.
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
            #Reads the image from the file path.
            user_img = cv2.imread(user_path)
            if user_img is not None:
                #Converts the image from BGR to RGB format (OpenCV uses BGR by default).
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
                #Converts BGR to RGB.
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
                
                #Converting Bytes to NumPy Array
                nparr = np.frombuffer(shirt_bytes, np.uint8)
                try:
                    #cv2.IMREAD_UNCHANGED → Preserves all channels (including alpha transparency if present).
                    decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    if decoded is not None:
                        # If shape[2] == 4, it means the image has 4 channels: Red (R), Green (G), Blue (B), and Alpha (A).  
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
    
@app.route('/tryon-pants', methods=['POST'])
def try_on_pants():
    try:
        user_img = None
        pants_img = None
        debug_info = {}  # For tracking what's happening

        # Process user image
        if 'user_image' in request.files:
            user_image = request.files['user_image']
            debug_info['user_source'] = 'file'
            debug_info['user_filename'] = user_image.filename
            
            user_path = os.path.join(UPLOAD_FOLDER, secure_filename(user_image.filename))
            user_image.save(user_path)
            # #Reads the image from the file path.
            user_img = cv2.imread(user_path)
            if user_img is not None:
                ##Converts the image from BGR to RGB format (OpenCV uses BGR by default).
                user_img = cv2.cvtColor(user_img, cv2.COLOR_BGR2RGB)
                debug_info['user_image_loaded'] = True
            else:
                debug_info['user_image_loaded'] = False

        # Process pants image
        if 'bottom_image' in request.files:
            pants_image = request.files['bottom_image']
            debug_info['pants_source'] = 'file'
            debug_info['pants_filename'] = pants_image.filename
            
            pants_path = os.path.join(UPLOAD_FOLDER, secure_filename(pants_image.filename))
            pants_image.save(pants_path)
            pants_img = cv2.imread(pants_path)
            if pants_img is not None:
                pants_img = cv2.cvtColor(pants_img, cv2.COLOR_BGR2RGB)
                debug_info['pants_image_loaded'] = True
            else:
                debug_info['pants_image_loaded'] = False
        else:
            return jsonify({"error": "No pants image provided"}), 400

        if user_img is None or pants_img is None:
            raise ValueError(f"Failed to load images: {debug_info}")

        # Process the images for pants overlay
        output_path = overlay_pants_on_user(user_img, pants_img)
        debug_info['output_path'] = output_path
        print(f"Debug info: {debug_info}")

        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        print(f"Error in /tryon-pants: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    


@app.route('/tryon-dress', methods=['POST'])
def try_on_dress():
    try:
        user_img = None
        dress_img = None
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

        # Process dress image
        if 'dress_image' in request.files:
            dress_image = request.files['dress_image']
            debug_info['dress_source'] = 'file'
            debug_info['dress_filename'] = dress_image.filename
            
            dress_path = os.path.join(UPLOAD_FOLDER, secure_filename(dress_image.filename))
            dress_image.save(dress_path)
            dress_img = cv2.imread(dress_path)
            if dress_img is not None:
                dress_img = cv2.cvtColor(dress_img, cv2.COLOR_BGR2RGB)
                debug_info['dress_image_loaded'] = True
            else:
                debug_info['dress_image_loaded'] = False
        else:
            # Handle the case where dress image is passed directly as bytes in multipart/form-data
            dress_image = request.files.get('dress_image')
            if dress_image:
                debug_info['dress_source'] = 'multipart_bytes'
                
                # Read the raw bytes
                dress_bytes = dress_image.read()
                debug_info['dress_bytes_length'] = len(dress_bytes)
                
                # Converting Bytes to NumPy Array
                nparr = np.frombuffer(dress_bytes, np.uint8)
                try:
                    decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    if decoded is not None:
                        if decoded.shape[2] == 4:  # Has alpha channel
                            # Extract RGB and store alpha separately
                            dress_img = cv2.cvtColor(decoded[:,:,:3], cv2.COLOR_BGR2RGB)
                            dress_img.alpha_channel = decoded[:,:,3]
                        else:
                            dress_img = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                        
                        debug_info['dress_image_loaded'] = True
                        
                        # Save for debugging
                        debug_path = os.path.join(UPLOAD_FOLDER, "debug_multipart_dress.png")
                        cv2.imwrite(debug_path, cv2.cvtColor(dress_img, cv2.COLOR_RGB2BGR))
                        debug_info['debug_dress_saved'] = True
                    else:
                        debug_info['dress_image_decode_failed'] = True
                except Exception as e:
                    debug_info['dress_decode_error'] = str(e)
            else:
                return jsonify({"error": "No dress image provided"}), 400

        if user_img is None or dress_img is None:
            raise ValueError(f"Failed to load images: {debug_info}")

        # Process the images for dress overlay
        output_path = overlay_dress_on_user(user_img, dress_img)
        debug_info['output_path'] = output_path
        print(f"Debug info: {debug_info}")

        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        print(f"Error in /tryon-dress: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def overlay_dress_on_user(user_img, dress_img):
    """
    Overlay dress onto the user image by detecting body landmarks and placing the dress correctly.
    Handles both the upper body (like shirts) and lower body (like pants) in one piece.
    """
    # Resize the user image to a specific dimension
    target_width = 400
    target_height = 600
    scale_factor = min(target_width / user_img.shape[1], target_height / user_img.shape[0])
    user_img = cv2.resize(user_img, None, fx=scale_factor, fy=scale_factor)

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
        
        # Extract key points for dress placement
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        
        # Calculate shoulder width
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        
        # Find hip points for lower part of dress
        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
        right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
        
        # Calculate hip width
        hip_width = abs(right_hip[0] - left_hip[0])
        
        # Calculate neck position for dress top alignment
        neck_y = min(left_shoulder[1], right_shoulder[1]) - int(shoulder_width * 0.25)
        
        # Use neck position for top alignment
        top_y = neck_y - int(shoulder_width * 0.1)
        
        # Calculate knee position to estimate dress length
        left_knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * w),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h))
        right_knee = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * w),
                      int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * h))
        
        # Calculate average knee position
        knee_y = (left_knee[1] + right_knee[1]) // 2
        
        # Calculate torso height (distance from shoulder to hip)
        torso_height = abs((left_hip[1] + right_hip[1]) // 2 - (left_shoulder[1] + right_shoulder[1]) // 2)
        
        # Calculate total dress height (from neck to below knees)
        dress_height = knee_y - top_y + int(torso_height * 0.1)  # Add a little extra below knees
        
        # Process dress image - extract it carefully
        dress_mask = extract_shirt_improved(dress_img)  # Reuse the existing function
        
        # Debug: Check if dress_mask is valid
        print(f"Dress mask shape: {dress_mask.shape if dress_mask is not None else 'None'}")
        
        # Find dress contours
        dress_contours, _ = cv2.findContours(dress_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(dress_contours) > 0:
            largest_contour = max(dress_contours, key=cv2.contourArea)
            x_dress, y_dress, w_dress, h_dress = cv2.boundingRect(largest_contour)
            
            # Extract just the dress portion with padding
            y_top_padded = max(0, y_dress - int(h_dress * 0.05))
            h_padded = min(dress_img.shape[0] - y_top_padded, h_dress + int(h_dress * 0.05))
            
            dress_only = dress_img[y_top_padded:y_top_padded+h_padded, x_dress:x_dress+w_dress]
            dress_mask_only = dress_mask[y_top_padded:y_top_padded+h_padded, x_dress:x_dress+w_dress]
            
            # Debug: Check dress_only and dress_mask_only
            print(f"Dress only shape: {dress_only.shape}")
            print(f"Dress mask only shape: {dress_mask_only.shape}")
            
            # Calculate new width for the dress based on shoulder and hip width
            # For dresses, we need to accommodate both shoulders and hips
            max_width = max(shoulder_width, hip_width)
            new_width = int(max_width * 0.5)  # Give extra room for sleeves and dress flow
            
            # Maintain aspect ratio while ensuring dress covers from neck to knees
            aspect_ratio = w_dress / h_padded
            calculated_height = int(new_width / aspect_ratio)
            
            # Ensure the dress fits the body properly
            if calculated_height < dress_height:
                new_height = dress_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_height = calculated_height
            
            # Resize dress and mask
            resized_dress = cv2.resize(dress_only, (new_width, new_height))
            resized_mask = cv2.resize(dress_mask_only, (new_width, new_height))
            
            # Debug: Check resized_dress and resized_mask
            print(f"Resized dress shape: {resized_dress.shape}")
            print(f"Resized mask shape: {resized_mask.shape}")
            
            # Center the dress horizontally
            center_x = (left_shoulder[0] + right_shoulder[0]) // 2
            start_x = center_x - new_width // 2
            
            # Overlay the dress
            for i in range(new_height):
                for j in range(new_width):
                    y_pos = top_y + i
                    x_pos = start_x + j
                    
                    # Make sure we're within image bounds and mask is valid
                    if (0 <= y_pos < h and 0 <= x_pos < w and 
                        i < resized_mask.shape[0] and j < resized_mask.shape[1] and
                        resized_mask[i, j] > 10):
                        
                        # Get the dress pixel color
                        dress_pixel = resized_dress[i, j]
                        
                        # Apply alpha blending for all colors (including black)
                        alpha = min(1.0, resized_mask[i, j] / 255.0)
                        
                        # Skip pixels with low alpha
                        if alpha > 0.2:
                            result_img[y_pos, x_pos] = (
                                alpha * dress_pixel + 
                                (1 - alpha) * result_img[y_pos, x_pos]
                            ).astype(np.uint8)
    else:
        print("Pose detection failed, using original image as result")
    
    # Save the output
    output_path = os.path.join(RESULT_FOLDER, "tryon_dress_output.png")
    cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    
    return output_path


def bytes_to_cv_image(image_bytes):
    """Convert image bytes (base64) to OpenCV image with alpha channel support"""
    import base64
    
    # Decode the base64 string
    if isinstance(image_bytes, str) and image_bytes.startswith('data:image'):
        #Extracting Base64 String from Data URL
        image_bytes = image_bytes.split(',')[1]
    
    #Decoding the Base64 String to Bytes
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
        # #Converts the image from BGR to RGB format (OpenCV uses BGR by default).
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.alpha_channel = None
    
    return img

def overlay_images_direct(user_img, shirt_img):
    """
    Overlay shirt onto the user image.
    Ensures black pixels are included in the overlay.
    """
    target_width = 400
    target_height = 600
    scale_factor = min(target_width / user_img.shape[1], target_height / user_img.shape[0])
    user_img = cv2.resize(user_img, None, fx=scale_factor, fy=scale_factor)



    # Initialize result_img
    result_img = user_img.copy()
    
    # Requires at least 50% confidence to detect landmarks.
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    
    # Detect pose landmarks
    #Results contain body landmarks like shoulders, hips, etc.
    results = pose.process(user_img)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Get image dimensions
        #height and width of the user image
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
       # Increase torso height by 20% to extend the shirt downward
        torso_height = int(abs((left_hip[1] + right_hip[1]) // 2 - (left_shoulder[1] + right_shoulder[1]) // 2) * 1.5)
        
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
            
            shirt_only = shirt_img[y_top_padded:y_top_padded + h_padded, x_shirt:x_shirt + w_shirt]
            shirt_mask_only = shirt_mask[y_top_padded:y_top_padded + h_padded, x_shirt:x_shirt + w_shirt]
            
            # Debug: Check shirt_only and shirt_mask_only
            print(f"Shirt only shape: {shirt_only.shape}")
            print(f"Shirt mask only shape: {shirt_mask_only.shape}")

            #mask of the shirt is  a binary image or grayscale image where:
            
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
                        
                        # Apply alpha blending for all colors (including black)
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

def overlay_pants_on_user(user_img, pants_img):
    """
    Overlay pants onto the user image by detecting hips and placing the pants correctly.
    """
    # Resize the user image to a specific dimension (e.g., 600x800)
    target_width = 400
    target_height = 600
    scale_factor = min(target_width / user_img.shape[1], target_height / user_img.shape[0])
    user_img = cv2.resize(user_img, None, fx=scale_factor, fy=scale_factor)


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
        
        # Extract key points for pants placement (hips)
        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
        right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))
        
        # Calculate hip width
        hip_width = abs(right_hip[0] - left_hip[0])
        
        # Calculate the vertical position for the top of the pants
        pants_top_y = (left_hip[1] + right_hip[1]) // 2  # Use the average y-coordinate of the hips
        
        # Calculate the distance from hips to ankles
        left_ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * w),
                      int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h))
        right_ankle = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w),
                       int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h))
        
        # Calculate the vertical distance from hips to ankles
        leg_height = abs((left_ankle[1] + right_ankle[1]) // 2 - pants_top_y)
        
        # Adjust the pants_top_y to move the pants higher
        pants_top_y -= int(leg_height * 0.17)  # Adjust this factor as needed
        
        # Process pants image - extract it more carefully
        pants_mask = extract_shirt_improved(pants_img)  # Reuse the shirt mask extraction logic
        
        # Debug: Check if pants_mask is valid
        print(f"Pants mask shape: {pants_mask.shape if pants_mask is not None else 'None'}")
        
        # Find pants contours
        pants_contours, _ = cv2.findContours(pants_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(pants_contours) > 0:
            largest_contour = max(pants_contours, key=cv2.contourArea)
            x_pants, y_pants, w_pants, h_pants = cv2.boundingRect(largest_contour)
            
            # Extract just the pants portion with padding
            y_top_padded = max(0, y_pants - int(h_pants * 0.1))
            h_padded = min(pants_img.shape[0] - y_top_padded, h_pants + int(h_pants * 0.1))
            
            pants_only = pants_img[y_top_padded:y_top_padded+h_padded, x_pants:x_pants+w_pants]
            pants_mask_only = pants_mask[y_top_padded:y_top_padded+h_padded, x_pants:x_pants+w_pants]
            
            # Debug: Check pants_only and pants_mask_only
            print(f"Pants only shape: {pants_only.shape}")
            print(f"Pants mask only shape: {pants_mask_only.shape}")
            
            # Calculate new width based on hips with some extra for fit
            new_width = int(hip_width * 2.0)  # Adjust this factor as needed
            
            # Maintain aspect ratio
            aspect_ratio = w_pants / h_padded
            new_height = int(new_width / aspect_ratio)
            
            # Ensure the pants height is proportional to the leg height
            if new_height < leg_height * 0.85:  # Adjust this factor as needed
                new_height = int(leg_height * 0.85)
                new_width = int(new_height * aspect_ratio)
            elif new_height > leg_height * 0.9:
               new_height = int(leg_height * 1.28)
            
            # Resize pants and mask
            resized_pants = cv2.resize(pants_only, (new_width, new_height))
            resized_mask = cv2.resize(pants_mask_only, (new_width, new_height))
            
            # Debug: Check resized_pants and resized_mask
            print(f"Resized pants shape: {resized_pants.shape}")
            print(f"Resized mask shape: {resized_mask.shape}")
            
            # Center the pants horizontally
            center_x = (left_hip[0] + right_hip[0]) // 2
            start_x = center_x - new_width // 2
            
            # Overlay the pants
            for i in range(new_height):
                for j in range(new_width):
                    y_pos = pants_top_y + i
                    x_pos = start_x + j
                    
                    # Make sure we're within image bounds and mask is valid
                    if (0 <= y_pos < h and 0 <= x_pos < w and 
                        i < resized_mask.shape[0] and j < resized_mask.shape[1] and
                        resized_mask[i, j] > 10):
                        
                        # Get the pants pixel color
                        pants_pixel = resized_pants[i, j]
                        
                        # Apply alpha blending for all colors (including black)
                        alpha = min(1.0, resized_mask[i, j] / 255.0)
                        
                        # Skip pixels with low alpha
                        if alpha > 0.2:
                            result_img[y_pos, x_pos] = (
                                alpha * pants_pixel + 
                                (1 - alpha) * result_img[y_pos, x_pos]
                            ).astype(np.uint8)
    else:
        print("Pose detection failed, using original image as result")
    
    # Save the output
    output_path = os.path.join(RESULT_FOLDER, "tryon_pants_output.png")
    cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    
    return output_path


def extract_shirt_improved(image):
    """
    Improved shirt extraction that works with both regular and transparent images.
    Ensures black pixels are included in the mask.
    """
    # Check if the image has an alpha channel from previous processing
    has_alpha = hasattr(image, 'alpha_channel') and image.alpha_channel is not None
    
    if has_alpha:
        print("Using alpha channel for shirt mask")
        # Use the alpha channel directly as the mask
        shirt_mask = image.alpha_channel
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        shirt_mask = cv2.morphologyEx(shirt_mask, cv2.MORPH_CLOSE, kernel)
    else:
        # Convert to grayscale for better background detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create a mask for the likely background (black or very dark colors)
        _, background_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        
        # Invert to get the foreground (shirt)
        shirt_mask = cv2.bitwise_not(background_mask)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
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
    app.run(host='0.0.0.0', port=5000, debug=True)
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import numpy as np
import cv2
import webbrowser
from PIL import Image
import base64
from io import BytesIO
import json
import os
import re
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import socket
import ssl
import threading

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', '2b1f9b07bd7905d8b029a1ecdaa4dbee')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# Serve assets folder
from flask import send_from_directory
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('assets', filename)

# Database Configuration
# Supporting multiple common env var names for database connections
# FALLBACK: Using hardcoded Atlas URI as requested for immediate deployment
mongodb_uri = os.getenv('MONGODB_URI') or os.getenv('MONGO_URL') or os.getenv('DATABASE_URL') or 'mongodb+srv://Radhe:Radhe2059@cluster0.flk4ry8.mongodb.net/?appName=Cluster0'

if not mongodb_uri or "localhost" in mongodb_uri:
    print("WARNING: Using local database fallback.")
    mongodb_uri = 'mongodb://localhost:27017/MOODIFY'
else:
    # Log connection attempt (hiding credentials for security)
    masked_uri = mongodb_uri.split('@')[-1] if '@' in mongodb_uri else "Remote URI provided"
    print(f"INFO: Connecting to MongoDB at {masked_uri}")

client = MongoClient(mongodb_uri)
try:
    # Try getting default database from URI (e.g. Atlas string with /dbname)
    db = client.get_default_database()
except Exception:
    # Fallback to 'moodify_db' if no database is defined in the URI
    db = client['moodify_db']

users_collection = db['users']

# SMTP Settings (Hardcoded fallbacks for Railway compatibility)
SMTP_SENDER_EMAIL = os.getenv('SMTP_SENDER_EMAIL', 'radheshyamt028@gmail.com')
SMTP_SENDER_PASSWORD = os.getenv('SMTP_SENDER_PASSWORD', 'qekzkhrqplhvydlt')
SMTP_RECEIVER_EMAIL = os.getenv('SMTP_RECEIVER_EMAIL', 'radheshyamt028@gmail.com')

def send_email_optimized(receiver_email, subject, body):
    """
    Spawns a background thread to send an email.
    Prevents the main web worker from hanging or timing out.
    """
    thread = threading.Thread(target=_send_email_async_task, args=(receiver_email, subject, body))
    thread.daemon = True # Ensure thread doesn't block app exit
    thread.start()
    return True

def _send_email_async_task(receiver_email, subject, body):
    """
    The actual SMTP connection logic run in a background thread.
    Includes longer timeout and fallback ports.
    """
    sender_email = SMTP_SENDER_EMAIL
    sender_password = SMTP_SENDER_PASSWORD
    
    if not sender_email or not sender_password:
        print("Error: SMTP credentials not configured.")
        return
        
    msg = MIMEMultipart()
    msg['From'] = f"Moodify <{sender_email}>"
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    # Try Port 587 (TLS) first and then 465 (SSL)
    try:
        print(f"DEBUG [Background]: Attempting connection to Port 587 for {receiver_email}...")
        with smtplib.SMTP('smtp.gmail.com', 587, timeout=30) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"DEBUG [Background]: Email sent successfully via Port 587 to {receiver_email}.")
    except Exception as e_tls:
        print(f"DEBUG [Background]: Port 587 failed: {e_tls}. Trying Port 465 fallback...")
        try:
            print(f"DEBUG [Background]: Attempting connection to Port 465 for {receiver_email}...")
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=30, context=context) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
            print(f"DEBUG [Background]: Email sent successfully via Port 465 to {receiver_email}.")
        except Exception as e_ssl:
            print(f"DEBUG [Background] CRITICAL: Both ports failed for {receiver_email}: {e_ssl}")
            # Optionally log to a file for persistence
            with open("email_error.log", "a") as f:
                f.write(f"{datetime.utcnow().isoformat()} - Error sending to {receiver_email}: {e_ssl}\n")

# Authentication Setup
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

from datetime import datetime

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.email = user_data['email']
        self.name = user_data['name']
        self.scans = user_data.get('scans', 0)
        self.preferences = user_data.get('preferences', {})
        self.profile_pic = user_data.get('profile_pic', None)
        # created_at stored as ISO string in DB on signup; parse year when needed
        self.created_at = user_data.get('created_at', None)

@login_manager.user_loader
def load_user(user_id):
    from bson.objectid import ObjectId
    try:
        user_data = users_collection.find_one({"_id": ObjectId(user_id)})
        if user_data:
            return User(user_data)
    except:
        pass
    return None

info = {}

haarcascade = "Models/haarcascade_frontalface_default.xml"
label_map = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
print("+"*50, "setup cascades")
model = None

def get_model():
	global model
	if model is None:
		print("Lazy loading model...")
		import keras
		from keras.models import load_model
		from keras.layers import Dense

		# Monkeypatch Dense to ignore quantization_config (Keras 3 compatibility)
		original_dense_from_config = Dense.from_config
		@classmethod
		def patched_from_config(cls, config):
			if 'quantization_config' in config:
				config.pop('quantization_config')
			return original_dense_from_config(config)
		Dense.from_config = patched_from_config

		model = load_model('Models/model_new.h5', compile=False)
	return model

cascade = cv2.CascadeClassifier(haarcascade)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/mood')
def mood():
    return render_template('mood.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        user_name = request.form.get('name')
        user_email = request.form.get('email')
        user_message = request.form.get('message')
        
        print(f"New Contact Message from {user_name} ({user_email}): {user_message}")
        
        if SMTP_SENDER_EMAIL and SMTP_SENDER_PASSWORD and SMTP_RECEIVER_EMAIL:
            try:
                msg = MIMEMultipart()
                msg['From'] = SMTP_SENDER_EMAIL
                msg['To'] = SMTP_RECEIVER_EMAIL # You receive the email
                msg['Subject'] = f"New Moodify Inquiry from {user_name}"
                
                # Add Reply-To so you can just click 'Reply' in your inbox
                msg.add_header('reply-to', user_email)
                
                body = f"You have received a new message from your website contact form.\n\n" \
                       f"Name: {user_name}\n" \
                       f"Email: {user_email}\n\n" \
                       f"Message:\n{user_message}"
                
                send_email_optimized(SMTP_RECEIVER_EMAIL, f"New Moodify Inquiry from {user_name}", body)
                flash("Message sent successfully! We'll get back to you soon.", "success")
            except Exception as e:
                print(f"Email Error: {e}")
                flash("Oops! Something went wrong while sending the message.", "error")
        else:
            print("Warning: SMTP not configured in .env. Email not sent.")
            # flash("Message received (Demo Mode). Configure your SMTP settings in .env to receive emails.", "success")
            
        return redirect(url_for('contact'))
        
    return render_template('contact.html')

# Authentication Routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            
            if not name or not email or not password:
                flash("All fields are required", "error")
                return render_template('signup.html')
            
            if len(password) < 6:
                flash("Password must be at least 6 characters", "error")
                return render_template('signup.html')
            
            # Check if email already exists (case-insensitive)
            existing_user = users_collection.find_one({"email": {"$regex": f"^{re.escape(email)}$", "$options": "i"}})
            
            if existing_user:
                flash("Email already exists! Please login or use a different email.", "error")
                return render_template('signup.html')
                flash("Email already exists!", "error")
                return redirect(url_for('signup'))
                
            hashed_password = bcrypt.generate_password_hash(password.encode('utf-8')).decode('utf-8')
            users_collection.insert_one({
                "name": name,
                "email": email,
                "password": hashed_password,
                "scans": 0,
                "preferences": {},
                "created_at": datetime.utcnow().isoformat()
            })
            flash("Account created! Please login.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash("MongoDB is not running. Please start MongoDB and try again. Error: " + str(e), "error")
            return redirect(url_for('signup'))
        
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            remember = True if request.form.get('remember') else False
            
            print(f"DEBUG: Attempting login with email: '{email}'")
            print(f"DEBUG: Password length: {len(password)}")
            
            if not email or not password:
                flash("Please enter email and password", "error")
                return render_template('login.html')
            
            # Debug: Print all emails in database
            all_users = list(users_collection.find({}, {"email": 1}))
            print(f"DEBUG: All users in DB: {all_users}")
            
            # Search for user with exact lowercase match first
            user_data = users_collection.find_one({"email": email})
            print(f"DEBUG: Found user with exact match: {user_data is not None}")
            
            # If not found, try case-insensitive
            if not user_data:
                user_data = users_collection.find_one({"email": {"$regex": f"^{re.escape(email)}$", "$options": "i"}})
                print(f"DEBUG: Found user with case-insensitive match: {user_data is not None}")
            
            if user_data:
                print(f"DEBUG: User found: {user_data.get('email')}")
                if bcrypt.check_password_hash(user_data['password'], password.encode('utf-8')):
                    user_obj = User(user_data)
                    login_user(user_obj, remember=remember)
                    flash(f"Welcome back, {user_obj.name}!", "success")
                    # Pass username to template for success modal
                    return render_template('login.html', login_success=True, username=user_obj.name)
                else:
                    flash("Invalid password", "error")
            else:
                print(f"DEBUG: No user found for email: {email}")
                flash("Email not found. Please create an account first.", "error")
        except Exception as e:
            print(f"DEBUG: Login error: {str(e)}")
            flash(f"Login error: {str(e)}", "error")
            
    return render_template('login.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            print(f"DEBUG: Forgot password attempt for email: {email}")
            user_data = users_collection.find_one({"email": email})
            
            if user_data:
                # Generate token
                from itsdangerous import URLSafeTimedSerializer
                s = URLSafeTimedSerializer(app.secret_key)
                token = s.dumps(email, salt='email-confirm')
                
                # Create reset link
                link = url_for('reset_password', token=token, _external=True)
                print(f"DEBUG: Password reset link generated: {link}")
                
                # Send Email
                if SMTP_SENDER_EMAIL and SMTP_SENDER_PASSWORD:
                    try:
                        msg = MIMEMultipart()
                        msg['From'] = SMTP_SENDER_EMAIL
                        msg['To'] = email
                        msg['Subject'] = "Moodify Password Reset"
                        
                        body = f"Click the link below to reset your password:\n{link}\n\nIf you did not request this, please ignore this email."
                        
                        send_email_optimized(email, "Moodify Password Reset", body)
                        flash(f"Password reset link sent to {email}.", "success")
                        print("DEBUG: Reset email sent successfully.")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"Email Error in /forgot_password: {e}")
                        flash("Error sending email. Please try again later.", "error")
                else:
                    print(f"DEBUG Simulation: Reset Link: {link}") 
                    flash("Reset link sent (Demo Mode: check console)", "success")
            else:
                flash("Email not found.", "error")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"DEBUG: /forgot_password generic error: {e}")
            flash("Internal Server Error occurred. Our team has been notified.", "error")
            
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        from itsdangerous import URLSafeTimedSerializer, SignatureExpired
        s = URLSafeTimedSerializer(app.secret_key)
        
        try:
            email = s.loads(token, salt='email-confirm', max_age=3600) # 1 hour expiration
        except SignatureExpired:
            flash("The reset link has expired.", "error")
            return redirect(url_for('forgot_password'))
        except Exception as e:
            print(f"DEBUG: Token validation error: {e}")
            flash("Invalid reset link.", "error")
            return redirect(url_for('forgot_password'))
            
        if request.method == 'POST':
            password = request.form.get('password')
            # Ensure proper encoding for bcrypt
            hashed_password = bcrypt.generate_password_hash(password.encode('utf-8')).decode('utf-8')
            
            users_collection.update_one({"email": email}, {"$set": {"password": hashed_password}})
            flash("Your password has been updated! Please login.", "success")
            return redirect(url_for('login'))
            
        return render_template('reset_password.html', token=token)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"DEBUG: /reset_password error: {e}")
        return render_template('reset_password.html', token=token, error=str(e))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    from bson.objectid import ObjectId
    user_id = current_user.id
    
    if request.method == 'POST':
        # Check if this is a profile pic update or data update
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file and file.filename != '':
                # Ensure uploads directory exists
                upload_folder = os.path.join('static', 'uploads')
                if not os.path.exists(upload_folder):
                    os.makedirs(upload_folder)
                
                # Secure filename and save
                # Simple unique filename strategy
                import time
                filename = f"user_{user_id}_{int(time.time())}.{file.filename.split('.')[-1]}"
                file.save(os.path.join(upload_folder, filename))
                
                # Update DB
                users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": {"profile_pic": filename}})
                current_user.profile_pic = filename
                flash('Profile picture updated!', 'success')
                return redirect(url_for('profile'))

        # Update Profile Data Logic
        new_name = request.form.get('name')
        
        if new_name:
            # Determine preferences from multiple potential inputs
            fav_language = request.form.get('language')
            fav_singer = request.form.get('singer')
            fav_singer_other = request.form.get('singer_other')
            
            # Prepare update data (only name and preferences, NOT email)
            update_data = {
                'name': new_name
            }
            
            # Update preferences if provided
            if fav_language:
                update_data['preferences.language'] = fav_language
                
            # Prioritize manual entry, then radio selection
            if fav_singer_other and fav_singer_other.strip():
                update_data['preferences.singer'] = fav_singer_other.strip()
            elif fav_singer:
                update_data['preferences.singer'] = fav_singer
                
            users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": update_data})
            flash('Profile updated successfully!', 'success')
        
        return redirect(url_for('profile'))

    # Fetch fresh user data from DB and convert to User object
    user_data = users_collection.find_one({"_id": ObjectId(user_id)})
    if user_data:
        user_obj = User(user_data)
        # compute fallback current year
        current_year = datetime.utcnow().year
        return render_template('profile.html', user=user_obj, current_year=current_year)
    else:
        flash("User not found", "error")
        return redirect(url_for('logout'))

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    from bson.objectid import ObjectId
    import json
    
    try:
        data = request.get_json()
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        user_id = current_user.id
        
        # Fetch user from DB
        user_data = users_collection.find_one({"_id": ObjectId(user_id)})
        
        if not user_data:
            return jsonify({'success': False, 'message': 'User not found'})
        
        # Verify current password
        if not bcrypt.check_password_hash(user_data['password'], current_password.encode('utf-8')):
            return jsonify({'success': False, 'message': 'Current password is incorrect'})
        
        # Hash new password
        hashed_new_password = bcrypt.generate_password_hash(new_password.encode('utf-8')).decode('utf-8')
        
        # Update password in DB
        users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"password": hashed_new_password}}
        )
        
        return jsonify({'success': True, 'message': 'Password changed successfully'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/delete_account', methods=['POST'])
@login_required
def delete_account():
    from bson.objectid import ObjectId
    import shutil
    
    try:
        user_id = current_user.id
        
        # Fetch user to get profile pic
        user_data = users_collection.find_one({"_id": ObjectId(user_id)})
        
        if not user_data:
            return jsonify({'success': False, 'message': 'User not found'})
        
        # Delete profile picture if exists
        if user_data.get('profile_pic'):
            pic_path = os.path.join('static', 'uploads', user_data['profile_pic'])
            if os.path.exists(pic_path):
                try:
                    os.remove(pic_path)
                except:
                    pass
        
        # Delete user from database
        users_collection.delete_one({"_id": ObjectId(user_id)})
        
        # Logout user
        logout_user()
        
        return jsonify({'success': True, 'message': 'Account deleted successfully'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return render_template('logout.html')

@app.route('/choose_singer', methods = ["POST"])
def choose_singer():
	info['language'] = request.form['language']
	print(info)
	return render_template('choose_singer.html', data = info['language'])


@app.route('/emotion_detect', methods=["GET", "POST"])
def emotion_detect():
	if request.method == 'POST':
		singer = request.form.get('singer')
		singer_other = request.form.get('singer_other')
		
		# Use 'singer_other' if it has value, otherwise use 'singer'
		if singer_other and singer_other.strip():
			info['singer'] = singer_other
		else:
			info['singer'] = singer
			
	# For GET requests, use previously stored singer info
	singer = info.get('singer', 'Unknown')
	language = info.get('language', 'english')
	return render_template("emotion_detect.html", singer=singer, language=language)

@app.route('/process_emotion', methods=["POST"])
def process_emotion():
	try:
		# Get data from client
		data = request.get_json()
		
		# Handle direct emotion selection from emoji
		if data.get('isEmoji') and data.get('emotion'):
			emotion = data.get('emotion')
			info['emotion'] = emotion
			
			if current_user.is_authenticated:
				from bson.objectid import ObjectId
				users_collection.update_one(
					{"_id": ObjectId(current_user.id)},
					{"$inc": {"scans": 1}}
				)
			
			return jsonify({
				'success': True,
				'emotion': emotion,
				'confidence': 1.0
			})
		
		# Handle image-based face detection
		image_data = data.get('image')
		
		if image_data:
			# Decode base64 image (handle possible missing prefix)
			try:
				if ',' in image_data:
					b64 = image_data.split(',', 1)[1]
				else:
					b64 = image_data
				image_bytes = base64.b64decode(b64)
				image = Image.open(BytesIO(image_bytes)).convert('RGB')
			except Exception as e:
				print('Image decode error in /process_emotion:', e)
				return jsonify({'success': False, 'message': 'Unsupported image format. Please upload JPG, PNG, GIF, WebP, BMP, or TIFF images.'})
			
			# Convert to OpenCV format
			frm = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
			gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
			
			# Improve contrast for better face detection
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
			gray_enhanced = clahe.apply(gray)
			
			# Multiple scale factors for better face detection
			faces = cascade.detectMultiScale(gray_enhanced, 1.05, 5, minSize=(50,50), maxSize=(300,300))
			
			if len(faces) == 0:
				# Try with different parameters if first attempt fails
				faces = cascade.detectMultiScale(gray_enhanced, 1.1, 4, minSize=(40,40))
			
			if len(faces) > 0:
				# Select the largest face (most likely the main subject)
				face_areas = [w*h for (x, y, w, h) in faces]
				largest_face_idx = np.argmax(face_areas)
				x, y, w, h = faces[largest_face_idx]
				
				# Extract face region with slight padding
				padding = int(w * 0.1)
				x = max(0, x - padding)
				y = max(0, y - padding)
				w = min(gray.shape[1] - x, w + 2*padding)
				h = min(gray.shape[0] - y, h + 2*padding)
				
				roi = gray[y:y+h, x:x+w]

				# Verify presence of eyes inside detected face region to avoid palms/false positives
				try:
					eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
					# Detect eyes in the ROI (smaller sizes)
					eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
					if len(eyes) == 0:
						# No eyes found — likely a hand or occlusion; treat as no face detected
						return jsonify({
							'success': False,
							'message': 'No face detected. Please remove any obstruction (hand/fingers) from the camera and try again.'
						})
				except Exception as e:
					# If eye detector fails for any reason, continue but log error
					print('Eye detection error:', e)

				# Save the processed face for display
				face_img = frm[y:y+h, x:x+w] # Use the color frame for saving
				cv2.imwrite("static/face.jpg", face_img)

				# Increment scan count for authenticated users
				if current_user.is_authenticated:
				    from bson.objectid import ObjectId
				    users_collection.update_one(
				        {"_id": ObjectId(current_user.id)},
				        {"$inc": {"scans": 1}}
				    )

				# Preprocess for model input
				roi_resized = cv2.resize(roi, (48, 48))

				# Normalize pixel values
				roi_normalized = roi_resized.astype(np.float32) / 255.0
				roi_input = np.reshape(roi_normalized, (1, 48, 48, 1))
				# Get prediction with confidence scores for all emotions
				print("[DEBUG] Calling model.predict...")
				current_model = get_model()
				prediction = current_model.predict(roi_input, verbose=0)
				emotion_idx = np.argmax(prediction)
				print(f"[DEBUG] Prediction done. Emotion idx: {emotion_idx}")
				confidence = prediction[0][emotion_idx]
				emotion = label_map[emotion_idx]
				
				# Store all emotion predictions
				all_emotions = {}
				for idx, label in enumerate(label_map):
					all_emotions[label] = float(prediction[0][idx])
				
				info['emotion'] = emotion
				info['all_emotions'] = all_emotions
				info['confidence'] = float(confidence)
				
				return jsonify({
					'success': True,
					'emotion': emotion,
					'confidence': float(confidence),
					'all_emotions': all_emotions
				})
			else:
				return jsonify({
					'success': False,
					'message': 'No face detected. Please ensure good lighting and face the camera directly.'
				})
		else:
			return jsonify({
				'success': False,
				'message': 'No image data received.'
			})
	except Exception as e:
		import traceback
		traceback.print_exc()
		print(f"Error in emotion detection: {str(e)}")
		# Return a more descriptive error to the client so UI can show helpful message
		return jsonify({
			'success': False,
			'message': 'Error processing image: ' + str(e)
		})

@app.route('/check_face', methods=['POST'])
def check_face():
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data received.'})

        # handle optional data url prefix
        try:
            if ',' in image_data:
                image_data_b64 = image_data.split(',', 1)[1]
            else:
                image_data_b64 = image_data
            image_bytes = base64.b64decode(image_data_b64)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            print('Check face decode error:', e)
            return jsonify({'success': False, 'message': 'Invalid image data.'})

        frm = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        # quick enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_enhanced = clahe.apply(gray)

        faces = cascade.detectMultiScale(gray_enhanced, 1.05, 5, minSize=(50,50), maxSize=(300,300))
        if len(faces) == 0:
            faces = cascade.detectMultiScale(gray_enhanced, 1.1, 4, minSize=(40,40))

        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected'})

        # verify eyes
        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]
        try:
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(10,10))
            if len(eyes) == 0:
                return jsonify({'success': False, 'message': 'Face appears occluded'})
        except Exception:
            pass

        return jsonify({'success': True, 'message': 'Face present'})
    except Exception as e:
        print('Check face error:', e)
        return jsonify({'success': False, 'message': 'Error checking face'})


@app.route('/emotion_result')
def emotion_result():
	emotion = info.get('emotion', 'Unknown')
	singer = info.get('singer', 'Unknown')
	language = info.get('language', 'english')
	
	# Emotion emojis
	emotion_emojis = {
		'Angry': '😠',
		'Neutral': '😐',
		'Fear': '😨',
		'Happy': '😊',
		'Sad': '😢',
		'Surprise': '😮'
	}
	
	emoji = emotion_emojis.get(emotion, '🎵')
	
	return render_template('emotion_result.html', 
		emotion=emotion, 
		singer=singer, 
		language=language,
		emoji=emoji
	)

@app.route('/admin/backfill_created_at', methods=['POST','GET'])
def backfill_created_at():
	# Safety: only allow in debug mode or from localhost
	if not app.debug:
		return jsonify({'success': False, 'message': 'Not allowed in production'}), 403
	from datetime import datetime
	# Set created_at for documents missing the field
	res1 = users_collection.update_many({ 'created_at': { '$exists': False } }, { '$set': { 'created_at': datetime.utcnow().isoformat() } })
	# Also handle explicit null values
	res2 = users_collection.update_many({ 'created_at': None }, { '$set': { 'created_at': datetime.utcnow().isoformat() } })
	return jsonify({
		'success': True,
		'matched_missing': int(res1.matched_count),
		'modified_missing': int(res1.modified_count),
		'matched_null': int(res2.matched_count),
		'modified_null': int(res2.modified_count)
	})

if __name__ == "__main__":
	app.run(debug=True, use_reloader=True, threaded=False)
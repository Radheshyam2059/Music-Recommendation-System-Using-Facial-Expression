# 🎵 MOODIFY - AI Music Recommendation System

Moodify is a professional, premium web application that uses advanced facial recognition to detect your emotions and suggest the perfect musical soundtrack for your vibe.

## ✨ Key Features

- **Real-time Emotion Detection**: Scan your face via webcam for instant mood analysis.
- **Image Upload**: Don't have a camera? Upload any photo and our AI will decode your expression.
- **Curated Personalization**: Choose your preferred language and favorite artists for tailored results.
- **Multi-Platform Search**: Instantly find tracks on YouTube and Spotify.
- **Modern Aesthetic**: Glassmorphism design with a responsive, premium user experience.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A webcam (for live detection)


1. **Set up a virtual environment (Recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 🏃 Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the web app**:
   Open your browser and navigate to `http://localhost:5000`

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Computer Vision**: OpenCV
- **Deep Learning**: Keras / TensorFlow
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), JavaScript
- **Icons**: Font Awesome 6

## 📂 Project Structure

- `app.py`: Main application logic and routes.
- `Models/`: Contains the pre-trained emotion detection model and Haar cascades.
- `templates/`: Professional HTML templates extending a base layout.
- `static/`: Modern design system and assets.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Built with ❤️ for Music Lovers.

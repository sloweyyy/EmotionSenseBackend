import pickle
import pandas as pd
import cv2
import numpy as np
from skimage import feature
from PIL import Image
import dlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the pre-trained model
pkl_filename = "Model_SVM.h5"
with open(pkl_filename, 'rb') as f_in:
    model = pickle.load(f_in)

# Initialize the dlib face detector
detector = dlib.get_frontal_face_detector()

emotion_labels = {
    0: "Angry",
    1: "Fear",
    2: "Happy",
    3: "Neutral",
    4: "Sad",
}

def preprocess_image(image):
    """Convert image to grayscale, resize, and compute HOG features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    hog_features = feature.hog(resized, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), block_norm='L2-Hys',
                               visualize=False, transform_sqrt=True)
    return hog_features

def predict(image):
    """Detect faces, draw rectangles, preprocess image, and predict emotion."""
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    if faces:
        face = faces[0]  # Focus on the first detected face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_region = image[y:y+h, x:x+w]
        features = preprocess_image(face_region)
        features = features.reshape(1, -1)

        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        prediction = model.predict(features_scaled)
        predicted_emotion = emotion_labels[int(prediction[0])]

        # Preparing the face image for display
        output_face_image = Image.fromarray(face_region)  # Convert NumPy array to PIL Image for Gradio
    else:
        predicted_emotion = "No face detected"
        output_face_image = Image.fromarray(image)  # Return the original image if no face detected

    return predicted_emotion, output_face_image

# Example usage:
if __name__ == "__main__":
    from PIL import Image
    import requests
    from io import BytesIO

    # Load an example image from the internet
    url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ0e0W0z9LGb3MT8z6GZw7zqTM9c_6st7U3EA&s"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    emotion, face_img = predict(img)
    print(f"Predicted Emotion: {emotion}")
    face_img.show()

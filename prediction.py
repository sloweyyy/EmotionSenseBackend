import pickle
import cv2
import numpy as np
from skimage import feature
from PIL import Image
import dlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from io import BytesIO
import requests

# URLs của model trên Google Drive
pkl_filename_with_hog = "https://drive.google.com/uc?id=1s0Go_6NFt3RCe2Hva_XskJGCFsNkh-OG"
pkl_filename_without_hog = "https://drive.google.com/uc?id=1i0BLhmVCOl2BYjdFGFqlbEfynY0AhhLo"

# Tải model từ Google Drive
def load_model(url):
    response = requests.get(url)
    model = pickle.loads(response.content)
    return model

# Load models
model_with_hog = load_model(pkl_filename_with_hog)
model_without_hog = load_model(pkl_filename_without_hog)

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

def resize_image(image, target_size=(300, 300)):
    """Resize image to the target size."""
    return image.resize(target_size)

def predict_with_hog(image):
    """Detect faces, draw rectangles, preprocess image, and predict emotion."""
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    if faces:
        face = faces[0]  
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_region = image[y:y+h, x:x+w]
        features = preprocess_image(face_region)
        features = features.reshape(1, -1)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        prediction = model_with_hog.predict(features_scaled)
        predicted_emotion = emotion_labels[int(prediction[0])]

        # Preparing the face image for display
        output_face_image = Image.fromarray(face_region)  
        output_face_image = resize_image(output_face_image) 
    else:
        predicted_emotion = "No face detected"
        output_face_image = Image.fromarray(image)  
        output_face_image = resize_image(output_face_image)  

    return predicted_emotion, output_face_image

def predict_without_hog(image):
    """Predict emotion using the model trained without HOG features."""
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    if faces:
        face = faces[0]  
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_region = image[y:y+h, x:x+w]

        resized_image = cv2.resize(face_region, (48, 48))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        reshaped_image = gray_image.reshape(1, -1) 

        scaler = MinMaxScaler()
        normalized_image = scaler.fit_transform(reshaped_image)

        prediction = model_without_hog.predict(normalized_image)
        predicted_emotion = emotion_labels[int(prediction[0])]

        output_face_image = Image.fromarray(resized_image) 
        output_face_image = resize_image(output_face_image) 
    else:
        predicted_emotion = "No face detected"
        output_face_image = Image.fromarray(image) 
        output_face_image = resize_image(output_face_image) 

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

    # Example usage of predict_with_hog
    emotion_with_hog, face_img_with_hog = predict_with_hog(img)
    print(f"Predicted Emotion with HOG: {emotion_with_hog}")
    face_img_with_hog.show()

    # Example usage of predict_without_hog
    emotion_without_hog, face_img_without_hog = predict_without_hog(img)
    print(f"Predicted Emotion without HOG: {emotion_without_hog}")
    face_img_without_hog.show()

import pickle
import cv2
import numpy as np
from skimage import feature
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PIL import Image
import dlib

# Define the model filenames
model_filenames = {
    'svm_with_hog': 'svm_model_with_hog.pkl',
    'svm_without_hog': 'svm_model_without_hog.pkl',
    'dt_with_hog': 'dt_model_with_hog.pkl',
    'dt_without_hog': 'dt_model_without_hog.pkl',
    'rf_with_hog': 'rf_model_with_hog.pkl',
    'rf_without_hog': 'rf_model_without_hog.pkl'
}

# Load model function
def load_model(model_key):
    with open(model_filenames[model_key], 'rb') as f_in:
        model = pickle.load(f_in)
    return model

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Emotion labels
emotion_labels = {
    0: "Angry",
    1: "Fear",
    2: "Happy",
    3: "Neutral",
    4: "Sad",
}

# Preprocess image function
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    hog_features = feature.hog(resized, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), block_norm='L2-Hys',
                               visualize=False, transform_sqrt=True)
    return hog_features

# Resize image function
def resize_image(image, target_size=(300, 300)):
    return image.resize(target_size)

# Prediction functions
def predict_with_model(image, model_key):
    model = load_model(model_key)

    if 'with_hog' in model_key:
        return predict_with_hog(image, model)
    else:
        return predict_without_hog(image, model)

def predict_with_hog(image, model):
    if isinstance(image, Image.Image):
        image = np.array(image)

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

        prediction = model.predict(features_scaled)
        predicted_emotion = emotion_labels[int(prediction[0])]

        output_face_image = Image.fromarray(face_region)
        output_face_image = resize_image(output_face_image)
    else:
        predicted_emotion = "No face detected"
        output_face_image = Image.fromarray(image)
        output_face_image = resize_image(output_face_image)

    return predicted_emotion, output_face_image

def predict_without_hog(image, model):
    if isinstance(image, Image.Image):
        image = np.array(image)

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

        prediction = model.predict(normalized_image)
        predicted_emotion = emotion_labels[int(prediction[0])]

        output_face_image = Image.fromarray(resized_image)
        output_face_image = resize_image(output_face_image)
    else:
        predicted_emotion = "No face detected"
        output_face_image = Image.fromarray(image)
        output_face_image = resize_image(output_face_image)

    return predicted_emotion, output_face_image

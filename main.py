import numpy as np
import cv2
from PIL import Image
from mtcnn.mtcnn import MTCNN
import gradio as gr
from keras.models import model_from_json

# Load the emotion classification model
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Assuming the file paths will be added correctly before running the code
json_file = open('Emotion_detection_with_CNN-main/model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("Emotion_detection_with_CNN-main/model/emotion_model.h5")

# Initialize MTCNN detector for face detection
mtcnn_detector = MTCNN()

def preprocess_face_image(face_image):
    face_image = cv2.resize(face_image, (48, 48))  # Resize the image to 48x48
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_image = face_image.astype("float") / 255.0
    face_image = np.expand_dims(face_image, axis=-1)  # Add channel dimension
    face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension
    return face_image

def classify_emotion(face_image):
    processed_image = preprocess_face_image(face_image)
    emotion_prediction = emotion_model.predict(processed_image)
    # Use the emotion_dict for mapping prediction to emotion label
    emotion_text = emotion_dict[np.argmax(emotion_prediction)]
    return emotion_text

def detect_faces_with_haar(image):
    image_np = np.array(image)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return Image.fromarray(image_np)

def detect_faces_with_mtcnn(image):
    image_np = np.array(image)
    results = mtcnn_detector.detect_faces(image_np)
    for result in results:
        bounding_box = result['box']
        face = image_np[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]
        emotion_text = classify_emotion(face)
        cv2.rectangle(image_np, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 155, 255), 2)
        cv2.putText(image_np, f"{emotion_text}", (bounding_box[0], bounding_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
    return Image.fromarray(image_np)

def detect_faces(image, method):
    if method == "Haar Cascade":
        return detect_faces_with_haar(image)
    elif method == "MTCNN":
        return detect_faces_with_mtcnn(image)
    else:
        return image  # Return the original image if an invalid method is chosen

interface = gr.Interface(
    fn=detect_faces,
    inputs=[gr.Image(label="Upload Image"), gr.Radio(["Haar Cascade", "MTCNN"], label="Detection Method")],
    outputs=gr.Image(type="pil", label="Detected Faces"),
    title="Face Detection",
    description="Select a detection method and upload an image to detect faces."
)

if __name__ == "__main__":
    interface.launch()

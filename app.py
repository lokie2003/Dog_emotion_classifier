import streamlit as st
import numpy as np
import tensorflow as tf
import pathlib
import tensorflow_hub as hub
from PIL import Image

# Register the custom KerasLayer
hub_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4", trainable=False)

# Define the emotion labels
emotion_labels = ['angry', 'happy', 'relaxed', 'sad']

# Load the pre-trained model
def load_model():
    with tf.keras.utils.custom_object_scope({'KerasLayer': hub_layer}):
        model = tf.keras.models.load_model('dog_emotion.h5')
    return model

model = load_model()

# Define a function to preprocess the image using Pillow (PIL)
def preprocess_image_pil(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    return img

# Streamlit UI
st.title('Dog Emotion Classifier')

# Upload an image
uploaded_image = st.file_uploader('Upload a dog image', type=['jpg', 'jpeg'])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process the image
    processed_image = preprocess_image_pil(image)

    # Make a prediction
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class_index = np.argmax(prediction)
    predicted_class = emotion_labels[predicted_class_index]
    predicted_class_probability = prediction[0][predicted_class_index]

    # Display the prediction
    st.write(f'Predicted Emotion: {predicted_class}')
    st.write(f'Predicted Emotion Probability: {predicted_class_probability:.2f}')

# Streamlit footer
st.sidebar.markdown('**About**')
st.sidebar.text('This is a dog emotion classifier using a pre-trained ResNet model.')
st.sidebar.text('Built with TensorFlow and Streamlit.')

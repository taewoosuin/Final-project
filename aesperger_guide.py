import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image, ExifTags
import random
import os

# Load the model
model = load_model('MPANET.h5', compile=False)

# Class names (emotions)
class_names = ['Anger', 'Happy', 'Sadness', 'Panic']

# Emotion descriptions
emotion_descriptions = {
    'Anger': '보통 눈썹이 찌푸려지고 입이 꽉 다물어집니다.',
    'Happy': '보통 입꼬리가 올라가고 눈이 밝아집니다.',
    'Sadness': '보통 입꼬리가 내려가고 눈에 눈물이 맺힙니다.',
    'Panic': '보통 눈이 크게 뜨이고 입이 벌어집니다.'
}

# Initialize prediction history
if 'history' not in st.session_state:
    st.session_state.history = {'Anger': 0, 'Happy': 0, 'Sadness': 0, 'Panic': 0}

def preprocess_image(image):
    """
    Function to preprocess the image for the model input.
    """
    # Resize the image to 224x224
    image = cv2.resize(image, (224, 224))
    # Convert the image to RGB if it's not
    if image.shape[2] == 4:  # If the image has an alpha channel, remove it
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[2] == 1:  # If the image is grayscale, convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Normalize the pixel values to the [0, 1] range
    image = image / 255.0
    # Reshape the image to the model input format
    image = np.expand_dims(image, axis=0)
    return image

def predict_emotion(image):
    """
    Function to predict the emotion from the preprocessed image.
    """
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    return class_names[predicted_class]

def show_example_images(emotion):
    """
    Function to show example images from a specific emotion folder.
    """
    example_folder = os.path.join('examples', emotion.lower())
    example_images = random.sample(os.listdir(example_folder), 5)
    # Streamlit 열 생성
    columns = st.columns(5)

    # 각 열에 이미지와 캡션을 표시
    for i, col in enumerate(columns):
        img_path = os.path.join(example_folder, example_images[i])
        img = Image.open(img_path)
        img = rotate_image_based_on_exif(img)
        col.image(img, caption=emotion, use_column_width=True)
        
def rotate_image_based_on_exif(image):
    """
    이미지 객체의 EXIF 정보를 기반으로 이미지를 올바른 방향으로 회전시킵니다.
    """
    try:
        # EXIF 태그 중 Orientation 키를 찾습니다.
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        
        # 이미지의 EXIF 정보를 가져옵니다.
        exif=dict(image._getexif().items())

        # 이미지의 방향에 따라 이미지를 회전시킵니다.
        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # 이미지에 EXIF 정보가 없거나 처리 중 오류가 발생한 경우
        # 이미지를 그대로 반환합니다.
        pass
    return image


# Streamlit app setup
st.title('Aesper Guide')
st.write('사진을 업로드하고 감정을 맞춰보세요!')

# Image upload
uploaded_file = st.file_uploader("사진 올리기", type=["jpg", "jpeg", "png"])

# Initialize prediction history for correct and incorrect guesses
if 'correct_history' not in st.session_state:
    st.session_state.correct_history = {'Anger': 0, 'Happy': 0, 'Sadness': 0, 'Panic': 0}
if 'incorrect_history' not in st.session_state:
    st.session_state.incorrect_history = {'Anger': 0, 'Happy': 0, 'Sadness': 0, 'Panic': 0}

if uploaded_file is not None:
    # Open the image as a PIL image
    image = Image.open(uploaded_file)
    
    # EXIF 정보를 기반으로 이미지 회전
    image = rotate_image_based_on_exif(image)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    # Convert the image to a numpy array
    image_np = np.array(image)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image_np)
    
    # Predict the emotion
    predicted_emotion = predict_emotion(preprocessed_image)
    
    # Display buttons for emotion guessing
    st.write("감정을 맞춰보세요:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button('Anger'):
            user_guess = 'Anger'
    with col2:
        if st.button('Happy'):
            user_guess = 'Happy'
    with col3:
        if st.button('Sadness'):
            user_guess = 'Sadness'
    with col4:
        if st.button('Panic'):
            user_guess = 'Panic'
    
    # Check if user has made a guess
    if 'user_guess' in locals():
        if user_guess == predicted_emotion:
            st.write(f'정답입니다! {predicted_emotion}')
            st.write(emotion_descriptions[predicted_emotion])
            # Update correct history
            st.session_state.correct_history[predicted_emotion] += 1
        else:
            st.write(f'오답입니다. 실제 정답은 {predicted_emotion}입니다.')
            st.write(emotion_descriptions[predicted_emotion])
            st.write('예시 이미지를 참고하세요:')
            show_example_images(predicted_emotion)
            # Update incorrect history
            st.session_state.incorrect_history[user_guess] += 1
            
            
    st.markdown("<h3><b>히스토리</b></h3>", unsafe_allow_html=True)
    # 두 개의 컬럼 생성
    col_correct, col_incorrect = st.columns(2)

    # 첫 번째 컬럼에 정답 히스토리 출력
    with col_correct:
        st.markdown("<b>정답 히스토리</b>", unsafe_allow_html=True)
        for emotion, count in st.session_state.correct_history.items():
            st.write(f'{emotion}: {count}회')

    # 두 번째 컬럼에 오답 히스토리 출력
    with col_incorrect:
        st.markdown("<b>오답 히스토리</b>", unsafe_allow_html=True)
        for emotion, count in st.session_state.incorrect_history.items():
            st.write(f'{emotion}: {count}회')

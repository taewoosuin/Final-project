from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import base64
import torch
from transformers import ElectraForSequenceClassification, AutoTokenizer
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# 감정 레이블 설정
emotion_map_text = {
    0: "화남",
    1: "행복",
    2: "공황",
    3: "슬픔"
}

emotion_labels_image = ['화남', '행복', '공황', '슬픔']

# 모델과 토크나이저 불러오기
num_classes = 4  # 클래스 수에 맞게 설정
text_model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=num_classes)
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

# 저장된 텍스트 모델 불러오기
text_model.load_state_dict(torch.load("model.pt"))

# MPANET 모델 로드
image_model = load_model('MPANET.h5')

# GAN 생성기 모델 로드
gan_generator = load_model('gan_generator.h5')

# 텍스트 감정 분류 함수
def classify_emotion_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = text_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    return emotion_map_text[predicted_class]

# 이미지 감정 분류 함수
def classify_emotion_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    predictions = image_model.predict(img)
    predicted_class = np.argmax(predictions)
    
    return emotion_labels_image[predicted_class]

# 감정 조합에 따른 문구 출력 함수
def get_combination_message(text_emotion, image_emotion):
    combination = f"{text_emotion}_{image_emotion}"
    
    message_map = {
        "화남_화남": "분노가 불꽃처럼 타올라 마음을 태운다.",
        "화남_행복": "분노 속에서도 웃음을 잃지 않는 강인함.",
        "화남_공황": "화염 같은 분노가 공황과 뒤섞여 혼란을 일으킨다.",
        "화남_슬픔": "분노의 불씨 속에서 슬픔의 눈물이 흐른다.",
        "행복_화남": "행복의 빛 뒤에 숨은 어둠의 그림자.",
        "행복_행복": "햇살처럼 따스한 행복이 마음을 가득 채운다.",
        "행복_공황": "행복의 정점에서 느껴지는 불안한 떨림.",
        "행복_슬픔": "행복의 한편에 자리한 고요한 슬픔.",
        "공황_화남": "공황으로 인한 분노가 용암처럼 끓어오른다.",
        "공황_행복": "뜻밖의 기쁨이 공황처럼 다가온다.",
        "공황_공황": "공황과 공황이 동시에 밀려와 마음을 휘감는다.",
        "공황_슬픔": "공황의 여파로 슬픔이 파도처럼 밀려온다.",
        "슬픔_화남": "슬픔 속에서 피어오르는 분노의 불꽃.",
        "슬픔_행복": "슬픔의 한가운데서 피어나는 작은 행복.",
        "슬픔_공황": "슬픔과 공황이 엉켜 끝없는 나락으로 떨어진다.",
        "슬픔_슬픔": "깊고 어두운 슬픔이 마음을 잠식한다."
    }
    
    return message_map.get(combination, "조합에 따른 문구가 없습니다.")

# GAN을 사용하여 이미지를 생성하는 함수
def generate_image(text_emotion, image_emotion):
    # 감정 레이블 매핑
    emotion_map = {"화남": 0, "행복": 1, "공황": 2, "슬픔": 3}
    
    # 노이즈 벡터 생성
    noise = np.random.normal(0, 1, (1, 100))
    
    # 레이블 벡터 생성
    text_label = np.array([emotion_map[text_emotion]])
    image_label = np.array([emotion_map[image_emotion]])
    
    # 두 감정 레이블의 평균을 구하여 혼합된 레이블 생성
    mixed_label = (text_label + image_label) / 2.0
    
    # 생성기 모델을 사용하여 이미지 생성
    generated_image = gan_generator.predict([noise, mixed_label])
    
    # 이미지 후처리 (이미지 값을 [0, 1]로 변환)
    generated_image = 0.5 * generated_image + 0.5
    generated_image = (generated_image * 255).astype(np.uint8)
    
    # 이미지를 base64로 인코딩
    pil_img = Image.fromarray(generated_image[0])
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return encoded_image

@app.route('/analyze', methods=['POST'])
def analyze_emotion():
    data = request.json
    text = data['text']
    image_data = data['image']

    # Decode the image
    image_bytes = base64.b64decode(image_data.split(",")[1])
    
    text_emotion = classify_emotion_text(text)
    image_emotion = classify_emotion_image(image_bytes)
    
    message = get_combination_message(text_emotion, image_emotion)
    generated_image = generate_image(text_emotion, image_emotion)
    
    return jsonify({
        "text_emotion": text_emotion,
        "image_emotion": image_emotion,
        "message": message,
        "generated_image": generated_image
    })

@app.route('/')
def serve_index():
    return render_template('Instory.html')

if __name__ == "__main__":
    app.run(debug=True)

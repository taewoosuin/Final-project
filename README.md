# [2기] AI 모델 개발자 부트캠프 WASSUP
## Final Project 1조

### 프로젝트 파일
- **메인 이미지 분류 모델 코드:** MPANET.ipynb
- **이미지 분류 모델:** MPANET.h5
- **텍스트 분류 모델:** model.pt
- **GAN 모델:** gan_generator.h5

### 웹사이트 실행 방법 (AsuperGuide, Addstory)

```bash
# Step 01: TensorFlow Model Compatibility
# 모델은 TensorFlow 2.15에서 작동합니다 (TensorFlow 2.16.0 이상에서는 실행 불가).
# TensorFlow 2.15는 Python 3.12 이상 버전에서는 설치할 수 없습니다.
# Python 3.11로 가상환경을 설정하고 TensorFlow 2.15.0을 설치합니다.

# Step 02: 가상환경 생성
# 기존 가상환경을 제거합니다 (필요시).
rmdir /s /q venv
# VSCode에서 프로젝트 폴더 (app.py가 있는 폴더)를 엽니다.
# 터미널을 열고 Python 3.11을 사용하여 가상환경을 생성합니다.
(python 3.11 경로) -m venv venv
# 예시:
C:\Users\public.DESKTOP-KSOI6C0\AppData\Local\Programs\Python\Python311\python -m venv venv

# Step 03: 가상환경 접속
# 가상환경을 활성화합니다.
venv\Scripts\activate

# Step 04: 필요한 패키지 설치
# 각 서비스의 requirements.txt 파일을 참고하여 필요한 패키지를 설치합니다.
pip install -r requirements.txt

# Step 05: 프로그램 실행
# 가상환경이 활성화된 상태에서 해당 Python 파일을 실행합니다 (예: AsperGuide.py).
python AsperGuide.py

# Step 06: 애플리케이션 접속
# 웹 브라우저를 열고 아래 주소로 접속합니다:
http://127.0.0.1:5000/

# 데스크탑 프로그램 (실시간 감정 감지, 면접 연습 프로그램)
# .ipynb 파일을 열고 필요한 라이브러리를 설치합니다. 이때 TensorFlow는 2.15.0 버전을 사용해야 합니다.
# 필요한 라이브러리를 설치한 후 노트북을 실행합니다.

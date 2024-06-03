# [2nd Term] AI Model Developer Bootcamp WASSUP
## Final Project Team 1

### Project Files
- Models
  - **Main Image Classification Model Code:** MPANET.ipynb
  - **Image Classification Model:** MPANET.h5
  - **Text Classification Model:** model.pt
  - **GAN Model:** gan_generator.h5

- Services
  - **AsperGuide:** A web assistant for Asperger's Syndrome therapy
  - **Realtime Emotion Analysis:** A real-time emotion detection program
  - **Addstory:** A photo + diary program
  - **Interview Practice:** An interview practice program

### How to Run the Website (AsperGuide, Instory)

# Step 01: TensorFlow Model Compatibility
The model works with TensorFlow 2.15 (it does not run on TensorFlow 2.16.0 or higher).
TensorFlow 2.15 cannot be installed on Python 3.12 or higher.
Set up a virtual environment with Python 3.11 and install TensorFlow 2.15.0.

# Step 02: Create a Virtual Environment
Remove the existing virtual environment if necessary.
rmdir /s /q venv
Open the project folder (where app.py is located) in VSCode.
Open the terminal and create a virtual environment using Python 3.11.
(python 3.11 path) -m venv venv
Example:
C:\Users\public.DESKTOP-KSOI6C0\AppData\Local\Programs\Python\Python311\python -m venv venv

# Step 03: Activate the Virtual Environment
Activate the virtual environment.
venv\Scripts\activate

# Step 04: Install Required Packages
Refer to the requirements.txt file for each service and install the necessary packages.
pip install -r requirements.txt

# Step 05: Run the Program
With the virtual environment activated, run the corresponding Python file (e.g., AsperGuide.py).
python AsperGuide.py

# Step 06: Access the Application
Open a web browser and go to the following address:
http://127.0.0.1:5000/

# Desktop Programs (Real-time Emotion Detection, Interview Practice Program)
Open the .ipynb file and install the necessary libraries. Make sure to use TensorFlow 2.15.0.

Refer to the requirements.txt file to install the necessary libraries and then run the notebook.

================================================================================================================ 

# [2기] AI 모델 개발자 부트캠프 WASSUP
## Final Project 1조

### 프로젝트 파일
- 모델
  - **메인 이미지 분류 모델 코드:** MPANET.ipynb
  - **이미지 분류 모델:** MPANET.h5
  - **텍스트 분류 모델:** model.pt
  - **GAN 모델:** gan_generator.h5

- 서비스
  - **AsperGuide:** 아스파거증후군 치료 보조 웹사이트
  - **Realtime Emotion Analysis:** 실시간 감정 감지 프로그램
  - **Addstory:** 사진+일기 프로그램
  - **interview practice:** 면접 연습 프로그램

### 웹사이트 실행 방법 (AsperGuide, Instory)

# Step 01: TensorFlow Model Compatibility
모델은 TensorFlow 2.15에서 작동합니다 (TensorFlow 2.16.0 이상에서는 실행 불가).
TensorFlow 2.15는 Python 3.12 이상 버전에서는 설치할 수 없습니다.
Python 3.11로 가상환경을 설정하고 TensorFlow 2.15.0을 설치합니다.

# Step 02: 가상환경 생성
기존 가상환경을 제거합니다 (필요시).
rmdir /s /q venv
VSCode에서 프로젝트 폴더 (app.py가 있는 폴더)를 엽니다.
터미널을 열고 Python 3.11을 사용하여 가상환경을 생성합니다.
(python 3.11 경로) -m venv venv
예시:
C:\Users\public.DESKTOP-KSOI6C0\AppData\Local\Programs\Python\Python311\python -m venv venv

# Step 03: 가상환경 접속
가상환경을 활성화합니다.
venv\Scripts\activate

# Step 04: 필요한 패키지 설치
각 서비스의 requirements.txt 파일을 참고하여 필요한 패키지를 설치합니다.
pip install -r requirements.txt

# Step 05: 프로그램 실행
가상환경이 활성화된 상태에서 해당 Python 파일을 실행합니다 (예: AsperGuide.py).
python AsperGuide.py

# Step 06: 애플리케이션 접속
웹 브라우저를 열고 아래 주소로 접속합니다:
http://127.0.0.1:5000/

# 데스크탑 프로그램 (실시간 감정 감지, 면접 연습 프로그램)
.ipynb 파일을 열고 필요한 라이브러리를 설치합니다. 이때 TensorFlow는 2.15.0 버전을 사용해야 합니다.

requirements를 참고해 필요한 라이브러리를 설치한 후 노트북을 실행합니다.

================================================================================================================ 

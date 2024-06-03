How to Run the Website(AsuperGuide, Addstory)
Step 01: TensorFlow Model Compatibility

The model works with TensorFlow 2.15 (not compatible with TensorFlow 2.16.0 or higher).
TensorFlow 2.15 cannot be installed with Python version 3.12 or higher.
Set up a virtual environment with Python 3.11 and install TensorFlow 2.15.0.
Step 02: Create a Virtual Environment

Remove the existing virtual environment if necessary.
Open the project folder in VSCode (the folder containing app.py).
Open the terminal and run the command to create a virtual environment using Python 3.11.
Step 03: Activate the Virtual Environment

Activate the virtual environment.
Step 04: Install Required Packages

Refer to the requirements.txt file of each service to install necessary packages.
Step 05: Run the Program

While the virtual environment is activated, execute the appropriate Python file (e.g., AsperGuide.py).
Step 07: Access the Application

Open your web browser and navigate to http://127.0.0.1:5000/.

Desktop Program (Real-time Emotion, Interview Practice)
Open the .ipynb file and install the necessary libraries, ensuring TensorFlow is version 2.15.0.
Execute the notebook after installing the required libraries.

kor

웹사이트 실행 방법(AsuperGuide, Addstory)

01
tensorflow 2.15에서 작동하는 모델(tf 2.16.0 이상에서는 실행 불가)
->파이썬 버전(3.12 이상)에 따라 2.15 설치 불가능
->python 3.11로 가상환경 설정하고 tf 2.15.0으로 설치

02 가상환경 생성
-기존 가상환경 제거(필요시)
rmdir /s /q venv or 탐색기에서직접 삭제
-VScode에서 프로젝트 폴더 열기(app.py가 있는 폴더)
-터미널 열고 아래의 명령어
(python 3.11 경로) -m venv venv
ex
C:\Users\public.DESKTOP-KSOI6C0\AppData\Local\Programs\Python\Python311\python -m venv venv

03 가상환경 접속
venv\Scripts\activate

04 필요한 패키지 설치
각 서비스의 requirements.txt 참고

05 실행(가상환경 접속상태에서)
python AsperGuide.py
(실행할 py파일)

07 접속
http://127.0.0.1:5000/

데스크탑 프로그램(실시간 감정 감지, 면접 연습 프로그램)
ipynb 파일로 들어가 필요한 라이브러리 설치 후 실행(tf는 2.15.0)
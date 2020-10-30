import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc
#h5
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence 
from PIL import Image

app = Flask(__name__)


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        # 업로드 파일 처리 분기
        file = request.files['image']
        if not file: 
            return render_template('index.html', label="No Files")

        # 이미지 픽셀 정보 읽기
        # 알파 채널 값 제거 후 1차원 Reshape
        #img = misc.imread(file)
        img = Image.open(file)
        img = img.convert("RGB")
        img = img.resize((128, 128))
        data = np.asarray(img)
        
        X=[]
        X.append(data)
        X = np.array(X)
        # 입력 받은 이미지 예측
        prediction = model.predict(X)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        
        for i in prediction:
            pre_ans = i.argmax()  # 예측 레이블
            label = ''
            if pre_ans == 0:  label= "총"
            elif pre_ans == 1: label = "칼"
            elif pre_ans == 2: label = "스마트폰"
            else: pre_ans_str = "없음"
            if i[0] >= 0.3 : 
                label = '총'
            if i[1] >= 0.3: 
                label = '칼'
            if i[2] >= 0.3: 
                label = '스마트폰'

        # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
        #label = str(np.squeeze(prediction))

        # 숫자가 10일 경우 0으로 처리
        #if label == '10': label = '0'

        # 결과 리턴
        return render_template('index.html', label=label)


if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    # 모델 확장자가 pkl인 경우, model = joblib.load('./model/model.pkl')
    # 모델 확장자가 h5인 경우, model = keras.models.load_model('./model/Con2D_step3.h5')
    model = keras.models.load_model('./model/Con2D_model_batch20_adam_9_1_mid_trainargue.h5')
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=5000, debug=True)
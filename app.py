# /app.py
from flask import Flask, render_template, request
from flask import url_for, flash, redirect, session
from werkzeug.utils import secure_filename
from flask import make_response
import pickle
#from flask_socketio import SocketIO
#import cv2
import tkinter as tk
from PIL import ImageTk, Image
import os
import time
import re
import similar
import merge
import warnings
import os
import random
warnings.filterwarnings('ignore')


app = Flask(__name__)

#HTML 초기화면 렌더링
@app.route('/')
def home_page():
	return render_template('first_screen.html')

@app.route('/jeju')
def jeju_page():
    return render_template('jeju.html')

@app.route('/jeju_is1')
def jeju_is1_page():
    return render_template('jeju/jeju_is1.html')
@app.route('/jeju_is2')
def jeju_is2_page():
    return render_template('jeju/jeju_is2.html')
@app.route('/jeju_is3')
def jeju_is3_page():
    return render_template('jeju/jeju_is3.html')
@app.route('/jeju_is4')
def jeju_is4_page():
    return render_template('jeju/jeju_is4.html')
@app.route('/jeju_is5')
def jeju_is5_page():
    return render_template('jeju/jeju_is5.html')
@app.route('/jeju_is6')
def jeju_is6_page():
    return render_template('jeju/jeju_is6.html')
@app.route('/jeju_is7')
def jeju_is7_page():
    return render_template('jeju/jeju_is7.html')
@app.route('/jeju_is8')
def jeju_is8_page():
    return render_template('jeju/jeju_is8.html')
@app.route('/jeju_is9')
def jeju_is9_page():
    return render_template('jeju/jeju_is9.html')
@app.route('/jeju_is10')
def jeju_is10_page():
    return render_template('jeju/jeju_is10.html')
@app.route('/jeju_is11')
def jeju_is11page():
    return render_template('jeju/jeju_is11.html')
@app.route('/jeju_is12')
def jeju_is12_page():
    return render_template('jeju/jeju_is12.html')

@app.route('/six_is1_1')
def six_picture_is1_1():
    return render_template('six_picture/six_picture_is1_1.html')
@app.route('/six_is1_2')
def six_picture_is1_2():
    return render_template('six_picture/six_picture_is1_2.html')
@app.route('/six_is1_3')
def six_picture_is1_3():
    return render_template('six_picture/six_picture_is1_3.html')
@app.route('/six_is1_4')
def six_picture_is1_4():
    return render_template('six_picture/six_picture_is1_4.html')
@app.route('/six_is2_1')
def six_picture_is2_1():
    return render_template('six_picture/six_picture_is2_1.html')
@app.route('/six_is2_2')
def six_picture_is2_2():
    return render_template('six_picture/six_picture_is2_2.html')
@app.route('/six_is3_1')
def six_picture_is3_1():
    return render_template('six_picture/six_picture_is3_1.html')
@app.route('/six_is3_2')
def six_picture_is3_2():
    return render_template('six_picture/six_picture_is3_2.html')
@app.route('/six_is3_3')
def six_picture_is3_3():
    return render_template('six_picture/six_picture_is3_3.html')
@app.route('/six_is3_4')
def six_picture_is3_4():
    return render_template('six_picture/six_picture_is3_4.html')
@app.route('/six_is4_1')
def six_picture_is4_1():
    return render_template('six_picture/six_picture_is4_1.html')
@app.route('/six_is4_2')
def six_picture_is4_2():
    return render_template('six_picture/six_picture_is4_2.html')
@app.route('/six_is4_3')
def six_picture_is4_3():
    return render_template('six_picture/six_picture_is4_3.html')
@app.route('/six_is4_4')
def six_picture_is4_4():
    return render_template('six_picture/six_picture_is4_4.html')
@app.route('/six_is4_5')
def six_picture_is4_5():
    return render_template('six_picture/six_picture_is4_5.html')
@app.route('/six_is5_1')
def six_picture_is5_1():
    return render_template('six_picture/six_picture_is5_1.html')
@app.route('/six_is5_2')
def six_picture_is5_2():
    return render_template('six_picture/six_picture_is5_2.html')
@app.route('/six_is5_3')
def six_picture_is5_3():
    return render_template('six_picture/six_picture_is5_3.html')
@app.route('/six_is5_4')
def six_picture_is5_4():
    return render_template('six_picture/six_picture_is5_4.html')
@app.route('/six_is6_1')
def six_picture_is6_1():
    return render_template('six_picture/six_picture_is6_1.html')
@app.route('/six_is6_2')
def six_picture_is6_2():
    return render_template('six_picture/six_picture_is6_2.html')
@app.route('/six_is6_3')
def six_picture_is6_3():
    return render_template('six_picture/six_picture_is6_3.html')
@app.route('/six_is6_4')
def six_picture_is6_4():
    return render_template('six_picture/six_picture_is6_4.html')
@app.route('/six_is6_5')
def six_picture_is6_5():
    return render_template('six_picture/six_picture_is6_5.html')
@app.route('/six_is7_1')
def six_picture_is7_1():
    return render_template('six_picture/six_picture_is7_1.html')
@app.route('/six_is7_2')
def six_picture_is7_2():
    return render_template('six_picture/six_picture_is7_2.html')
@app.route('/six_is7_3')
def six_picture_is7_3():
    return render_template('six_picture/six_picture_is7_3.html')
@app.route('/six_is7_4')
def six_picture_is7_4():
    return render_template('six_picture/six_picture_is7_4.html')
@app.route('/six_is7_5')
def six_picture_is7_5():
    return render_template('six_picture/six_picture_is7_5.html')
@app.route('/six_is8_1')
def six_picture_is8_1():
    return render_template('six_picture/six_picture_is8_1.html')
@app.route('/six_is8_2')
def six_picture_is8_2():
    return render_template('six_picture/six_picture_is8_2.html')
@app.route('/six_is8_3')
def six_picture_is8_3():
    return render_template('six_picture/six_picture_is8_3.html')
@app.route('/six_is8_4')
def six_picture_is8_4():
    return render_template('six_picture/six_picture_is8_4.html')
@app.route('/six_is9_1')
def six_picture_is9_1():
    return render_template('six_picture/six_picture_is9_1.html')
@app.route('/six_is9_2')
def six_picture_is9_2():
    return render_template('six_picture/six_picture_is9_2.html')
@app.route('/six_is9_3')
def six_picture_is9_3():
    return render_template('six_picture/six_picture_is9_3.html')
@app.route('/six_is9_4')
def six_picture_is9_4():
    return render_template('six_picture/six_picture_is9_4.html')
@app.route('/six_is9_5')
def six_picture_is9_5():
    return render_template('six_picture/six_picture_is9_5.html')
@app.route('/six_is10_1')
def six_picture_is10_1():
    return render_template('six_picture/six_picture_is10_1.html')
@app.route('/six_is10_2')
def six_picture_is10_2():
    return render_template('six_picture/six_picture_is10_2.html')
@app.route('/six_is10_3')
def six_picture_is10_3():
    return render_template('six_picture/six_picture_is10_3.html')
@app.route('/six_is10_4')
def six_picture_is10_4():
    return render_template('six_picture/six_picture_is10_4.html')
@app.route('/six_is11_1')
def six_picture_is11_1():
    return render_template('six_picture/six_picture_is11_1.html')
@app.route('/six_is11_2')
def six_picture_is11_2():
    return render_template('six_picture/six_picture_is11_2.html')
@app.route('/six_is11_3')
def six_picture_is11_3():
    return render_template('six_picture/six_picture_is11_3.html')
@app.route('/six_is12_1')
def six_picture_is12_1():
    return render_template('six_picture/six_picture_is12_1.html')
@app.route('/six_is12_2')
def six_picture_is12_2():
    return render_template('six_picture/six_picture_is12_2.html')
@app.route('/six_is12_3')
def six_picture_is12_3():
    return render_template('six_picture/six_picture_is12_3.html')
@app.route('/six_is12_4')
def six_picture_is12_4():
    return render_template('six_picture/six_picture_is12_4.html')

@app.route('/six', methods=['GET', 'POST'])
def six():
    if request.method == 'POST':
        selected_image = request.form.get('image')  # 선택된 이미지의 파일 이름을 가져옴
        image_folder_path = f"static/pca_image/{selected_image}"  # 이미지 폴더 경로 생성

        merge.main(image_folder_path)  # 이미지 폴더 경로를 merge.main() 함수에 전달하여 실행

        return render_template('result_sim.html')
        
		
@app.route('/result')
def result():
    x = similar.simi()
    #x = 71
    return render_template('similar.html', x=x)


#서버 실행
if __name__ == '__main__':
	app.run(host='0.0.0.0', port=7800, debug = True)
 


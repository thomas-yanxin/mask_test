#-*- coding:utf-8 -*-
#@Time :  22:53
#@Author: Thomas
#@File :mask.py
#@Software : PyCharm

import paddlehub as hub
import cv2
from playsound import playsound

module = hub.Module(name="pyramidbox_lite_mobile_mask")     #口罩检测模型
face_cascade = cv2.CascadeClassifier("D:/mask/haarcascade_frontalface_default.xml")     #人脸识别分类器
capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)     #初始化摄像头

while(True):

    # 获取一帧
    ret, frame = capture.read()
    #灰度转换
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0:

        for faceRect in faces:
            x, y, w, h = faceRect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h // 2, x:x + w]
            roi_color = frame[y:y + h // 2, x:x + w]


    cv2.imshow('frame', frame)      #展现
    file_name="D:/mask/mask_test.png"       #图片存储路径
    cv2.imwrite(file_name, frame)       #写入获取的一帧
    #口罩检测模型
    input_dict = {"data": [cv2.imread(file_name)]}
    results = module.face_detection(data=input_dict)
    #判断是否检测到人脸和是否戴口罩
    mask = results[0]["data"]
    if len(mask) != 0:
        mask_test = mask[0]['label']
        if mask_test=="NO MASK":
            playsound('D:\\mask\\11750.wav')    #若检测到未戴口罩则发出警报  （路径注意！）
            print(mask_test)
        else:
            print(mask_test)
    else:
        print("未检测到人脸！请将脸移入摄像头视角范围内！")

    cv2.imwrite(file_name,frame)

    if cv2.waitKey(3) & 0xff == 27:
        print("退出程序！谢谢使用！")
        capture.release()           #释放摄像头q
        cv2.destroyAllWindows()         #删除建立的全部窗口
        break



# mask_test
Face Mask Detection
面部口罩识别检测@人脸识别
项目介绍
【

项目背景：随着新冠疫情的爆发，公共卫生防护程度被提高到空前状态。为防止新冠病毒的交叉传染，导致疫情扩散，人们在各大公共场所活动时均被要求佩戴口罩。因此，面部口罩检测项目诞生。

项目前景：随着公共卫生防护程度的逐步提升，且佩戴口罩成为出入各大公共场所的前提，面部口罩检测成为各大公共场所管理的必备操作。而随着智能化、自动化进程的加快，口罩面部检测已逐步从人工向机器转移。

项目价值：提高公共卫生防护能力；节约社会资源和运营成本；降低人力资源成本。

使用场景：公交车、地铁站、超市、学校等各大公共场所。

……
】

·假正经·
【

新冠疫情在中国爆发，百度某团队率先开源口罩人脸检测及分类模型，（极大地体现了百度这家公司的社会责任感），随即一维弦防疫巡检机器人部署清华李文正馆，为学生开学保驾护航……

】

……

用马校长的话讲，“我对任何事情都感到好奇，想踹开门，迈开步，进去看看到底是怎么一回事”。

出于好奇，加之受百度良心课程之【百度架构师手把手带你零基础实践深度学习】的启发，在家写了一个简易的面部口罩检测程序，即“50行代码实现面部口罩检测”。

看上去好像NB的样子，实则是百度太良心！！paddlehub太好用！！

思路其实很简单，即：获取图像→人脸识别→口罩检测

·下面进入快乐代码时间 ·
以下代码主要实现前两者，即获取图像和人脸识别：

import cv2
face_cascade = cv2.CascadeClassifier("D:\\mask\\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True:
   ret, frame = cap.read()
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.1, 5)
   if len(faces) > 0:
       for faceRect in faces:
           x, y, w, h = faceRect
           cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
           roi_gray = gray[y:y + h // 2, x:x + w]
           roi_color = frame[y:y + h // 2, x:x + w]
   cv2.imshow('frame', frame)  # 展现
   file_name = "D:/mask/mask_test.png"  # 图片存储路径
   cv2.imwrite(file_name, frame)  # 写入获取的一帧
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release() # 释放摄像头
cv2.destroyAllWindows()

通过opencv调取内置摄像头以获取图像，并对图像进行人脸识别检测，将获取的图片保存在本地以便后面进行口罩检测。

以下代码主要实现面部的口罩检测识别：

import paddlehub as hub
import cv2
module = hub.Module(name=”pyramidbox_lite_mobile_mask”) #口罩检测模型
test_img_path = “C:\Users\Dell\Pictures\3.png”# 读取本地图片
input_dict = {“data”: [cv2.imread(test_img_path)]}
results = module.face_detection(data=input_dict)
print(results)
通过导入面部口罩检测模型，读取本地图片并对其进行分析检测，从而得到是否佩戴口罩的结果。

简单的整合一下，就形成了项目完整的50行代码：

-- coding:utf-8 --
@Time : 22:53
@Author: Thomas
@File :mask.py
@Software : PyCharm

import paddlehub as hub
import cv2
from playsound import playsound

module = hub.Module(name=”pyramidbox_lite_mobile_mask”) #口罩检测模型
face_cascade = cv2.CascadeClassifier(“D:/mask/haarcascade_frontalface_default.xml”) #人脸识别分类器
capture = cv2.VideoCapture(0,cv2.CAP_DSHOW) #初始化摄像头

while(True):

    # 获取一帧
    ret, frame = capture.read()     #以帧换视频流
    #灰度转换
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0:

        for faceRect in faces:
            x, y, w, h = faceRect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h // 2, x:x + w]
            roi_color = frame[y:y + h // 2, x:x + w]

    #口罩检测模型
    cv2.imshow('frame', frame)      #展现
    file_name="D:/mask/mask_test.png"       #图片存储路径
    cv2.imwrite(file_name, frame)       #写入获取的一帧
    input_dict = {"data": [cv2.imread(file_name)]}
    results = module.face_detection(data=input_dict)
    #判断是否检测到人脸和是否戴口罩
    mask = results[0]["data"]
    if len(mask) == 0:

        print("未检测到人脸！请将脸移入摄像头视角范围内！")
    else:
        mask_test = mask[0]['label']
        if mask_test == "NO MASK":
            playsound('D:\\mask\\11750.wav')  # 若检测到未戴口罩则发出警报
            print(mask_test)
        else:
            print(mask_test)

    cv2.imwrite(file_name,frame)

    if cv2.waitKey(3) & 0xff == 27:
        print("退出程序！谢谢使用！")
        capture.release()           #释放摄像头q
        cv2.destroyAllWindows()         #删除建立的全部窗口
        break

准确率
口罩人脸检测部分在准确度上达到了 98%，且口罩人脸分类部分准确率同样达到了 96.5%。

AI Studio项目链接：https://aistudio.baidu.com/aistudio/projectdetail/724861

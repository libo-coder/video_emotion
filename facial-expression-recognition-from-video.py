import numpy as np
import cv2
from keras.preprocessing import image

# -----------------------------
# opencv initialization
face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

# -----------------------------
# face expression recognizer initialization
from keras.models import model_from_json

model = model_from_json(open("./model/facial_expression_model_structure.json", "r").read())
model.load_weights('./model/facial_expression_model_weights.h5')

# -----------------------------
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

cap = cv2.VideoCapture('example.flv')
frame = 0
while True:
    ret, img = cap.read()
    img = cv2.resize(img, (640, 360))
    img = img[0:308, :]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # 灰度视频，将捕获的视频转换为灰色并且保存
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)     # 在图像中找到面孔，将检测到的face位置返回

    for (x, y, w, h) in faces:
        if w > 30:  # 忽略过于小的人脸
            # cv2.rectangle(img,(x,y), (x+w,y+h), (64,64,64),2)                   # 突出显示 face 位置
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]           # crop detected face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)     # transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48))                 # resize to 48x48
            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)         # store probabilities of 7 expressions
            max_index = np.argmax(predictions[0])
            # emotion = emotions[max_index]

            # background of expression list
            overlay = img.copy()
            opacity = 0.4           # 不透明度
            cv2.rectangle(img, (x + w + 10, y - 25), (x + w + 150, y + 115), (64, 64, 64), cv2.FILLED)
            cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

            # connect face and expressions
            cv2.line(img, (int((x + x + w) / 2), y + 15), (x + w, y - 20), (255, 255, 255), 1)
            cv2.line(img, (x + w, y - 20), (x + w + 10, y - 20), (255, 255, 255), 1)

            emotion = ""
            for i in range(len(predictions[0])):
                emotion = "%s %s%s" % (emotions[i], round(predictions[0][i] * 100, 2), '%')

                """
                if i != max_index:
                    color = (255,0,0)
                """

                color = (255, 255, 255)
                cv2.putText(img, emotion, (int(x+w+15), int(y-12+i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow('img', img)
    frame = frame + 1
    # print(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
import numpy as np
import cv2
from keras.preprocessing import image

# -----------------------------
# opencv initialization

face_cascade = cv2.CascadeClassifier('./cascades/haarcascades/haarcascade_frontalface_default.xml')

# cap = cv2.VideoCapture('/Users/libo/Documents/Deep_Learning/Facial_Emotion/example.mp4')
cap = cv2.VideoCapture(0)

# -----------------------------
# face expression recognizer initialization
from keras.models import model_from_json

model = model_from_json(open("./model/facial_expression_model_structure.json", "r").read())
model.load_weights('./model/facial_expression_model_weights.h5')

# -----------------------------
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
while True:
    ret, img = cap.read()       # 按帧读取视频
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # 灰度视频，将摄像头捕获的视频转换为灰色并保存
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # print(faces)              # locations of detected faces

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)    # draw rectangle to main image

        detected_face = img[int(y):int(y + h), int(x):int(x + w)]       # crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) # transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48))             # resize to 48x48

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

        predictions = model.predict(img_pixels)     # store probabilities of 7 expressions

        # find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])

        emotion = emotions[max_index]

        # write emotion text above rectangle
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # process on detected face end
    # -------------------------

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
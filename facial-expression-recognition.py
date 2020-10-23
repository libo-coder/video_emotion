import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ------------------------------
sess = tf.InteractiveSession()
keras.backend.set_session(sess)
# ------------------------------
# variables
num_classes = 7  # angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 512
epochs = 5
# ------------------------------
# 读取数据集内容 (fer2013.csv)
with open("/Users/libo/Documents/Deep_Learning/datasets/fer2013.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print("number of instances: ", num_of_instances)
print("instance length: ", len(lines[1].split(",")[1].split(" ")))

# ------------------------------
# initialize train set and test set
x_train, y_train, x_test, y_test = [], [], [], []

# ------------------------------
# 之前已经加载过数据集，现在将训练集和测试集存储到专用变量中
for i in range(1, num_of_instances):
    try:
        emotion, img, usage = lines[i].split(",")
        val = img.split(" ")
        pixels = np.array(val, 'float32')
        emotion = keras.utils.to_categorical(emotion, num_classes)
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
        print("", end="")

# ------------------------------
# data transformation for train and test sets
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

# normalize inputs between [0, 1]
x_train /= 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# ------------------------------
# construct model structure
model = Sequential()

# 1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))   # 64个卷积核
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

# 2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

# 3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())

# fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

# ------------------------------
# batch process
gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

# ------------------------------

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# ------------------------------

fit = False

if fit:
    model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)
else:
    model.load_weights('/Users/libo/Documents/Deep_Learning/Facial_Emotion/model/facial_expression_model_weights.h5')

# ------------------------------

# 进行整体测试，得出准确率
score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])

# ------------------------------
# 将每个预测可视化为条形图
def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')

    plt.show()
# ------------------------------

monitor_testset_results = False

if monitor_testset_results:
    predictions = model.predict(x_test)     # make predictions for test set
    index = 0
    for i in predictions:
        if index < 30 and index >= 20:
            # print(i)                        # predicted scores
            # print(y_test[index])            # actual scores

            testing_img = np.array(x_test[index], 'float32')
            testing_img = testing_img.reshape([48, 48])

            plt.gray()
            plt.imshow(testing_img)
            plt.show()

            print(i)

            emotion_analysis(i)
            print("----------------------------------------------")
        index = index + 1

# ------------------------------
# 进行测试，识别自定义图像的面部表情

img = image.load_img("/Users/libo/Documents/Deep_Learning/Facial_Emotion/dataset/jackman.png",
                     color_mode="grayscale", target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x /= 255

custom = model.predict(x)
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48])

plt.gray()
plt.imshow(x)
plt.show()
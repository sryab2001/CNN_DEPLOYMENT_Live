""""@sryab@2001"""

""" SAVE THE MODEL USING mode.save() FUNCTION AND COPY AND PASTE YOUR MODEL WHICH USED """


import tensorflow as tf
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
i=0

while (cap.isOpened()):
    ret, frame = cap.read()



    if ret == False:
        break

    s=frame


    i += 1
    icd = {0: 'No Ship', 1: 'Ship'}

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

""" THE MODEL"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(100, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(100, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

""""LOAD YOUR MODEL EXACTLY USING THE SAME NAME"""
    model.load_weights("enemynavyvessel_model.h5")


""" FUNCTION FOR PREDICITION USING CNN """

    from keras.preprocessing.image import img_to_array

    def output(location):
        dim = (150,150)
        img = cv2.resize(location,dim)
        img = img_to_array(img)
        img = img / 255
        img = np.expand_dims(img, [0])
        answer = model.predict_classes(img)
        probability = round(np.max(model.predict_proba(img) * 100), 2)
        print(icd[answer[0]], 'With probability', probability)

    output(s)

cap.release()
cv2.destroyAllWindows()

""""@sryab@2001"""

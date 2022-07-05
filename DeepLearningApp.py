from ast import If
from re import I
import numpy as np  # linear algebra
import cv2 # opencv

# keras
from keras import models
from keras.layers import Flatten, Dense
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


model = models.load_model('new_model.h5')
model.load_weights('newmodelweight.hdf5')
video = cv2.VideoCapture("asdad.mp4")
face_detection_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img=cv2.imread("InkedHIMYM_LI.jpg")

while True:
   
    ret, img = video.read()
    
    if not ret:
        continue
    orig_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # colored output image

# Convert image to grayscale
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    return_faces = face_detection_model.detectMultiScale(
        img,
        scaleFactor=1.1,
        minNeighbors=5,
    )




    main_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # colored output image

    # For detected faces in the image
    for i in range(len(return_faces)):
        (x, y, w, h) = return_faces[i]
        cropped_face = main_img[y : y + h, x : x + w]
        cropped_face = cv2.resize(cropped_face, (128, 128))
        cropped_face = np.reshape(cropped_face, [1, 128, 128, 3]) / 255.0
        mask_result = model.predict(cropped_face)  # make model prediction

        if(mask_result.argmax()==0):
            print_label = "MASKELI" 
            label_colour = (0,255,0)
        else:
            print_label = "MASKESIZ" 
            label_colour = (255,0,0)

       
    
        cv2.putText(
            main_img,
            print_label,
            (x, y - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            label_colour,
            1,
        )  # print text

        cv2.rectangle(
            main_img,
            (x, y),
            (x + w, y + h),
            label_colour,
            5,
        )  # draw bounding box on face



    main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)  # colored output image
    cv2.imshow('Video', main_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# label for mask detection
video.release()
cv2.destroyAllWindows()
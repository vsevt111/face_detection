# This is a sample Python script.
import cv2
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
path = 'Faces'

recognizer = cv2.face.LBPHFaceRecognizer_create()
font = cv2.FONT_HERSHEY_SIMPLEX
# recognizer = cv2.face.EigenFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml");

def getLabel():
    labels={}
    names={}
    for i,person in enumerate(os.listdir(path)):
        labels[person]=i+1
        names[i+1]=person
    return labels,names
def getImageAndLabels(path,labels):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids=[]
    for imagePath in tqdm(imagePaths):
        imagePathsSinglePerson = [os.path.join(imagePath, f) for f in os.listdir(imagePath)]
        for image in imagePathsSinglePerson:
            PIL_img = Image.open(image).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            label = labels[image.split('\\')[1]]
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(label)

    return faceSamples, ids

# labels,names=getLabel()
# faces, ids = getImageAndLabels(path,labels)
# recognizer.train(faces, np.array(ids))
#
# recognizer.write('trainer.yml')
def detect_faces(img):
    # recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer.read('./trainer.yml')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,1.3,5)
    if faces is ():
        return img

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0),2)
        # id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        #
        # if confidence < 100:
        #     id = names[id]
        #     confidence = "{0}%".format(round(100 - confidence))
        # else:
        #     id = "unknown"
        #     confidence = "{0}%".format(round(100 - confidence))
        # cv2.putText(img, str(id), (x + 5, y - 5), font, 0.25, (255, 255, 255), 1)
        # cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 0.25, (255, 255, 255), 1)
    return img

cap = cv2.VideoCapture(0)

while True:

# for image in tqdm(os.listdir('Test')):
    res,frame = cap.read()
    # print(image)
    # frame = cv2.imread(os.path.join('Test',image))
    frame = detect_faces(frame)
    cv2.imshow('Video Face Detection', frame)
    # cv2.imwrite(f'outputs/{image}',frame)
    # print(plt.imshow(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# cv2.waitKey(0)

cv2.destroyAllWindows()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

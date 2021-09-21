import cv2
import numpy as np

img_rows = 64
img_cols = 64

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
print('CV2 haarcascades directory: ', cv2.data.haarcascades)

# Takes an Image as Input
# Returns location & size of detected face, detected face image in gray scale (ROI) of dim 64x64 pixel and the marked input image
# ROI = Region of Interest
def face_detector(img):

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return (0, 0, 0, 0), np.zeros((img_rows, img_cols), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

    try:
        roi_gray = cv2.resize(roi_gray, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
    except:
        return (x, y, w, h), np.zeros((img_rows, img_cols), np.uint8), img
    return (x, y, w, h), roi_gray, img


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)

    cv2.imshow('All', image)

    if cv2.waitKey(1) == 13:  # 13 is for the Enter Key
        break

cap.release()
cv2.destroyAllWindows()

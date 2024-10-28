import cv2
import face_recognition
import pickle
import os

# Importing student image
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])
print(studentIds)


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # encode = face_recognition.face_encodings(img)[0]
        # encodeList.append(encode)
        encodes = face_recognition.face_encodings(img)
        if encodes:  # Check if there's at least one face encoding
            encodeList.append(encodes[0])
        else:
            print("No face found in an image.")

    return encodeList


print("Encoding Start .....")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("Five Saved")
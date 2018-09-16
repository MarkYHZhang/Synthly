import numpy as np
import cv2
import math

import time

import subprocess
import threading

# Get input from Mark's thingy
# 0 1 2 3 4


sounds = ["A.wav", "B.wav", "C.wav", "D.wav", "E.wav"]


def playSound(note):
    subprocess.call(["aplay", sounds[note]])


global lastNote
lastNote = None
global nextTime
nextTime = time.time() + 0.3


def playNote(note):
    global nextTime
    global lastNote
    if note is lastNote: return
    if time.time() < nextTime: return
    t = threading.Thread(target=playSound, args=(), kwargs={"note": note})
    nextTime = time.time() + 0.3
    lastNote = note
    t.start()


# Open Camera
capture = cv2.VideoCapture(0)

capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1000)
capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1000)

logo_img = cv2.imread("logo.png", -1)

while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()

    frame = cv2.flip( frame, 1 )

    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (0, 0), (450, 450), (0, 255, 0), 0)
    crop_image = frame[0:450, 0:450]

    y1, y2 = 10, 10 + logo_img.shape[0]
    x1, x2 = 0, 0 + logo_img.shape[1]

    alpha_s = logo_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (alpha_s * logo_img[:, :, c] +
                                  alpha_l * frame[y1:y2, x1:x2, c])

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2    , 0, 0]), np.array([20, 255, 255]))

    # Kernel for morphological transformation
    kernel = np.ones((1, 1))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    # Show threshold image
    cv2.imshow("Thresholded", thresh)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        if cv2.contourArea(contour) > 500 :

            # Create bounding rectangle around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

            # Find convex hull
            hull = cv2.convexHull(contour)

            # Draw contour
            drawing = np.zeros(crop_image.shape, np.uint8)
            cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
            cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

            # Find convexity defects
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)

            # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
            # tips) for all defects
            count_defects = 0

            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

                # if angle > 90 draw a circle at the far point
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

                cv2.line(crop_image, start, end, [0, 255, 0], 2)

            # a floor tom
            # b hi-hat
            # c bass
            # d cymbal
            # e snare

            # Print number of fingers
            if count_defects == 0:
                cv2.putText(frame, "Floor Tom", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                playNote(0)
            elif count_defects == 1:
                cv2.putText(frame, "Hi-Hat", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                playNote(1)
            elif count_defects == 2:
                cv2.putText(frame, "Bass", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                playNote(2)
            elif count_defects == 3:
                cv2.putText(frame, "Cymbal", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                playNote(3)
            elif count_defects == 4:
                cv2.putText(frame, "Snare", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                playNote(4)
            else:
                pass
    except:
        pass


    try:
        # Show required images


        cv2.imshow("Gesture", frame)
        all_image = np.hstack((drawing, crop_image))
        cv2.imshow('Contours', all_image)
    except:
        pass

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

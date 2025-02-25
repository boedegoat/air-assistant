import cv2
import sys
import numpy as np
import mediapipe as mp
from pynput.mouse import Button, Controller
import time
import math
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

# TODO: 
#       3. bikin klik kanan (rencana 3 jari) (1.5 detik)

class handDetector():
    #maxHands = jangan diganti jadi dua(karena kita hanya mau track 1 saja)
    def __init__(self, mode=False, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, int(self.detectionCon), int(self.trackCon))
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        handedness = None
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            handedness = self.results.multi_handedness[handNo].classification[0].label  # "Left" atau "Right"
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        return lmList, handedness

    def fingersUp(self, img, lmList):
        if len(lmList) != 0:
            fingers = []  # List untuk menyimpan status jari
            for handNo in range(len(self.results.multi_hand_landmarks)): # loop untuk mencari koordinat tangan
                lmList, handedness = self.findPosition(img, handNo, draw=False)
                if lmList[self.tipIds[0]][1] < lmList[self.tipIds[4]][1]:  # Tangan kanan
                    if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:  # Ibu jari
                        fingers.append(0)  # Ketutup
                    else:
                        fingers.append(1)

                    for i in range(1, 5):  # Selain ibu jari
                        if lmList[self.tipIds[i]][2] < lmList[self.tipIds[i] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                else:  # Tangan kiri
                    if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0] - 1][1]:  # Ibu jari
                        fingers.append(0)  # Ketutup
                    else:
                        fingers.append(1)

                    for i in range(1, 5):  # Selain ibu jari
                        if lmList[self.tipIds[i]][2] < lmList[self.tipIds[i] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
        return fingers

    def findDistance(self, lmList, p1, p2, img, draw=True,r=15, t=3): #cari jarak antar jari
        x1, y1 = lmList[p1][1:] # jari 1
        x2, y2 = lmList[p2][1:] # jari 2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 #mencari nilai tengah

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1) # mengukur jarak Euclidean distance

        return length, img, [x1, y1, x2, y2, cx, cy]

class VirtualMouse(QMainWindow):
    def __init__(self, wCam=640, hCam=480, frameR=170, smoothening=5):
        super().__init__()
        self.setWindowTitle("mouse")
        self.setGeometry(100,200,wCam, hCam)
        self.video_label = QLabel(self)
        self.video_label.setGeometry(0,0,wCam,hCam)
        

        self.wCam, self.hCam = wCam, hCam
        self.frameR = frameR  # Frame bingkai kotak {ubah jikalau mau dikalibrasi sesuai keinginan(semakin tinggi semakin cepat)}
        self.smoothening = smoothening #kalibrasi
        self.plocX, self.plocY = 0, 0 # koordinat untuk kursor
        self.clocX, self.clocY = 0, 0
        self.mouse = Controller()
        self.up = 0
        
        # Camera Setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        
        # Hand Detector
        self.detector = handDetector() 
        
        # Screen Size
        self.wScr, self.hScr = 1920,1080

        #auto run
        self.timer = QTimer()
        self.timer.timeout.connect(self.cam)
        self.timer.start(20)
        
        #timer variabel
        self.clicked = time.time()
    def update_display(self):
        rgb_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))
    
    def cam(self):
        ret, self.img = self.cap.read()
        self.img = cv2.flip(self.img, 1)
        self.img = self.detector.findHands(self.img) 
        self.lmList, handedness = self.detector.findPosition(self.img)
        cv2.rectangle(self.img, (self.frameR, self.frameR), (self.wCam - self.frameR, self.hCam - self.frameR), (255, 0, 255), 2)
         # Display the image
        self.update_display()
        if len(self.lmList) != 0:
             self.hand_gesture()
        return self

    def hand_gesture(self):
        x1, y1 = self.lmList[8][1:] #telunjuk (mengambil koordinat x dan y)
        fingers = self.detector.fingersUp(self.img, self.lmList)
        print(fingers)
        if fingers[0] == 1 and fingers[2] != 1 and fingers: #jari telunjuk dan jempol
            self.click_mouse()
            self.mouse_release()#left
        elif fingers and fingers[0] != 1 and fingers[1] == 1 and fingers[2] != 1: #jari telunjuk
            self.move_mouse(x1, y1)
            self.mouse_release()
        elif fingers and fingers[0] != 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:  # 4 jari scroll bawah
            self.scroll_page("down")
        elif fingers and fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:  # 5 jari scroll atas
            self.scroll_page("up")
        elif fingers and fingers[0] != 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            self.click_mouse(ops="right")
            self.mouse_release()
        elif fingers and fingers[0] != 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] != 1 and fingers[4] != 1: #jari telunjuk dan jari tengah
            self.drag_mouse(x1, y1)
        else:
            self.mouse_release()
        # print(self.up)
        return self

    def move_mouse(self, x1, y1):
        x3 = np.interp(x1, (self.frameR, self.wCam - self.frameR), (0, self.wScr))
        y3 = np.interp(y1, (self.frameR, self.hCam - self.frameR), (0, self.hScr))
        self.clocX = self.plocX + (x3 - self.plocX) / self.smoothening
        self.clocY = self.plocY + (y3 - self.plocY) / self.smoothening
        if time.time() - self.clicked > 0.2:
            self.mouse.position = (self.clocX, self.clocY)
        self.plocX, self.plocY = self.clocX, self.clocY
    
    def drag_mouse(self, x1, y1):
        length, self.img, lineInfo = self.detector.findDistance(self.lmList,8, 12, self.img)
        x3 = np.interp(x1, (self.frameR, self.wCam - self.frameR), (0, self.wScr))
        y3 = np.interp(y1, (self.frameR, self.hCam - self.frameR), (0, self.hScr))
        if y3 > 1079:
            y3 = 1079
        self.clocX = self.plocX + (x3 - self.plocX) / self.smoothening
        self.clocY = self.plocY + (y3 - self.plocY) / self.smoothening
        if length < 25 and self.up == 0:
            self.mouse.press(Button.left)
            self.mouse.position = (self.clocX, self.clocY)
            self.up = 1
        elif length < 25 and self.up == 1:
            self.mouse.position = (self.clocX, self.clocY)
            self.up = 1
        self.plocX, self.plocY = self.clocX, self.clocY
    
    def mouse_release(self):
        if self.up == 1:
            self.mouse.release(Button.left)
            self.up = 0

    def click_mouse(self, ops="left"):
        if ops == "left":
            length1, self.img, lineInfo = self.detector.findDistance(self.lmList,4, 8, self.img) #ujung jari telunjuk dan jempol
            length2, self.img, lineInfo = self.detector.findDistance(self.lmList,4, 7, self.img) #sendi sebelum ujung jari telunjuk dan ujung jari jempol
            length = min(length1, length2)
            if length < 20 and time.time() - self.clicked > 0.4:
                self.mouse.click(Button.left, 1)
                self.clicked = time.time()
        elif ops == "right":
            length1, self.img, lineInfo = self.detector.findDistance(self.lmList, 8, 12, self.img) #ujung jari telunjuk dan tengah
            length2, self.img, lineInfo = self.detector.findDistance(self.lmList, 12, 16, self.img) #ujung jari tengah dan manis
            if length1 < 30 and length2 < 30 and time.time() - self.clicked > 1:
                self.mouse.click(Button.right, 1)
                self.clicked = time.time()
    
    def scroll_page(self, direction="up"):
        if direction == "up":
            print('scroll up')
            self.mouse.scroll(0, 1)  # Scroll up
        else:
            print('scroll down')
            self.mouse.scroll(0, -1)  # Scroll down

if __name__ == "__main__":
    app = QApplication(sys.argv)  # Create the Qt application instance
    virtual_mouse = VirtualMouse()  # Initialize the virtual mouse
    virtual_mouse.show()
    sys.exit(app.exec_())

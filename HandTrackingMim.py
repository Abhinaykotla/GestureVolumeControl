import cv2  
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                if id == 4 or id == 8:
                    print(id,lm)
                h, w, c =img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                #print(id , cx , cy)
                if id == 4 or id == 8: #(indent)
                    cv2.circle(img , (cx , cy), 10, (255,0,255), cv2.FILLED)
                    #print(id,lm)
                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    
    cv2.putText(img, str(int(fps)), (20,20) , cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
    
    
    cv2.imshow("UR tatti hand", img)
    cv2.waitKey(100) 
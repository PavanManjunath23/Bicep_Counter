import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose= mp.solutions.pose

def calculate_angle(a,b,c):
    a= np.array(a)   # x1,y1
    b= np.array(b)   # x2,y2
    c= np.array(c)   # x3,y3
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)
    
    if angle>180.0:
        angle = 360-angle
        
    return angle


cap=cv2.VideoCapture(0)
counter=0
stage=None
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret,frame=cap.read()
        #detect stuff and render
        #recolor
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make detection
        results = pose.process(image)
        # rrecoloring back to bgr
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            angle= calculate_angle(shoulder, elbow, wrist)
            angle=int(angle)
            #visualise angles
            cv2.putText(image, str(angle),
                       tuple(np.multiply(elbow, [640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2,cv2.LINE_AA
                       )
            # CURL COUNTER
            if angle>160:
                stage = "down"
            if angle<35 and stage=='down':
                stage="up"
                counter+=1
                print(counter)
        except:
            pass
        #render curl counter
        cv2.rectangle(image,(0,0),(225,73),(245,117,16),-1)
        cv2.putText(image,'REPS',(15,12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1,cv2.LINE_AA)
        cv2.putText(image,str(counter),(10,60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2  ,cv2.LINE_AA)
        cv2.putText(image,'STAGE',(65,12),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1,cv2.LINE_AA)
        cv2.putText(image,stage,(80,60),cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2  ,cv2.LINE_AA)
     # rendering
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66),thickness=2, circle_radius=2), # color change circle/joints
                                 mp_drawing.DrawingSpec(color=(245,66,230),thickness=5, circle_radius=2)) # color change
        
        cv2.imshow('MEDIAPIPE FEED',image)

        if cv2.waitKey(10)& 0xFF ==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
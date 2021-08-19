import cv2
import mediapipe as mp
import numpy as np
from dtaidistance import dtw
import matplotlib.pyplot as plt
from suppliments import class_handler as ch
import pickle

# https://github.com/AriAlavi/SigNN

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
Rotation=["clockwise","anticlockwise"]

sink_origin="Trikon/Bhumi_Trik.mp4"

sink = cv2.VideoCapture(sink_origin)
source = cv2.VideoCapture("Data/source.mp4")
source_fps=source.get(cv2.CAP_PROP_FPS)
width  = source.get(3)   # float `width`
height = source.get(4)

load_file=open("Trikon/IDEAL/trik_ideal.pickle","rb")
ideal=pickle.load(load_file)
load_file.close()

l_elbow=ch.Angles("left elbow",[11,13,15])
r_elbow=ch.Angles("right elbow",[12,14,16])

l_shoulder=ch.Angles("left shoulder",[13,11,23])
r_shoulder=ch.Angles("right shoulder",[14,12,24])

l_knee=ch.Angles("left knee",[23,25,27])
r_knee=ch.Angles("right knee",[24,26,28])

l_hip=ch.Angles("left hip",[11,23,25])
r_hip=ch.Angles("right hip",[12,24,26])

body_parts=[l_elbow,r_elbow,l_shoulder,r_shoulder,l_hip,r_hip,l_knee,r_knee]

def super_impose(sink_results,source_results):
    global height,width
    angles=[]
    base=np.array([1,0])
    up=[]
    hip_local=[]
    for i in [sink_results,source_results]:
        x_mid=0.50*(i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x+i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
        y_mid=0.50*(i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y+i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x=x_mid
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y=y_mid
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x=x_mid
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y=y_mid
        up.append(np.array([x_mid,y_mid]))
        x_mid=0.50*(i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x+i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x)
        y_mid=0.50*(i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y+i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y)
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x=x_mid
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y=y_mid
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x=x_mid
        i.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y=y_mid

        hip_local.append(np.array([x_mid,y_mid]))
        
    # sink_hip_world=np.array([sink_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,sink_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y])
    # source_hip_world=np.array([source_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,source_results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y])
    # source_hip_local=np.array([source_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,source_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y])

    # scale=[source_hip_local[0]/source_hip_world[0],source_hip_local[1]/source_hip_world[1]]
    offset=hip_local[0]-hip_local[1]

    for landmark_name in mp_pose.PoseLandmark:
        source_results.pose_landmarks.landmark[landmark_name].x+=offset[0]
        source_results.pose_landmarks.landmark[landmark_name].y+=offset[1]

    # torso=up-down
    # angles.append(np.degrees(np.arccos(np.dot(torso,base ) / (np.linalg.norm(torso) * np.linalg.norm(base)))))

    # print(offset,[width,height])
    # print([source_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,source_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y],[sink_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,sink_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y])

def trigger_func(A,B):
    return A-B

def align_pose(sink_pose, source_pose):
    return

trik_asana=ch.Asanas("Trikon",variables=[l_hip,r_hip],triggers={"left":{"g":30},"right":{"l":-30}},function=trigger_func)

def track_all_angles(source_pose,sink_pose,body_parts=body_parts):
    for tag,pose in zip(["source","sink"],[source_pose,sink_pose]):
        for part in body_parts:
            part.track_angle(pose,tag)

def display_instruction(stage,ideal=ideal,body_parts=body_parts):
    if stage=="right":
        load_file=open("Trikon/IDEAL/trik_ideal_right.pickle","rb")
        ideal=pickle.load(load_file)
        load_file.close()
    for n in range(len(body_parts)):
        match=body_parts[n].generate_instructions(ideal[n])
        if match==2:
            cv2.putText(sink_image,body_parts[n].name+"->"+"DONE",(300,30+n*30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(sink_image,body_parts[n].name+"->"+Rotation[match]+str(int(body_parts[n].sink_tracker[-1]))+"->"+str(int(ideal[n].source_tracker[-1])),(30,30+n*30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


sink_pose=mp_pose.Pose(
    min_detection_confidence=0.5,
    # model_complexity=1,
    min_tracking_confidence=0.5) 
source_pose=mp_pose.Pose(
    min_detection_confidence=0.5,
    # model_complexity=1,
    min_tracking_confidence=0.5) 

frame=0

# cv2.namedWindow("sink", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("sink", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while sink.isOpened():
    
    success, sink_image = sink.read()
    # source_image= cv2.imread("Data/pose.jpg")
    _,source_image= source.read()
    if not success or not _:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    frame+=1
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    sink_image = cv2.cvtColor(sink_image, cv2.COLOR_BGR2RGB)
    source_image = cv2.cvtColor(cv2.flip(source_image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    sink_image.flags.writeable = False
    source_image.flags.writeable = False
    sink_results = sink_pose.process(sink_image)
    source_results = source_pose.process(source_image)
    cv2.flip(sink_image, 1)
    try:
        track_all_angles(source_results,sink_results)
    except:
        pass
    
    # Draw the pose annotation on the image.
    sink_image.flags.writeable = True
    sink_image = cv2.cvtColor(sink_image, cv2.COLOR_RGB2BGR)
    try:
        super_impose(sink_results,source_results)
    except Exception as e:
        print("super impose:",e)

    mp_drawing.draw_landmarks(
        sink_image, sink_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(),mp_drawing.DrawingSpec(color=(0,0,255)))

    mp_drawing.draw_landmarks(
        sink_image, source_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)



    # try:
        
    # except Exception as e:
    #     print("instructions:",e)

    try:
        stage=trik_asana.detect_stage()
        display_instruction(stage)
    except Exception as e:
        print("stage detection:",e)

    cv2.imshow("sink",sink_image)

    # cv2.imshow("source",source_image)


    


    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(-1)
    elif key == ord('d'):
        frame+=60
        sink.set(cv2.CAP_PROP_POS_FRAMES, frame)
        source.set(cv2.CAP_PROP_POS_FRAMES, frame)
    elif key == ord('a'):
        frame-=60
        sink.set(cv2.CAP_PROP_POS_FRAMES, frame)
        source.set(cv2.CAP_PROP_POS_FRAMES, frame)

    
    

sink.release()
source.release()



# for part in body_parts:
#     print(part.name,part.generate_score())

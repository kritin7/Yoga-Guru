import cv2
import mediapipe as mp
import numpy as np
from dtaidistance import dtw
import matplotlib.pyplot as plt

# https://github.com/AriAlavi/SigNN

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


sink = cv2.VideoCapture(0)
source = cv2.VideoCapture("Data/test3_trim_trim.mp4")

def torso_length(result):
    ## images are inverted so left = right
    RIGHT_SHOULDER = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    RIGHT_HIP = result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    point1 = np.array([RIGHT_SHOULDER.x,RIGHT_SHOULDER.y])
    point2 = np.array([RIGHT_HIP.x,RIGHT_HIP.y])

    length_r = np.linalg.norm(point1-point2)

    LEFT_SHOULDER = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    LEFT_HIP = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    point1 = np.array([LEFT_SHOULDER.x,LEFT_SHOULDER.y])
    point2 = np.array([LEFT_HIP.x,LEFT_HIP.y])

    length_l = np.linalg.norm(point1-point2)

    return length_r, length_l

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,3))
sink_pose=mp_pose.Pose(
    min_detection_confidence=0.5,
    # model_complexity=1,
    min_tracking_confidence=0.5) 
source_pose=mp_pose.Pose(
    min_detection_confidence=0.5,
    # model_complexity=1,
    min_tracking_confidence=0.5) 

frame=0
sink_action=[]
source_action=[]

torso_length_l=[]
torso_length_r=[]

def distance(sink_action,source_action):
    score=[]
    for i in range(33):
        sink_act = sink_action[i]/ np.linalg.norm(sink_action[i])
        source_act = source_action[i]/ np.linalg.norm(source_action[i])
        score.append(100*(1-dtw.distance(sink_act,source_act)))

    return score

while sink.isOpened():
    
    success, sink_image = sink.read()
    # source_image= cv2.imread("Data/pose.jpg")
    _,source_image= source.read()
    if not success or not _:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    sink_image = cv2.cvtColor(cv2.flip(sink_image, 1), cv2.COLOR_BGR2RGB)
    source_image = cv2.cvtColor(cv2.flip(source_image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    sink_image.flags.writeable = False
    source_image.flags.writeable = False
    sink_results = sink_pose.process(sink_image)
    source_results = source_pose.process(source_image)

    # Draw the pose annotation on the image.
    sink_image.flags.writeable = True
    sink_image = cv2.cvtColor(sink_image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        sink_image, sink_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(),mp_drawing.DrawingSpec(color=(0,0,255)))
    # image=image.copy()
    mp_drawing.draw_landmarks(
        sink_image, source_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # print("sink",results.pose_landmarks,"\nsource",source_results.pose_landmarks)
    # size_img = image[0].shape
    cv2.imshow("pose",sink_image)
    cv2.imshow("source",source_image)
    sink_landmark=sink_results.pose_landmarks
    source_landmark=source_results.pose_landmarks
    # print(source_landmark)
    # print(landmark_points(source_landmark))
    # exit()
    try:
        temp=[]
        for data_point in sink_landmark.landmark:
            temp.append([data_point.x,data_point.y])
            # if data_point.x > 1 or data_point.y > 1:
                # print("hi") 
        sink_action.append(np.array(temp))
        # print(temp)
        temp=[]
        temp2=[]
        for data_point in source_landmark.landmark:
            temp.append([data_point.x,data_point.y])
            
            if len(temp2)<=33:
                temp2.append([data_point.x,data_point.y,data_point.z])
        source_action.append(np.array(temp))
        frame+=1

        ## this line
        torso_lengths=torso_length(source_results)

        torso_length_r.append(torso_lengths[0])
        torso_length_l.append(torso_lengths[1])
        # plt.scatter(frame,torso_length_l[-1])
        # plt.pause(0.05)
    except:
        pass

    

    
    if cv2.waitKey(5) & 0xFF == 27:
      break
# out.release()
sink.release()
source.release()

for i in range(frame):
    sink_act = sink_action[i]/ np.linalg.norm(sink_action[i])
    source_act = source_action[i]/ np.linalg.norm(source_action[i])

# plt.scatter(sink_act[0][:],sink_act[1][:])
# plt.show()

# for i in mp_pose.PoseLandmark:
#     print(i,"\n",sink_landmark[i])

# exit()

sink_action=np.array(sink_action)
source_action=np.array(source_action)

print("##################",frame,sink_action.shape,source_action.shape)
sink_action=sink_action.reshape(2*frame,-1)
source_action=source_action.reshape(2*frame,-1)

score=distance(sink_action,source_action)

print(score)
print("score",np.mean(score))

# print(len(action))

# print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


# ax=plt.axes(projection="3d")
# x=[]
# y=[]
# z=[]
# for i in temp2:
#     x.append(i[0])
#     y.append(i[1])
#     z.append(i[2])



# print(z)

# ax.scatter(x[11:],y[11:],z[11:])
# ax.scatter(x[:11],y[:11],z[:11])

# plt.show()

print("FPS = ",source.get(cv2.CAP_PROP_FPS))

plt.plot(torso_length_r)
plt.plot(torso_length_l)
plt.show()

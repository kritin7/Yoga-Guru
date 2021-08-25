from suppliments import class_handler as ch
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

l_elbow=ch.Angles("left elbow",[11,13,15])
r_elbow=ch.Angles("right elbow",[12,14,16])

l_shoulder=ch.Angles("left shoulder",[13,11,23])
r_shoulder=ch.Angles("right shoulder",[14,12,24])

l_knee=ch.Angles("left knee",[23,25,27])
r_knee=ch.Angles("right knee",[24,26,28])

l_hip=ch.Angles("left hip",[11,23,25])
r_hip=ch.Angles("right hip",[12,24,26])

body_parts=[l_elbow,r_elbow,l_shoulder,r_shoulder,l_hip,r_hip,l_knee,r_knee]

def track_all_angles(pose,body_parts=body_parts):
    for part in body_parts:
        part.track_angle(pose)


with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.8) as pose:
    img=cv2.imread("Trikon/IDEAL/image.jpg")
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    track_all_angles(results)

for part in body_parts:
    print(part.name,part.source_tracker)

save_file=open("Trikon/IDEAL/trik_ideal.pickle","wb")
pickle.dump(body_parts,save_file)
save_file.close()
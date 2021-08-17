import numpy as np
import mediapipe as mp
from dtaidistance import dtw

mp_pose=mp.solutions.pose
landmarks_name=[]
Threshhold=25

for i in mp_pose.PoseLandmark:
    landmarks_name.append(i)
class Angles:
    def __init__(self, name="to be set",points="to be set"):
        self.name=name
        self.points=points
        self.source_tracker=[]
        self.sink_tracker=[]
        self.score=None
        self.threshold=Threshhold
        self.match=1

        # print("tracking "+str(self.name)+", tracking part "+str(landmarks_name[points[0]])+str(landmarks_name[points[1]])+str(landmarks_name[points[2]]))

    def calculate_angle(self,pose,tag="source"):
        if self.points=="to be set" or len(self.points)!=3:
            print(self.name,"the points have not been set properly")
            return
        a=np.array([pose.pose_world_landmarks.landmark[landmarks_name[self.points[0]]].x,pose.pose_landmarks.landmark[landmarks_name[self.points[0]]].y]) 
        b=np.array([pose.pose_world_landmarks.landmark[landmarks_name[self.points[1]]].x,pose.pose_landmarks.landmark[landmarks_name[self.points[1]]].y]) 
        c=np.array([pose.pose_world_landmarks.landmark[landmarks_name[self.points[2]]].x,pose.pose_landmarks.landmark[landmarks_name[self.points[2]]].y]) 
        # print(a)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def track_angle(self,pose,tag="source"):
        # print(landmarks_name[23],pose.pose_landmarks.landmark[landmarks_name[23]].visibility ,pose.pose_landmarks.landmark[landmarks_name[23]].presence )
        angle=self.calculate_angle(pose)
        # print(angle)
        if tag=="source":
            self.source_tracker.append(angle)
        else:
            self.sink_tracker.append(angle)
        

    def generate_score(self):
        self.score=100*(1-dtw.distance(self.source_tracker,self.sink_tracker))
        return self.score

    def generate_instructions(self,ideal):
        angle1=self.sink_tracker[-1]
        angle2=ideal.source_tracker[-1]
        if angle1<angle2+self.threshold and angle1>angle2-self.threshold:
            self.match=2
        elif angle1>angle2+self.threshold:
            self.match=0
        elif angle1<angle2-self.threshold:
            self.match=1
        return self.match
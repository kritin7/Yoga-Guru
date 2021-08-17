import cv2
import mediapipe as mp
import numpy as np
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import matplotlib.pyplot as plt


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def dt(s1,s2):
    s1 = np.array(s1)
    s2 = np.array(s2)
    path = dtw.warping_path(s1, s2)
    dtwvis.plot_warping(s1, s2, path)
    dist = dtw.distance(s1,s2)
    dist_n = dtw.distance(s1/np.linalg.norm(s1),s2/np.linalg.norm(s2))
    print(dist,dist_n)
    dist = 100*(1-dist)
    dist_n = 100*(1-dist_n)
    return dist,dist_n


true = [89.65446228236605, 88.22130722956251, 87.92792602150264, 87.62829208879305, 86.74810904928303, 86.62085368202929, 87.25009184194896, 87.22100769087348, 86.60023013293987, 86.52896422822337, 86.7411108849349, 87.01055592600602, 86.32693682435315, 85.87498951431535, 83.8350768366698, 82.05179498991627, 81.34076220943686, 81.09483041233428, 80.64877652387032, 80.4006693376166, 78.87002689019417, 78.77294078905693, 79.66602162838907, 81.7986496033753, 88.23023531682692, 97.90381791768361, 110.32405838198102, 120.89959762216088, 138.4557526710933, 157.843313285859, 169.87439193116091, 175.43022757844184, 176.96950165270042, 175.00586605682707, 174.12399927870084, 174.55475507793187, 174.68702371122248, 174.91021152066713, 175.30976583805705, 175.23131029455456, 175.42226007914434, 175.2184881403495, 175.3123472985393, 175.13588778217039, 175.52811724790132, 175.34142222226293, 175.0894739817762, 174.71659126753758, 174.3870491567346, 171.65302881478715, 167.43601036795565, 160.95176231655344, 151.5161456453538, 138.90280008344044, 127.62397186460751, 116.02135930051445, 106.10992067924406, 98.93285358645724, 96.58720644485031, 83.52479366011731, 89.77755581770612, 83.41259043563194, 89.89287554027699, 98.1789559675451, 88.28062112463422, 103.95863988328747, 91.34168058951018, 99.70720267792623, 89.16115702045497, 76.2517574542091, 116.03167887933876, 94.39703327811495, 71.62302531875318, 74.42070348743164, 83.4389077394599, 85.45981304928371, 87.77777353278937, 91.09252807125567, 92.47023751961406, 94.60991366163513, 95.19741482022476, 96.00478335300387, 94.87912053640311, 91.4055284262194, 95.57840169851386, 98.11622336843942, 99.71822604873931, 101.88091807151814, 100.6880433534085, 101.15669211095812, 102.79198883947006, 109.65831306778522, 121.14959705830879, 127.16140337559091, 135.02750039315507, 145.71642174378113]
tr_without_bgr_rgb = [89.65446228236605, 88.22130722956251, 87.92792602150264, 87.62829208879305, 86.74810904928303, 86.62085368202929, 87.25009184194896, 87.22100769087348, 86.60023013293987, 86.52896422822337, 86.7411108849349, 87.01055592600602, 86.32693682435315, 85.87498951431535, 83.8350768366698, 82.05179498991627, 81.34076220943686, 81.09483041233428, 80.64877652387032, 80.4006693376166, 78.87002689019417, 78.77294078905693, 79.66602162838907, 81.7986496033753, 88.23023531682692, 97.90381791768361, 110.32405838198102, 120.89959762216088, 138.4557526710933, 157.843313285859, 169.87439193116091, 175.43022757844184, 176.96950165270042, 175.00586605682707, 174.12399927870084, 174.55475507793187, 174.68702371122248, 174.91021152066713, 175.30976583805705, 175.23131029455456, 175.42226007914434, 175.2184881403495, 175.3123472985393, 175.13588778217039, 175.52811724790132, 175.34142222226293, 175.0894739817762, 174.71659126753758, 174.3870491567346, 171.65302881478715, 167.43601036795565, 160.95176231655344, 151.5161456453538, 138.90280008344044, 127.62397186460751, 116.02135930051445, 106.10992067924406, 98.93285358645724, 96.58720644485031, 83.52479366011731, 89.77755581770612, 83.41259043563194, 89.89287554027699, 98.1789559675451, 88.28062112463422, 103.95863988328747, 91.34168058951018, 99.70720267792623, 89.16115702045497, 76.2517574542091, 116.03167887933876, 94.39703327811495, 71.62302531875318, 74.42070348743164, 83.4389077394599, 85.45981304928371, 87.77777353278937, 91.09252807125567, 92.47023751961406, 94.60991366163513, 95.19741482022476, 96.00478335300387, 94.87912053640311, 91.4055284262194, 95.57840169851386, 98.11622336843942, 99.71822604873931, 101.88091807151814, 100.6880433534085, 101.15669211095812, 102.79198883947006, 109.65831306778522, 121.14959705830879, 127.16140337559091, 135.02750039315507, 145.71642174378113]

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
#cap = cv2.VideoCapture('test.mp4')
cap = cv2.VideoCapture('Piyush_Padam.mp4')
elbow_list = []
shoulder_list = []
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        # Recolor image to RGB
        image = frame.copy()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            # Calculate angle
            angle_elbow = calculate_angle(shoulder, elbow, wrist)
            angle_shoulder = calculate_angle(hip, shoulder, elbow)
            elbow_list.append(angle_elbow)
            shoulder_list.append(angle_shoulder)
            # Visualize angle
            cv2.putText(image, str(int(angle_elbow)),
                        tuple(np.multiply(elbow, [frameWidth, frameHeight]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(angle_shoulder)),
                        tuple(np.multiply(shoulder, [frameWidth, frameHeight]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        except:
            print("not detected")
            pass


        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(elbow_list)
    score, score_n = dt(true,elbow_list)
    print(score, score_n)
    print(f'final score: {score_n}')
    plt.plot(elbow_list)
    plt.plot(shoulder_list)
    plt.show()

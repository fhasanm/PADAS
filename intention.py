



# noinspection PyUnresolvedReferences
import cv2
# noinspection PyUnresolvedReferences
import time
# noinspection PyUnresolvedReferences
import mediapipe as mp
# noinspection PyUnresolvedReferences
import numpy as np

def my_intention():

    mp_drawing = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



    cap = cv2.VideoCapture(0)


    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ingoring empty camera frame")
            #if loading a video, use "break" instead of "continue".
            continue

        start = time.time()


        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results1 = face_mesh.process(image)
        results2 = pose.process(image)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

    #pose landmarks estimation
        if results2.pose_landmarks:
            mp_drawing.draw_landmarks(image, results2.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results2.pose_landmarks.landmark):
                # print(id, lm)
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                cv2.circle(image, (x, y), 5, (255, 0, 0), cv2.FILLED)


    #head directions estimation
        if results1.multi_face_landmarks:
            for face_landmarks in results1.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # get coordinates
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                      [0, focal_length, img_w / 2],
                                      [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rmat, jac = cv2.Rodrigues(rot_vec)   #get rotational matrix

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)   #get angles

                x = angles[0] * 360    #get rotational degree
                y = angles[1] * 360
                z = angles[2] * 360

                #see where the user's head tilting

                if y < -5:
                    text = "turn left"
                elif y > 5:
                    text = "turn right"
                #elif x < -5:
                    #text = "looking down"
                #elif x > 20:
                    #text = "looking up"
                else:
                    text = "forward"




                #add the text on the image
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x:" + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y:" + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z:" + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            end = time.time()
            totalTime = end -start

            fps = 1 / totalTime
            print("FPS:", fps)

            cv2.putText(image, f'FPS:{int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)


        cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(5) == ord("q"):
            break

    cap.release()

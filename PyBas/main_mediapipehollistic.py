import cv2
import mediapipe as mp
import json
import pprint
import asyncio

# import IO.socketio_server as Server

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()


def extract_keypoints(results):
    keypoints = {}
    if results.pose_landmarks:
        keypoints["pose"] = [
            {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility,
            }
            for landmark in results.pose_landmarks.landmark
        ]
    if results.face_landmarks:
        keypoints["face"] = [
            {"x": landmark.x, "y": landmark.y, "z": landmark.z}
            for landmark in results.face_landmarks.landmark
        ]
    if results.left_hand_landmarks:
        keypoints["left_hand"] = [
            {"x": landmark.x, "y": landmark.y, "z": landmark.z}
            for landmark in results.left_hand_landmarks.landmark
        ]
    if results.right_hand_landmarks:
        keypoints["right_hand"] = [
            {"x": landmark.x, "y": landmark.y, "z": landmark.z}
            for landmark in results.right_hand_landmarks.landmark
        ]
    return keypoints


def visualize_and_send(frame, results, send_function):
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )
    if results.face_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION
        )
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )

    keypoints = extract_keypoints(results)
    keypoints_json = json.dumps({"keypoints": keypoints})
    pprint.pprint(keypoints_json)
    send_function(keypoints_json)

    return frame


def main(send_function):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("frames not found")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        frame_with_landmarks = visualize_and_send(frame, results, send_function)

        cv2.imshow("output", frame_with_landmarks)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    def empty(dingo):
        pass
    asyncio.run(main(empty))

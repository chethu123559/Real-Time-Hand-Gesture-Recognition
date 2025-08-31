import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils



def classify_gesture(landmarks, handedness="Right"):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    finger_pips = [6, 10, 14, 18] 
    thumb_tip = 4
    thumb_ip = 3 

    open_fingers = []

   
    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y: 
            open_fingers.append(1)
        else:
            open_fingers.append(0)

   
    if handedness == "Right":
        thumb_open = landmarks[thumb_tip].x < landmarks[thumb_ip].x
    else:  
        thumb_open = landmarks[thumb_tip].x > landmarks[thumb_ip].x

   
    if sum(open_fingers) == 4 and thumb_open:
        return "Open Palm"
    elif sum(open_fingers) == 0 and not thumb_open:
        return "Fist"
    elif open_fingers[0] == 1 and open_fingers[1] == 1 and open_fingers[2] == 0 and open_fingers[3] == 0 and not thumb_open:
        return "Peace Sign"
    elif sum(open_fingers) == 0 and thumb_open:
        return "Thumbs Up"
    else:
        return "Unknown"



cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            handedness_label = hand_handedness.classification[0].label

           
            gesture = classify_gesture(hand_landmarks.landmark, handedness_label)

            
            cv2.putText(frame, f"{gesture} ({handedness_label})", (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()

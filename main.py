import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,min_tracking_confidence=0.3)
interpreter = Interpreter(model_path='sign_language.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
hand_cascade = cv2.CascadeClassifier('haarcascade_hand.xml')
signs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p'
                         , 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']



def draw_hand_rectangle(frame, hand_landmarks):
    # Get the bounding box for the hand
    min_x = min(point.x for point in hand_landmarks.landmark)
    max_x = max(point.x for point in hand_landmarks.landmark)
    min_y = min(point.y for point in hand_landmarks.landmark)
    max_y = max(point.y for point in hand_landmarks.landmark)

    # Convert normalized coordinates to pixel coordinates
    h, w, _ = frame.shape
    x, y, w, h = int(min_x * w), int(min_y * h), int((max_x - min_x) * w), int((max_y - min_y) * h)
    frame_predict = frame[y:y+h, x:x+w]
    sign = classify_sign(frame_predict)
    # Draw a rectangle around the detected hand
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, f'Expression: {sign}', (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def classify_sign(hand_roi):
    if hand_roi is not None and hand_roi.shape[0] > 0 and hand_roi.shape[1] > 0: 
        resized_hand = cv2.resize(hand_roi, (28, 28))
        gray = cv2.cvtColor(resized_hand, cv2.COLOR_BGR2GRAY)
        input_data = gray / 255.0
        input_data = np.float32(input_data)
        input_data =input_data.reshape((1,28,28,1))
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get the predicted expression
        expression = np.argmax(output_data)
        return signs[expression]
    else : return None 
    

def main():
    cap = cv2.VideoCapture("/home/bakhil_aissa/My project/sign_languge/My first SHORT about SHORT in American Sign Language.mp4")

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame,(600,400))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        if not ret:
            break
       
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                  draw_hand_rectangle(frame,hand_landmarks)

        cv2.imshow('sign', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
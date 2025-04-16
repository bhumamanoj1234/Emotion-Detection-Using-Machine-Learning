import cv2
from deepface import DeepFace


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()


emotional_support = {
    "angry": "Take a deep breath. Everything will be okay.   ",
    "sad": "It's okay to feel sad. You're not alone. ",
    "happy": "You're glowing with happiness! Keep smiling! ",
    "surprise": "Wow! Something amazing just happened! ",
    "neutral": "Stay calm and carry on. You're doing great! ",
    "fear": "Don't worry. You're safe and strong. ",
    "disgust": "Eww! Let's move past this and focus on the good. "
}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        frame = cv2.resize(frame, (1280, 720))

        
        try:
            
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            
            if isinstance(result, list):
                result = result[0]

            
            emotion = result.get('dominant_emotion', 'neutral').lower()
            face_bbox = result.get('region', None)

            if face_bbox:
                x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['w'], face_bbox['h']

                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                
                message = emotional_support.get(emotion, "Stay positive! 😊")
                y_message = y + h + 30
                cv2.putText(frame, message, (x, y_message), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error in emotion detection: {e}")

        
        cv2.imshow('Emotion Detection - DeepFace', frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

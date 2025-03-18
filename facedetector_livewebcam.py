import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Kamera tidak bisa diakses.")
    exit()

print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
            cv2.putText(frame, 'Smiling', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 0, 0), 2)
            break 
    cv2.imshow("Face and Smile Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

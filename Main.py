import cv2
import os

# Create folder to save captured images
save_path = "captured_images"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)  # 0 is default webcam

print("Press 'c' to capture, 'q' to quit.")

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Camera", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Save image
        img_name = f"{save_path}/object_{count}.png"
        cv2.imwrite(img_name, frame)
        print(f"Captured {img_name}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
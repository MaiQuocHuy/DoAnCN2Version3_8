import os
import cv2
import random
import string

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes is 9
# number_of_classes = 10
dataset_size = 100

# Test different indices if 2 does not work
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Failed to open camera. Trying index 0 instead.")
    cap = cv2.VideoCapture(0)

# List of letters 'A' to 'I' and numbers '1' to '9'
# characters = list(string.ascii_uppercase[:5]) + [str(i) for i in range(1, 6)]
characters = list(string.ascii_uppercase[:9]) + [str(i) for i in range(1, 6)]
# print(len(characters[0]))
# break
for char in characters:
    class_dir = os.path.join(DATA_DIR,char)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {char}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Check the camera connection.")
            cap.release()
            cv2.destroyAllWindows()
            exit()  # Exit if frame capture fails

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Ending capture.")
            break  # Exit loop if frame capture fails

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()

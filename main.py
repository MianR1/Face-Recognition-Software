import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# create window
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 750)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

clock = 0 # frame counter for scanning frequency
face_match = False # initialize face match to false
match_img = cv2.imread("image.jpg") # set reference image

# function to check for a face match
def face_check(frame_captured):
    global face_match # refers to the same variable defined outside the function, at the global scope
    try:
        # use the DeepFace library to determine a match
        if DeepFace.verify(frame_captured, match_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        # prevention of program crash due to value errors
        face_match = False

while True:
    # capture video frames from a webcam or camera using OpenCV
    return_val, frame = cap.read()

    if return_val:
        # check for a match every 30 frames
        if clock % 30 == 0:
            try:
                # create a new thread to initiate the face_check function and pass the current frame
                threading.Thread(target=face_check, args=(frame,)).start()
            except ValueError:
                # if there is no match the program won't crash
                pass
        # increment by 1 frame
        clock += 1

        # Define text properties
        if face_match:
            text = "Match!"
            color = (0, 255, 0)
        else:
            text = "No Match!"
            color = (0, 0, 255)

        # ******************************* Make a background for text prominence *******************************
        # get the text size
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]

        # calculate rectangle coordinates
        text_x = 0
        text_y = 50
        rectangle_start = (text_x, text_y - text_size[1] - 10)  # Adjust for padding
        rectangle_end = (text_x + text_size[0], text_y + 10)

        # draw rectangle background
        cv2.rectangle(frame, rectangle_start, rectangle_end, (0, 0, 0), thickness=cv2.FILLED)

        # put the text on top of the rectangle
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        # *******************************************************************************************************

        # display video using OpenCV
        cv2.imshow("Face Analyser", frame)

    # check for window close event
    key = cv2.waitKey(1)
    # if q pressed or window is closed, the loop ends
    if key == ord("q") or cv2.getWindowProperty("Face Analyser", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

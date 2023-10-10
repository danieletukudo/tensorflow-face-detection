import  os
import time
import uuid
import cv2

image_path = os.path.join("side_images", "images")
number_images = 30
cap = cv2.VideoCapture (0)
for imgnum in range (10) :
    print( 'Collecting image {}'.format (imgnum))
    ret, frame = cap. read ()
    imgname = os.path. join(image_path, f'{str(uuid.uuid1())}.jpg')
    cv2. imwrite (imgname, frame)
    cv2.imshow(' frame', frame)
    time.sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



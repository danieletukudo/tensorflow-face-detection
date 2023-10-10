from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np

facetracker = load_model('model.h5')
#
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     _, frame = cap.read()
#     # frame = frame[50:500, 50:500, :]
#     frame = cv2.resize(frame,(450,450))
#
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     resized = tf.image.resize(rgb, (120, 120))
#
#     yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
#     sample_coords = yhat[1][0]
#
#     if yhat[0] > 0.5:
#         # Controls the main rectangle
#         cv2.rectangle(frame,
#                       tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
#                       tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
#                       (255, 0, 0), 2)
#         # Controls the label rectangle
#         cv2.rectangle(frame,
#                       tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
#                                    [0, -30])),
#                       tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
#                                    [80, 0])),
#                       (255, 0, 0), -1)
#
#         # Controls the text rendered
#         cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
#                                                 [0, -5])),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#
#     cv2.imshow('EyeTrack', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()






img = cv2.imread("download (1).jpeg")
img = cv2.resize(img,(450,450))
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = tf.image.resize(rgb, (120, 120))

yhat = facetracker.predict(np.expand_dims(resized / 255, 0))

sample_coords = yhat[1][0]
# print(sample_coords[:2])
# print(sample_coords[2:])
print( tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)))
cv2.rectangle(img,
              tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                           [0, -30])),
              tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                           [80, 0])),
              (255, 0, 0), -1)

# Controls the text rendered
cv2.putText(img, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                        [0, -5])),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.rectangle(img,
                      tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                      (255, 0, 0), 2)

print(yhat)
cv2.imshow('EyeTrack', img)

cv2.waitKey(0)
# & 0xFF == ord('q'):
#
#     break
# cap.release()
cv2.destroyAllWindows()
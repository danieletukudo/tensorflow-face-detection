import json
import os
import cv2
import numpy as np
import albumentations as alb
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow. keras.models import Model
from tensorflow.keras. layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Add, GlobalMaxPooling2D, Dropout, ReLU
from tensorflow. keras.applications import VGG16


def load_image(x):

    byte_img= tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

train_images = tf.data.Dataset.list_files('aug/train/images/*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
train_images = train_images.map(lambda x: x/255)

test_images = tf.data.Dataset.list_files('aug/test/images/*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
test_images = test_images.map(lambda x: x/255)

val_images = tf.data.Dataset.list_files('aug/val/images/*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
val_images = val_images.map(lambda x: x/255)

def load_labels(label_path):

    with open(label_path.numpy(),'r',encoding="utf-8") as f:
        label = json.load(f)
    return [label['class']],label['bbox']


train_labels = tf.data.Dataset.list_files('aug/train/labels/*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('aug/test/labels/*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files('aug/val/labels/*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
print(len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels))


train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(3000)
val = val.batch(8)
val = val.prefetch(4)



plot = False

if plot==True:
    data_samples = train.as_numpy_iterator()
    res = data_samples.next()
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx in range(4):
        sample_image = res[0][idx]
        sample_coords = res[1][1][idx]

        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                      (255, 0, 0), 2)

        ax[idx].imshow(sample_image)
    plt.show()

vgg = VGG16 (include_top =False)

# vgg.summary()

def build_model () :
    input_layer = Input (shape=(120, 120, 3))
    vgg = VGG16 (include_top= False) (input_layer)
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation ='relu')(f1)
    class2 = Dense(1, activation ='sigmoid') (class1)
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense (2048, activation ='relu')(f2)
    regress2 = Dense(4, activation ='sigmoid') (regress1)
    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])

    return facetracker

facetracker = build_model()
facetracker.summary()


batches_per_epoch = len (train)
Ir_decay = (1./0.75 -1) /batches_per_epoch
opt = tf.keras.optimizers.Adam(learning_rate= 0.0001, decay=Ir_decay)


def localization_loss (y_true, yhat):

    delta_coord = tf.reduce_sum(tf.square (y_true[:,:2] - yhat [:,:2]))
    h_true = y_true[:,3] - y_true[:, 1]
    w_true = y_true[:,2] - y_true[:, 0]
    h_pred = yhat[:,3] - yhat[:, 1]
    w_pred = yhat[ : ,2] - yhat[:,0]
    delta_size = tf.reduce_sum(tf.square (w_true - w_pred) + tf.square (h_true-h_pred))


    return delta_coord + delta_size

classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss


class FaceTracker(Model):
    def __init__(self, eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)

            total_loss = batch_localizationloss + 0.5 * batch_classloss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch

        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)

model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=2, validation_data=val, callbacks=[tensorboard_callback])

facetracker.save('model.h5')



# # fig, ax = plt.subplots(ncols=3, figsize=(20,5))
# #
# # ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
# # ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
# # ax[0].title.set_text('Loss')
# # ax[0].legend()
# #
# # ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
# # ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
# # ax[1].title.set_text('Classification Loss')
# # ax[1].legend()
# #
# # ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
# # ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
# # ax[2].title.set_text('Regression Loss')
# # ax[2].legend()
# #
# # plt.show()
# facetracker = load_model('facetracker.h5')
#
# test_data = test.as_numpy_iterator()
# img =cv2.imread('data/test/images/cbb34950-3d00-11ee-8b3e-dca9048677fe.jpg')
# # re = cv2.resize(img,(120,120))
#
# rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# resized = tf.image.resize(rgb, (120, 120))
# # resized = rgb
# # print(resized)
# yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
#
# # sample_coords = yhat[1][0]
# # test_data = test.as_numpy_iterator()
# # test_sample = test_data.next()
# # print(test_sample[0][1])
# # yhat = facetracker.predict(test_sample[0])
# # print(yhat)
# # print(re.shape)
# # test_sample = test_data.next()
# # print(test_sample[0][0])
# # print(type(test_sample[0][0]))
# # print(test_sample[0][0].shape)
# # print(test_sample[0][0].size)
#
# # yhat = facetracker.predict(re[0])
# #
# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# # for idx in range(4):
# # sample_image = re
# # print(yhat)
# sample_coords = yhat[1][0]
#
# print(tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)))
# print(tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)))
#
#
# cv2.rectangle(img,
#               tuple(np.multiply(sample_coords[:2], [450,720]).astype(int)),
#               tuple(np.multiply(sample_coords[2:], [1280,720]).astype(int)),
#               (255, 0, 0) , 2)
#
# # print(yhat)
# # # print(sample_coords[2:2])
# # # prin
#
# # print(yhat)
# # print(yhat[1][0])
#
# # for idx in range(4):
# sample_image = img
# h,w,_ = img.shape
# print(h,w)
#     # print(idx)
#     # print(yhat[1] [0][idx])
#
# #
# # x1 =  yhat[1][0][0]
# # x2 = yhat[1][0][2]
# #
# # y1 =  yhat[1][0][1]
# # y2 = yhat[1][0][3]
# #     # print(sample_coords)
# # #
# # # print(w,h)
# # # print(x1,x2,y1,y2)
# # cv2.rectangle(img,
# #                           (21,407),  (39,589),
# #                           (255, 0, 0), 2)
#
# cv2.imshow('Face Recognition', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
#     # print(tuple(np.multiply(sample_coords, [120, 120]).astype(int)))
#     #
#     # if yhat[0][0] > 0.5:
#     #         cv2.rectangle(img,
#     #                       tuple(np.multiply(sample_coords, [120, 120]).astype(int)),
#     #                       tuple(np.multiply(sample_coords, [120, 120]).astype(int)),
#     #                       (255, 0, 0), 2)
#     #
#     # plt.imshow(img)
#     #
#     # plt.show()
#
# #
# #     cv2.rectangle(img,
# #                       tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
# #                       tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
# #                       (255, 0, 0), 2)
# #
# #
# # ax[0].imshow(img)
# # plt.show()
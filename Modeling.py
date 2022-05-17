import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, Dropout, Layer, UpSampling2D, Conv2DTranspose, \
    Concatenate
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau


tr_data_dir = os.path.join('data', 'marmot_masked')
train_df = pd.read_csv('data.csv')

tf.keras.backend.clear_session()
dataset = tf.data.Dataset.from_tensor_slices(
    (train_df['image_path'].values, train_df['tablemask_path'].values, train_df['columnmask_path'].values))


def _parse_function(image, mask, colmask):
    """
    Функция загружает изображения в память, изменяет размер и нормализует
    """
    dim = (512, 512)
    image_decoded = tf.io.decode_png(tf.io.read_file(image), channels=3)
    image = tf.cast(image_decoded, tf.float32)
    image = tf.image.resize(image, dim)
    image = tf.cast(image, tf.float32) / 255.0

    mask_decoded = tf.io.decode_png(tf.io.read_file(mask), channels=1)
    mask = tf.cast(mask_decoded, tf.float32)
    mask = tf.image.resize(mask, dim)
    mask = tf.cast(mask, tf.float32) / 255.0

    colmask_decoded = tf.io.decode_png(tf.io.read_file(colmask), channels=1)
    colmask = tf.cast(colmask_decoded, tf.float32)
    colmask = tf.image.resize(colmask, dim)
    colmask = tf.cast(colmask, tf.float32) / 255.0

    mask_dict = {'table': mask, 'column': colmask}

    return image, mask_dict


DATA_SIZE = len(dataset)
VAL_SIZE = 0.1

VAL_LENGTH = int(VAL_SIZE * DATA_SIZE)
TRAIN_LENGTH = int((1 - VAL_SIZE) * DATA_SIZE)

BATCH_SIZE = 1
BUFFER_SIZE = TRAIN_LENGTH

VAL_BUFFER_SIZE = VAL_LENGTH

VALIDATION_STEPS = VAL_LENGTH // BATCH_SIZE

STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset.take(TRAIN_LENGTH)
val = dataset.skip(TRAIN_LENGTH)
train = train.map(_parse_function)
val = val.map(_parse_function)

train_dataset = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val.shuffle(VAL_BUFFER_SIZE).batch(BATCH_SIZE)


# function to display a list of images
def visualize(image_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'Table Mask', 'Column Mask']
    for i in range(len(image_list)):
        plt.subplot(1, len(image_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image_list[i]))
        plt.axis('off')
    plt.show()


for image, mask in train.take(1):
    # print(mask.shape)
    visualize([image, mask['table'], mask['column']])


class TableConvLayer(Layer):
    def __init__(self, name='table'):
        super().__init__(name=name)
        self.conv7 = Conv2D(512, (1, 1), activation='relu', name='conv7table')
        self.upsample_conv7 = UpSampling2D((2, 2))
        self.concat_p4 = Concatenate()
        self.upsample_p4 = UpSampling2D((2, 2))

        self.concat_p3 = Concatenate()
        self.upsample_p3 = UpSampling2D((2, 2))

        self.upsample_p3_2 = UpSampling2D((2, 2))
        self.convtranspose = Conv2DTranspose(3, (3, 3), strides=2, padding='same')

    def call(self, inputs):
        X, pool3, pool4 = inputs
        X = self.conv7(X)
        X = self.upsample_conv7(X)
        X = self.concat_p4([X, pool4])
        X = self.upsample_p4(X)
        X = self.concat_p3([X, pool3])
        X = self.upsample_p3(X)
        X = self.upsample_p3_2(X)
        X = self.convtranspose(X)

        return X


class ColumnConvLayer(Layer):
    def __init__(self, name='column'):
        super().__init__(name=name)
        self.conv7 = Conv2D(512, (1, 1), activation='relu', name='conv7column')
        self.dropout = Dropout(0.8)
        self.conv8 = Conv2D(512, (1, 1), activation='relu', name='conv8column')
        self.upsample_conv8 = UpSampling2D((2, 2))
        self.concat_p4 = Concatenate()
        self.upsample_p4 = UpSampling2D((2, 2))

        self.concat_p3 = Concatenate()
        self.upsample_p3 = UpSampling2D((2, 2))

        self.upsample_p3_2 = UpSampling2D((2, 2))
        self.convtranspose = Conv2DTranspose(3, (3, 3), strides=2, padding='same')

    def call(self, inputs):
        X, pool3, pool4 = inputs
        X = self.conv7(X)
        X = self.dropout(X)
        X = self.conv8(X)
        X = self.upsample_conv8(X)
        X = self.concat_p4([X, pool4])
        X = self.upsample_p4(X)
        X = self.concat_p3([X, pool3])
        X = self.upsample_p3(X)
        X = self.upsample_p3_2(X)
        X = self.convtranspose(X)

        return X


def build_tablenet():
    tf.keras.backend.clear_session()
    input_shape = (512, 512, 3)

    base = VGG19(input_shape=input_shape, include_top=False, weights='imagenet')

    end_layers_list = ['block3_pool', 'block4_pool', 'block5_pool']
    end_layers = [base.get_layer(i).output for i in end_layers_list]

    X = Conv2D(512, (1, 1), activation='relu', name='block6_conv1')(end_layers[-1])
    X = Dropout(0.8)(X)
    X = Conv2D(512, (1, 1), activation='relu', name='block6_conv2')(X)
    X = Dropout(0.8)(X)

    table_branch = TableConvLayer()([X, end_layers[0], end_layers[1]])
    column_branch = ColumnConvLayer()([X, end_layers[0], end_layers[1]])

    model = Model(inputs=base.input, outputs=[table_branch, column_branch], name='TableNetVGG19')
    return model


model = build_tablenet()
# tf.keras.utils.plot_model(model,show_shapes=True,show_layer_names=True)
model.summary()

losses = {
    "table": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    "column": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
}

lossWeights = {"table": 1.0, "column": 1.0}
optim = tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-08)
sc_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='sc_accuracy')

model.compile(optimizer=optim,
              loss=losses,
              metrics=[sc_acc],
              loss_weights=lossWeights)


def render(mask):
    mask = tf.argmax(mask, axis=-1)
    mask = mask[..., tf.newaxis]
    return mask[0]


for image, masks in val_dataset.take(1):
    table_mask, column_mask = masks['table'], masks['column']

pred_tab_mask, pred_col_mask = model.predict(image)
visualize([image[0], render(pred_tab_mask), render(pred_col_mask)])

checkpoint_path = "tablenet_weights/cp_{epoch:04d}_{val_loss:.4f}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

log_dir = "tablenet_logs"

if not os.path.exists(checkpoint_dir):
  os.makedirs(checkpoint_dir)
if not os.path.exists(log_dir):
  os.makedirs(log_dir)

cp_callback = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, monitor='val_loss', save_weights_only=True)

tb_callback = TensorBoard(log_dir)


callbacks = [cp_callback, tb_callback]

model.fit(
    train_dataset, epochs=200,
    steps_per_epoch= STEPS_PER_EPOCH,
    validation_data=val_dataset,
    validation_steps= VALIDATION_STEPS,
    callbacks=callbacks
)






















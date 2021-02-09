import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Confusion Matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for k, m in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(m, k, cm[k, m],
                 horizontalalignment="center",
                 color="white" if cm[k, m] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


RUN_NAME = "run0_" + str(time.time())    # for TensorBoard
RUN_NAME1 = "run0_" + str(time.time())

# GENERATORS
TRAIN_DATAGEN = ImageDataGenerator(
                        rescale=1. / 255.,
                        width_shift_range=0.5,
                        height_shift_range=0.5,
                        fill_mode="nearest",
                        rotation_range=60,
                        zoom_range=0.4,
                        horizontal_flip=True,
                        vertical_flip=True
)

TEST_DATAGEN = ImageDataGenerator(rescale=1.0 / 255.)

TRAIN_GENERATOR = TRAIN_DATAGEN.flow_from_directory(
                        directory=r"E:\THESIS\DATA\TRAIN",
                        target_size=(299, 299),
                        color_mode="rgb",
                        batch_size=16,
                        class_mode="categorical",
                        shuffle=True,
                        seed=42
)

VALID_GENERATOR = TEST_DATAGEN.flow_from_directory(
                        directory=r"E:\THESIS\DATA\VALIDATION",
                        target_size=(299, 299),
                        color_mode="rgb",
                        batch_size=8,
                        class_mode="categorical",
                        shuffle=True,
                        seed=42
)

# CALLBACKS
learn_control = ReduceLROnPlateau(
                        monitor='val_accuracy',
                        patience=5,
                        verbose=1,
                        factor=0.2,
                        min_lr=1e-10)

weights_path = r"C:\Users\user\Desktop\RESULTS\INC_V3\model_weights.h5"
checkpoint = ModelCheckpoint(
                        weights_path,
                        monitor='val_accuracy',
                        verbose=1,
                        save_best_only=True,
                        mode='max')

early_stopper = EarlyStopping(
                        monitor='val_loss',
                        min_delta=0.001,
                        patience=10,
                        verbose=1,
                        mode='min'
)

ten_board = TensorBoard(
                        log_dir="logs\\{}".format(RUN_NAME)
)

# PRE-TRAINED MODEL
base_model = InceptionV3(
                        input_shape=(299, 299, 3),
                        include_top=False,
                        weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False

inputs = Input(shape=(299, 299, 3))

x = base_model(inputs, training=False)

gap_1 = GlobalAveragePooling2D()(x)

dense_1 = Dense(1300, activation='relu')(gap_1)

drop_1 = Dropout(0.30)(dense_1)

output_layer = Dense(2, activation='softmax')(drop_1)

model = Model(inputs, outputs=output_layer)

model.compile(
            optimizer=Adam(lr=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
)

history = model.fit(
            x=TRAIN_GENERATOR,
            steps_per_epoch=(TRAIN_GENERATOR.n//TRAIN_GENERATOR.batch_size),
            validation_data=VALID_GENERATOR,
            validation_steps=(VALID_GENERATOR.n//VALID_GENERATOR.batch_size),
            validation_freq=1,
            epochs=50,
            verbose=2,
            callbacks=[learn_control, checkpoint, early_stopper, ten_board]
)

# FINE TUNING

model.load_weights(weights_path, by_name=False)

# unfreeze Inception blocks from the bottom
for layer in base_model.layers[:249]:
    layer.trainable = False
for layer in base_model.layers[249:]:
    layer.trainable = True

early_stopper1 = EarlyStopping(
                        monitor='val_loss',
                        min_delta=0.001,
                        patience=10,
                        verbose=1,
                        mode='min'
)

ten_board1 = TensorBoard(
                        log_dir="logs\\{}".format(RUN_NAME1)
)

model.compile(
            optimizer=Adam(lr=0.00001),
            loss='binary_crossentropy',
            metrics=['accuracy']
)

history = model.fit(
            x=TRAIN_GENERATOR,
            steps_per_epoch=(TRAIN_GENERATOR.n//TRAIN_GENERATOR.batch_size),
            validation_data=VALID_GENERATOR,
            validation_steps=(VALID_GENERATOR.n//VALID_GENERATOR.batch_size),
            validation_freq=1,
            epochs=50,
            verbose=2,
            callbacks=[learn_control, checkpoint, early_stopper1, ten_board1]
)

# EVALUATE
VALID_GENERATOR1 = TEST_DATAGEN.flow_from_directory(
    directory=r"E:\THESIS\DATA\VALIDATION",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

model.evaluate(
    VALID_GENERATOR1,
    steps=(VALID_GENERATOR1.n//VALID_GENERATOR1.batch_size),
    verbose=1,
    callbacks=[learn_control, checkpoint, early_stopper, ten_board]
)

# PREDICT
TEST_GENERATOR = TEST_DATAGEN.flow_from_directory(
    directory=r"E:\THESIS\DATA\TEST",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

TEST_GENERATOR.reset()

pred = model.predict(
    TEST_GENERATOR,
    steps=(TEST_GENERATOR.n//TEST_GENERATOR.batch_size),
    verbose=1,
    callbacks=[learn_control, checkpoint, early_stopper, ten_board]
)

Y_pred = np.argmax(pred, axis=1)

labels = TRAIN_GENERATOR.class_indices
labels = dict((v, k) for k, v in labels.items())

predictions = [labels[k] for k in Y_pred]

filenames = TEST_GENERATOR.filenames

results = pd.DataFrame({"Filename": filenames,
                        "Predictions": predictions})
results.to_csv(r"C:\Users\user\Desktop\RESULTS\INC_V3\model1_GAP+DENSE+DROPOUT\TLonly_Allfrozen_predictions.csv", index=False)

# plot confusion matrix
cnf_matrix = confusion_matrix(TEST_GENERATOR.classes, Y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names = list(TRAIN_GENERATOR.class_indices.keys())

plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()

accuracy = accuracy_score(TEST_GENERATOR.classes, Y_pred)
print('Accuracy: %f' % accuracy)

precision = precision_score(TEST_GENERATOR.classes, Y_pred)
print('Precision: %f' % precision)

recall = recall_score(TEST_GENERATOR.classes, Y_pred)
print('Recall: %f' % recall)

f1 = f1_score(TEST_GENERATOR.classes, Y_pred)
print('F1 score: %f' % f1)

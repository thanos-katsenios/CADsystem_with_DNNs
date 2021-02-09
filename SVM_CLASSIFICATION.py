import numpy as np
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
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


# datagen construction
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

TEST_DATAGEN = ImageDataGenerator(rescale=1./255.)

# base model
base_model = InceptionV3(
    input_shape=(299, 299, 3),
    include_top=False,
    weights='imagenet')

train_features = np.zeros(shape=(1497, 8, 8, 2048))  # equal to the output of the convolutional base
train_labels = np.zeros(shape=1497)

TRAIN_GENERATOR = TRAIN_DATAGEN.flow_from_directory(
    directory=r"E:\THESIS\DATA\TRAIN",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=16,
    class_mode="binary",
    shuffle=True,
    seed=42
)

# Pass data batches through Inception base
i = 0
for inputs_batch, labels_batch in TRAIN_GENERATOR:
    features_batch = base_model.predict(inputs_batch)
    train_features[(i * TRAIN_GENERATOR.batch_size): ((i + 1) * TRAIN_GENERATOR.batch_size)] = features_batch
    train_labels[(i * TRAIN_GENERATOR.batch_size): ((i + 1) * TRAIN_GENERATOR.batch_size)] = labels_batch
    i += 1
    if (i * TRAIN_GENERATOR.batch_size) >= 1497:
        break

val_features = np.zeros(shape=(355, 8, 8, 2048))  # equal to the output of the convolutional base
val_labels = np.zeros(shape=355)

VALID_GENERATOR = TEST_DATAGEN.flow_from_directory(
    directory=r"E:\THESIS\DATA\VALIDATION",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=8,
    class_mode="binary",
    shuffle=True,
    seed=42
)

# Pass data batches through Inception base
j = 0

for inputs_batch, labels_batch in VALID_GENERATOR:
    features_batch = base_model.predict(inputs_batch)
    val_features[(j * VALID_GENERATOR.batch_size): ((j + 1) * VALID_GENERATOR.batch_size)] = features_batch
    val_labels[(j * VALID_GENERATOR.batch_size): ((j + 1) * VALID_GENERATOR.batch_size)] = labels_batch
    j += 1
    if (j * VALID_GENERATOR.batch_size) >= 355:
        break

test_features = np.zeros(shape=(187, 8, 8, 2048))  # equal to the output of the convolutional base
test_labels = np.zeros(shape=187)

TEST_GENERATOR = TEST_DATAGEN.flow_from_directory(
    directory=r"E:\THESIS\DATA\TEST",
    target_size=(299, 299),
    color_mode="rgb",
    batch_size=8,
    class_mode="binary",
    shuffle=True,
    seed=42
)

# Pass data batches through Inception base
n = 0

for inputs_batch, labels_batch in TEST_GENERATOR:
    features_batch = base_model.predict(inputs_batch)
    test_features[(n * TEST_GENERATOR.batch_size): ((n + 1) * TEST_GENERATOR.batch_size)] = features_batch
    test_labels[(n * TEST_GENERATOR.batch_size): ((n + 1) * TEST_GENERATOR.batch_size)] = labels_batch
    n += 1
    if (n * TEST_GENERATOR.batch_size) >= 187:
        break

svm_features = np.concatenate((train_features, val_features))
svm_labels = np.concatenate((train_labels, val_labels))

X_train = svm_features.reshape(1852, 8*8*2048)
y_train = svm_labels

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [0.0001, 0.001, 0.01, 0.1, 1],
              'kernel': ['rbf', 'poly', 'sigmoid']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_estimator_)

# best model
clf = SVC(C=0.1, cache_size=200,  coef0=0.0, degree=3, gamma=0.0001, kernel='poly',
          max_iter=-1, probability=False, random_state=42, shrinking=True, verbose=False)
clf.fit(X_train, y_train)

test_features = test_features.reshape(187, 8*8*2048)
y_pred = clf.predict(test_features)

# evaluate
print("\nAccuracy score (mean):")
print(np.mean(cross_val_score(clf, X_train, y_train, cv=10)))
print("\nAccuracy score (standard deviation):")
print(np.std(cross_val_score(clf, X_train, y_train, cv=10)))

# plot confusion matrix
cnf_matrix = confusion_matrix(test_labels, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names = list(TRAIN_GENERATOR.class_indices.keys())
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(test_labels, y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(test_labels, y_pred)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(test_labels, y_pred)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(test_labels, y_pred)
print('F1 score: %f' % f1)

'''
ECSE 484 Final Project
Image classifier to distinguish 48 characters of Japanese hiragana alphabet
'''
import tensorflow as tf
from tensorflow.keras import callbacks, models, layers, losses
import numpy as np
import matplotlib.pyplot as plt
import os

# import training data
x_train = np.load('k49_dataset/k49-train-imgs.npz')['arr_0']
y_train = np.load('k49_dataset/k49-train-labels.npz')['arr_0']
# import testing data
x_test = np.load('k49_dataset/k49-test-imgs.npz')['arr_0']
y_test = np.load('k49_dataset/k49-test-labels.npz')['arr_0']

model_save_path = input('Enter path to save/load model:\n')

if os.path.exists(model_save_path) == False:
    print('Invalid path')
    exit()

# import model if previously saved, else train new model
model = 0
model_loaded_from_save = False
try:
    model = models.load_model(model_save_path)
    model_loaded_from_save = True
except:
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # reference: https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342
    model = models.Sequential([
        layers.Conv2D(6, activation='tanh', kernel_size=3, strides=1, input_shape=(28, 28, 1), padding='same'),
        layers.AveragePooling2D(),
        layers.Conv2D(16, activation='tanh', kernel_size=3, strides=1, padding='valid'),
        layers.AveragePooling2D(),
        layers.Conv2D(120, activation='tanh', kernel_size=3, strides=1,  padding='valid'),
        layers.Flatten(),
        layers.Dense(84, activation='tanh'),
        layers.Dense(49, activation='softmax')
    ])

# log model architecture
model.summary()
model.compile(optimizer='adam',
              loss=losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# train model, plot graph, save to directory
def train():
    history = model.fit(x_train, y_train, epochs=15,
                        validation_data=(x_test, y_test))
    model.save(model_save_path)

    # plost accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(model_save_path, 'accuracy.png'))
    plt.show()

    # plot loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(model_save_path, 'loss.png'))
    plt.show()

# decide whether to train model
if model_loaded_from_save:
    retrain = input('Model found. Do you want to train further? Y/n\n')
    if retrain == 'Y':
        train()
else:
    train()

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Accuracy: {:5.2f}%'.format(100 * test_acc))

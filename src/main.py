import os
import random

import matplotlib.pyplot as plt
import splitfolders
import tensorflow as tf
import numpy as np
import seaborn as sn
import pandas as pd

from src.data_exploration import show_grid, class_distribution, size_distribution, show_random_images
from src.globals import TARGET_SIZE, BATCH_SIZE, EPOCHS, PATIENCE, LEARNING_RATE
from src.models.models import xception_model, vgg16_model, mobilenetv2_model, \
    sequential_model1, sequential_model2, sequential_model3

SPLIT_LOC = os.path.join('..', 'output_limited')


def test_models(train_batches, val_batches, test_batches, epochs, batch_size, lr):

    models = [(sequential_model3(lr), 'cnn4')]
    # models.append((sequential_model1(lr), 'cnn3'))
    # models.append((sequential_model2(lr), 'cnn1'))
    # models.append((xception_model(lr), 'xception'))
    # models.append((vgg16_model(lr), 'vgg16'))
    # models.append((mobilenetv2_model(lr), 'mobilenetv2'))

    for i, m in enumerate(models):
        model = m[0]
        lr = model.optimizer.get_config()['learning_rate']
        print(f'Testing model {models[i][1]}. Learning rate: {lr}')
        test_model(model, epochs, batch_size, train_batches, val_batches, test_batches, model_name=models[i][1])


def test_model(model, epochs, batch_size, train_batches, val_batches, test_batches, model_name,
               visualize_wrong=False, save_model=False, conf_matrix=False):
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)

    history = model.fit(train_batches,
                        epochs=epochs,
                        validation_data=val_batches,
                        batch_size=batch_size,
                        callbacks=[es_callback])

    lr = model.optimizer.get_config()['learning_rate']
    plot_history(history, save_to_file=True, filename=f'{model_name}_{epochs}_{batch_size}_{lr}.png')

    score = model.evaluate(test_batches)
    with open(f'{model_name}_{epochs}_{batch_size}_{lr}.txt', 'w+') as f:
        f.write(f'test loss: {score[0]} - test acc:{score[1]}\n')

    if visualize_wrong:
        visualize_wrong_predictions(5, 5, model, test_batches, save_to_file=True, batch_size=batch_size)

    if save_model:
        model.save(f'{model_name}_{epochs}_{batch_size}')

    if conf_matrix:
        confusion_matrix(test_batches, model)


def split_folders():
    images_path = os.path.join('..', 'database_limited')
    splitfolders.ratio(images_path, output=SPLIT_LOC, seed=1337, ratio=(.8, .1, .1),
                       group_prefix=None)  # default values


def one_hot_to_label(one_hot, class_indices):
    decoded = tf.argmax(one_hot, axis=1).numpy()
    class_indices = {value: key for key, value in class_indices.items()}
    result = [class_indices[d] for d in decoded]
    return result


def plot_history(history, save_to_file=False, filename=''):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'val'], loc='upper left')

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'val'], loc='upper left')

    if save_to_file:
        if filename != '':
            plt.savefig(filename)
        else:
            plt.savefig('default.png')
    else:
        plt.show()


def visualize_wrong_predictions(rows, cols, model, test_batches, batch_size, save_to_file=False):
    x_errors = []
    labels = []
    class_indices = test_batches.class_indices
    for it, (x, y) in enumerate(test_batches):
        y = one_hot_to_label(y, class_indices)
        y_pred = one_hot_to_label(model.predict(x), class_indices)
        for i, data in enumerate(x):
            if y[i] != y_pred[i]:
                x_errors.append(data)
                labels.append("Pred: " + y_pred[i] + "\nActual: " + y[i])
        if it >= len(test_batches.filenames) / batch_size:
            break

    rand_nums = random.sample(range(len(x_errors)), rows*cols)
    x_to_disp = [x_errors[i] for i in rand_nums]
    labels_to_disp = [labels[i] for i in rand_nums]

    show_grid(x_to_disp, rows, cols, label_list=labels_to_disp, hpad=2.0, save_fig=save_to_file, filename='wrong_preds.png')


def confusion_matrix(test_batches, model):
    y_pred = np.argmax(model.predict(test_batches), axis=1)
    classes = list(test_batches.class_indices.keys())
    confusion = tf.math.confusion_matrix(test_batches.classes, y_pred)
    f, ax = plt.subplots(figsize=(8, 8))
    sn.heatmap(confusion, annot=True, xticklabels=classes, yticklabels=classes, ax=ax, cmap="rocket_r")
    plt.xlabel("Pred")
    plt.ylabel("Actual")
    plt.title("Macierz błędów")
    plt.show()


if __name__ == "__main__":
    # split_folders()   # One-time function, run this only when first time run

    # Data exploration
    # class_distribution(save_fig=True)
    # size_distribution(most_common=20, save_fig=True)
    # show_random_images(4, 6, save_fig=True)

    # Data generators
    default_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                    shear_range=0.2,
                                                                    zoom_range=0.2,
                                                                    rotation_range=20,
                                                                    brightness_range=(0.5, 1.2),
                                                                    horizontal_flip=True)

    train_batches = train_datagen.flow_from_directory(os.path.join(SPLIT_LOC, 'train'),
                                                      batch_size=BATCH_SIZE,
                                                      target_size=TARGET_SIZE)

    val_batches = default_datagen.flow_from_directory(os.path.join(SPLIT_LOC, 'val'),
                                                      batch_size=BATCH_SIZE,
                                                      target_size=TARGET_SIZE)

    test_batches = default_datagen.flow_from_directory(os.path.join(SPLIT_LOC, 'test'),
                                                       batch_size=BATCH_SIZE,
                                                       target_size=TARGET_SIZE,
                                                       shuffle=False)
    test_models(train_batches, val_batches, test_batches, EPOCHS, BATCH_SIZE, lr=LEARNING_RATE)

    # Test best model and save wrong predictions and confusion matrix
    # model = mobilenetv2_model(0.001)
    # test_model(model, EPOCHS, BATCH_SIZE, train_batches, val_batches, test_batches, 'mobilenetv2_best',
    #           visualize_wrong=True, conf_matrix=True, save_model=False)

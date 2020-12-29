# IMPORTING REQUIRED PACKAGES.
from django.shortcuts import render
import matplotlib.pyplot as plt
import io
import urllib
import base64
import json
import tensorflow as tf
import numpy as np
import random

#Loading the tensorflow ml-model.
model = tf.keras.models.load_model('model.h5')

feature_model = tf.keras.models.Model(
    model.inputs,
    [layer.output for layer in model.layers]
)

#Loading the mnist dataset.
_, (x_test, _) = tf.keras.datasets.mnist.load_data()

x_test = x_test / 255.


def get_prediction():
    '''Random Picking image from test data.'''
    index = np.random.choice(x_test.shape[0])
    image = x_test[index, :, :]
    image_arr = np.reshape(image, (1, 784))
    return feature_model.predict(image_arr), image


def get_uri(fig):
    '''Converting matplotlib figure into base-64 encoded image.'''
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return uri


def home(request):
    '''Rendering home page for random predicting the image.'''
    if request.method == 'POST':
        # handling POST request.
        preds, image = get_prediction()
        final_preds = [p.tolist() for p in preds]
        image = np.reshape(image, (28, 28))
        image = image.tolist()
        preds = final_preds
        ls = []
        fig = plt.figure()
        plt.imshow(image, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        input_image = get_uri(fig)

        for layer, p in enumerate(preds):

            numbers = np.squeeze(np.array(p))
            fig = plt.figure(figsize=(32, 4))

            if layer == 2:
                row = 1
                col = 10
            else:
                row = 2
                col = 16

            for i, number in enumerate(numbers):

                plt.subplot(row, col, i + 1)
                plt.imshow(number * np.ones((8, 8, 3)).astype('float32'))
                plt.xticks([])
                plt.yticks([])

                if layer == 2:
                    plt.xlabel(str(i), fontsize=40)
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.tight_layout()
            # appending each layer of prediction in ls.
            ls.append(get_uri(fig))
        return render(request, 'home.html', {'data': ls, 'input': input_image})

    return render(request, 'home.html', {})

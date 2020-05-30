import os
import sys

PATH = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
PATH_1 = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_PATH = os.path.join(PATH_1, '../templates')
UPLOAD_PATH = os.path.join(PATH_1, '../upload/')
CNN_MODEL = PATH + '/model/cnn_mnist.sav'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
import torch
from torch.autograd import Variable
import getopt
import sys
from cv2 import VideoCapture, waitKey, imshow, getRotationMatrix2D, warpAffine, destroyAllWindows, imread, moveWindow
from scipy.misc import imresize
from os.path import isfile
import numpy as np

from squeeze.model import SqueezeNet

location = 'cpu'
video = 0
model = ''
try:
    opts, args = getopt.getopt(sys.argv[1:], 'hm:l:v:', [
                               'model=', 'location=', 'video='])

except getopt.GetoptError:
    print('python run.py -l <location> -v <video>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('python run.py -l <location> -v <video>')
        sys.exit(0)
    elif opt in ('-m', '--model'):
        model = arg
    elif opt in ('-l', '--location'):
        location = arg
    elif opt in ('-v', '--video'):
        video = arg

net = SqueezeNet(input_shape=(196, 455))
if model == '' or not isfile(model):
    raise NotImplementedError('No model found')
net.load_state_dict(torch.load(model, map_location=location))

steer_wheel = imread('misc/steering_wheel_image.jpg', 0)
rows, cols = steer_wheel.shape
smoothed_angle = 0

cap = VideoCapture(video)
while waitKey(10) != ord('q'):
    _, frame = cap.read()
    img = imresize(frame, (196, 455))
    deg = net(Variable(torch.from_numpy(img.transpose(
        2, 0, 1).reshape((1, 3, 196, 455))).float())).data[0].numpy()[0]
    print('Predicted angle:', str(deg), 'degrees')

    smoothed_angle += 0.2 * pow(abs((deg - smoothed_angle)), 2.0 / 3.0) * (
        deg - smoothed_angle) / abs(deg - smoothed_angle)
    M = getRotationMatrix2D((cols/2, rows/2), -smoothed_angle, 1)
    dst = warpAffine(steer_wheel, M, (cols, rows))
    imshow('wheel', dst)
    moveWindow('wheel', 700, 300)
    imshow('frame', img)
    moveWindow('frame', 700, 80)

cap.release()
destroyAllWindows()

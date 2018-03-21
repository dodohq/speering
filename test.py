import torch
import torch.nn as nn
import numpy as np
import sys
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import getopt

import dataset
from speering import SqueezeNet

test_dataset = dataset.DrivingSet(
    root='./data/driving_dataset', training=False)

batch_size = 100
location = 'cpu'
test_cases = np.floor(np.random.rand(5) * len(test_dataset)).astype(int)
try:
    opts, args = getopt.getopt(sys.argv[1:], 'hl:c:', [
                               'location=', 'testcases='])
except getopt.GetoptError:
    print('python test.py -l <location> -c <testcases>')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('python test.py -l <location> -c <testcases>')
        sys.exit(0)
    elif opt in ('-l', '--location'):
        location = arg
    elif opt in ('-c', '--testcases'):
        test_cases = list(map(lambda x: int(x), arg.split(',')))

est_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

net = SqueezeNet(input_shape=(196, 455))

if 'cuda' in location:
    print('Using:', torch.cuda.get_device_name(torch.cuda.current_device()))
    net.cuda()

try:
    net.load_state_dict(torch.load('squeeze_net.pkl', map_location=location))
except:
    print('ERROR: please make sure you have a model with name `squeeze_net.pkl` in your path')

net.eval()

outputs = []
images = []
labels = []
for c in test_cases:
    image, label = test_dataset[c]
    image = Variable(torch.from_numpy(np.array([image.numpy()])))
    label = Variable(torch.from_numpy(np.array([label.numpy()])))
    if 'cuda' in location:
        image = image.cuda()
        label = label.cuda()

    images.append(image)
    labels.append(label)
    output = net(image)
    outputs.append(output)

f, axarr = plt.subplots(ncols=len(images))
for i in range(len(images)):
    axarr[i].axis('off')
    axarr.title('Output: {0} Label: {1}'.format(outputs[i], labels[i]))
    axarr.imshow(images[i])

plt.show()

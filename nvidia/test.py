import torch
import torch.nn as nn
import numpy as np
import sys
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import getopt

sys.path.append('..')
import dataset
from model import NvidiaNet

test_dataset = dataset.DrivingSet(
    root='../data/driving_dataset', training=False)

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

net = NvidiaNet(input_shape=(196, 455))

if 'cuda' in location:
    print('Using:', torch.cuda.get_device_name(torch.cuda.current_device()))
    net.cuda()

try:
    net.load_state_dict(torch.load('nvidia_net.pkl', map_location=location))
except:
    print('ERROR: please make sure you have a model with name `nvidia_net.pkl` in your path')

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

    images.append(image.data[0].numpy().transpose(1, 2, 0))
    labels.append(label.data[0].numpy()[0])
    output = net(image).data[0].numpy()
    outputs.append(output[0])

f, axarr = plt.subplots(nrows=1, ncols=len(images))
for i in range(len(images)):
    axarr[i].axis('off')
    axarr[i].set_title('Output: %.2f Label: %2f' %
                       (outputs[i], labels[i]), fontsize=7)
    axarr[i].imshow(images[i]/255)

plt.show()

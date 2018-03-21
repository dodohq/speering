import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
import os
import getopt

from speering import SqueezeNet
import dataset

train_dataset = dataset.DrivingSet(
    root='./data/driving_dataset', training=True)

continue_training = False
location = 'cpu'
try:
    opts, args = getopt.getopt(sys.argv[1:], 'hl:c', [
                               'location=', 'continue='])
except getopt.GetoptError:
    print('python train.py -l <location> -c')
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print('python train.py -l <location> -c <testcases>')
        sys.exit(0)
    elif opt in ('-l', '--location'):
        location = arg
    elif opt in ('-c', '--continue'):
        continue_training = True

batch_size = 100
num_epochs = 3

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

net = SqueezeNet(input_shape=(196, 455))
if continue_training and os.path.isfile('squeeze_net.pkl'):
    net.load_state_dict(torch.load(
        'squeeze_net.pkl', map_location=location))
    print('Model loaded!')
if 'cuda' in location:
    print('Using:', torch.cuda.get_device_name(torch.cuda.current_device()))
    net.cuda()

criterion = nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)

        if 'cuda' in location:
            images = images.cuda()
            labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Iter [%d/%d] Loss: %.4f' %
          (i+1, len(train_dataset)//batch_size, loss.data[0]))
        if (i+1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

    # lr decay
    # learning_rate /= 10
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    torch.save(net.state_dict(), 'squeeze_net.pkl')

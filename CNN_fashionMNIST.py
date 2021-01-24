import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

# hyper parameters
EPOCH = 10
BATCH_SIZI = 50
LR = 0.005
DOWNLOAD = False
DEVICE = 'cuda'

# dataset of training data
train_data = torchvision.datasets.FashionMNIST(root='D:\\ML_data,/fashionmnist',
                                               train=True,
                                               transform=torchvision.transforms.ToTensor(),
                                               download=DOWNLOAD,
)

# the image data for TensorDataset must in the form of [B,C,M,H]
train_data_cut = torch.unsqueeze(train_data.data[0:3000], dim=1)
train_labels_cut = train_data.train_labels[0:3000]
train_cut = Data.TensorDataset(train_data_cut, train_labels_cut)

DataLoader = Data.DataLoader(dataset=train_cut,
                             batch_size=BATCH_SIZI,
                             shuffle=True,
                             num_workers=0
                             )
"""
print(train_data.train_data.size())
print(train_data.train_labels.type())
"""
# dataset of evaluation data
test_data = torchvision.datasets.FashionMNIST(root='D:\\ML_data,/fashionmnist',
                                              train=False)
test_x = torch.unsqueeze(test_data.data, dim=1)
test_y = test_data.targets.type(torch.LongTensor)
plt.imshow(test_data.data[9999].numpy(), cmap='gray')

# construction of the CNN


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=6,
                      kernel_size=5,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),)

        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self. fc3 = nn.Linear(84, 10)
        '''
        self.dropout = nn.Dropout(0.25)
        '''
    def forward(self, x):
        x = self.conv1(x.float())
        x = self.conv2(x)
        '''from 2d to 1d
        '''
        x = x.view(-1, 256)
        '''
        x = self.dropout(x)
        '''
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)

        output = self.fc3(x)
        return output


cnn = CNN()
cnn = cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

# training
if __name__ == '__main__':
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(DataLoader):
            batch_x, batch_y = Variable(batch_x.cuda()), Variable(batch_y.cuda())

            print('bx size:', batch_x.data.size(), 'by size:', batch_y.data.size())
            out = cnn(batch_x)
            """train_pred = torch.max(out, dim=1)[1].data.numpy()
            batch_acc = sum(train_pred==batch_y.data.numpy())/BATCH_SIZI"""
            loss = loss_function(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                pred = cnn(test_x.cuda())
                pred_y = torch.max(pred, dim=1)[1]
                pred_y = pred_y.cpu()
                test_y_d = test_y.data.numpy()
                print(sum(pred_y.data.numpy()==test_y_d))
                test_accuracy = sum(pred_y.data.numpy()==test_y_d)/test_y.data.size(0)
                print('epoch: ', epoch, 'prediction: ', pred_y.data.numpy(), 'target: ', test_y_d,
                      'test_accuracy: ', test_accuracy)


def train(training_data):
    if __name__ == '__main__':
        if DEVICE == 'cuda':
            for epoch_ in range(EPOCH):
                for step_, (batch_xx, batch_yy) in enumerate(training_data):
                    batch_xx, batch_yy = Variable(batch_xx.cuda()), Variable(batch_yy.cuda())
                    optimizer.zero_grad()
                    outt = cnn(batch_xx)
                    loss_ = loss_function(outt, batch_yy)
                    loss_.backward()
                    optimizer.step()
                    if step_ % 50 == 0:
                        pre_ = torch.max(outt, dim=1)[1].cpu().numpy()
                        train_ac = sum(pre_==batch_yy.cpu().numpy()) / BATCH_SIZI
                        print('epoch: ', epoch_, 'step:', step_, 'training acc: ', train_ac)



# play with the prediction model

round = input('plz give the number of rounds that u wanna play')
round = int(round)
for r in range(round):
    index = input('plz give an index of fashion items, in the range of 0-9999 ')
    index = int(index)
    plt.imshow(test_data.data[index].numpy(), cmap='gray')
    plt.pause(5)
    plt.close()
    game_output = cnn(torch.unsqueeze(test_x.data[index], dim=1).cuda())
    game_output = game_output.cpu()
    game_predict = torch.max(game_output, dim=1)[1].data.numpy()
    if game_predict == 0:
        print('Top')
    elif game_predict == 1:
        print('Trousers')
    elif game_predict == 2:
        print('Pullover')
    elif game_predict == 3:
        print('Dress')
    elif game_predict == 4:
        print('Coat')
    elif game_predict == 5:
        print('Sandals')
    elif game_predict == 6:
        print('Shirt')
    elif game_predict == 7:
        print('Sneaker')
    elif game_predict == 8:
        print('Bag')
    elif game_predict == 9:
        print('Ankle boots')

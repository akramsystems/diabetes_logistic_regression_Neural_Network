import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split

#fail or pass 0 or 1 logistic

xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)


#Split into training and testing
x_data = xy[:,0:-1]

y_data = xy[:,[-1]]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

x_data_train = torch.from_numpy(x_train)

y_data_train = torch.from_numpy(y_train)



# print(x_data_train.data.shape)
# print(y_data_train.data.shape)


#
#	1. Design Model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 30) #   now we have multiple layers 8 inputs go to hidden layer of 6 nodes
        self.l2 = torch.nn.Linear(30, 10) #   6 nodes to 4 nodes
        self.l3 = torch.nn.Linear(10, 1) #   4 to one node which will be our output layer
        self.sigmoid = torch.nn.Sigmoid()   #   since we have binary data we make the output 
                                            #   from our last layer either a one or zero
    def forward(self, x):
        # use activation function on the output of each layer
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred


    
# our model
model = Model()

#	2. Construct loss and optimizer (select from PyTorch API)

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=8)
 #stochastic gradient decent lr = learning rate


# Training loop
for epoch in range(10):
        # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data_train)
    # Compute and print loss
    loss = criterion(y_pred, y_data_train)
    perentage_loss = loss.item()
    print(epoch, perentage_loss)
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#	After Training



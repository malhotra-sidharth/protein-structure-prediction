import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.helper import  Extract
import math


class FCNN(nn.Module):
  def __init__(self, window_size=10, hidden_size=20):
    super(FCNN, self).__init__()

    self.linear1 = torch.nn.Linear(20*window_size, hidden_size)
    self.linear2 = torch.nn.Linear(hidden_size, window_size)

  def forward(self, x):
    out_1 = self.linear1(x)
    h1_relu = F.relu(out_1)
    out_2 = self.linear2(h1_relu)
    y_pred = F.relu(out_2)

    return y_pred



class TrainFCNN:
  def __init__(self, window_size=10, hidden_size=20):
    self.extractor = Extract()
    self.model = FCNN(window_size, hidden_size)
    self.window_size = window_size


  def trainNN(self, one_hot_encoded_df_list, batch_size=32, num_epochs=5, logging=False):
    batch_size = 32
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    print('Test-Train extraction started..')
    allX, allY = self.extractor.get_training_data(one_hot_encoded_df_list, self.window_size, logging)
    print("Test-Train data extracted. Initializing model training...")
    xd, yd = allX.shape

    split_indices = [i for i in range(batch_size, xd, batch_size)]

    batchXlist = np.split(allX, split_indices)[:-1]
    batchYlist = np.split(allY, split_indices)[:-1]

    num_batches = math.ceil(xd / batch_size) - 1
    batches = [(batchXlist[i], batchYlist[i]) for i in range(num_batches)]

    for epoch in range(num_epochs):
      # shuffle dataset
      np.random.shuffle(batches)
      total_epoch_loss = 0
      for i in range(num_batches):
        batchX, batchY = batches[i]
        tensorX = torch.Tensor(batchX, dtype=torch.float).view(batch_size, 20*self.window_size)
        tensorY = torch.Tensor(batchY, dtype=torch.float).view(batch_size, self.window_size)

        # Compute Loss
        y_pred = self.model(tensorX)
        loss = torch.sqrt(criterion(y_pred, tensorY))

        # Zero gradients, perform backward pass, up(date weights
        optimizer.zero_grad()
        loss.backward()  # @TODO is this correct?
        optimizer.step()
        total_epoch_loss += loss.item()
      avg_epoch_loss = total_epoch_loss / num_batches
      print("Epoch: {} Current Loss: {} Avg Loss: {}".format(epoch + 1, loss.item(), avg_epoch_loss))

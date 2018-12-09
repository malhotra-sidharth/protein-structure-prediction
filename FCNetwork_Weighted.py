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
    #self.linear15 = torch.nn.Linear(hidden_size, hidden_size)
    self.linear2 = torch.nn.Linear(hidden_size, window_size)

  def forward(self, x):
    out_1 = self.linear1(x)
    h1_relu = F.relu(out_1)
    #out_15 = self.linear15(h1_relu)
    #h15_relu = F.relu(out_15)
    out_2 = self.linear2(h1_relu)
    y_pred = F.relu(out_2)

    return y_pred


class TrainFCNN:
  def __init__(self, window_size=10, hidden_size=20, lr=1e-3, weighted='nw'):
    """
      weighted: string 
        'nw' for non weighted
        'gauss' for gaussian weighted
        'center' for center weighted
    """
    self.extractor = Extract()
    self.model = FCNN(window_size, hidden_size)
    self.window_size = window_size
    if weighted == 'center':
      self.criterion = self.myCenteredLoss
    elif weighted == 'gauss':
      self.criterion = self.myWeightedLoss
    else:
      self.criterion = torch.nn.MSELoss(reduction='sum')
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
    self.trainX = None
    self.trainY = None
    self.testdfX = None
    self.testdfY = None
    self.xd = None
    self.yd = None
    # if torch.cuda.is_available():
      # self.model.cuda()
      # print("Using CUDA.")
      
  def myWeightedLoss(self, outputs, labels):
    norm_input = lambda j: ((j-(self.window_size-1)/2)/ self.window_size) * 10
    gauss = lambda j: math.exp(-(norm_input(j))**2/3**2) 
    w = torch.tensor([gauss(i) for i in range(0, self.window_size)],dtype = torch.float)
    w_sum = torch.sum(w,dim=0)
    w = w/w_sum
    w = w.view(self.window_size,1)
    residuals = outputs-labels
    square_resid = torch.mul(residuals,residuals)
    return torch.sum(torch.mm(square_resid,w))

  def myCenteredLoss(self, outputs, labels):
    centered = lambda j: 1 if j == int(self.window_size/2) else 0.01    
    w = torch.tensor([centered(i) for i in range(0, self.window_size)],dtype = torch.float)
    w = w.view(self.window_size,1)
    residuals = outputs-labels
    square_resid = torch.mul(residuals,residuals)
    return torch.sum(torch.mm(square_resid,w))
#    square_resid = square_resid[:, int(self.window_size/2)]
#    return torch.sum(square_resid)
    
  def loadTestTrainData(self, one_hot_encoded_df_train_list, one_hot_encoded_df_test_list, max_size=40000, logging=False):
    if logging:
        print('Train data extraction started.') 
    self.trainX, self.trainY = self.extractor.get_training_data(one_hot_encoded_df_train_list, self.window_size,max_size, logging)
    self.xd, self.yd = self.trainX.shape
    
    if logging:
        print('Test data extraction started.') 
    self.testdfX, self.testdfY = self.extractor.get_test_data(one_hot_encoded_df_test_list, self.window_size, max_size, logging)
    
    if logging:
        print("Test and Train data extracted.")

  def trainNN(self, batch_size=32, num_epochs=5, logging=False, save_path=None, save_after_epochs=None):
    split_indices = [i for i in range(batch_size, self.xd, batch_size)]

    batchXlist = np.split(self.trainX, split_indices)[:-1]
    batchYlist = np.split(self.trainY, split_indices)[:-1]

    num_batches = math.ceil(self.xd / batch_size) - 1
    batches = [(batchXlist[i], batchYlist[i]) for i in range(num_batches)]

    for epoch in range(num_epochs):
      # shuffle dataset
      np.random.shuffle(batches)
      total_epoch_loss = 0
      for i in range(num_batches):
        batchX, batchY = batches[i]
        tensorX = torch.tensor(batchX, dtype=torch.float).view(batch_size, 20*self.window_size)
        tensorY = torch.tensor(batchY, dtype=torch.float).view(batch_size, self.window_size)

        # if torch.cuda.is_available():
          # tensorX.cuda()
          # tensorY.cuda()
        # else:
          # print("CUDA fail")
        # print(tensorX.type)

        # Compute Loss
        y_pred = self.model(tensorX)
        loss = torch.sqrt(self.criterion(y_pred, tensorY))

        # Zero gradients, perform backward pass, up(date weights
        self.optimizer.zero_grad()
        loss.backward()  # @TODO is this correct?
        self.optimizer.step()
        total_epoch_loss += loss.item()
      avg_epoch_loss = total_epoch_loss / num_batches
      
      if save_after_epochs != None and epoch%save_after_epochs == 0:
        torch.save(self.model.state_dict(), save_path)
      
      if logging and (epoch%5==0):
        print("Epoch: {} Current Loss: {} Avg Loss: {}".format(epoch + 1, loss.item(), avg_epoch_loss))


  def predict(self, inputdfX, inputdfY, batch_size = 32, start = 0, single_protein = False):

    #Loop through test proteins
    protein_list = []
    for i in range(start,len(inputdfX)):
      testX = inputdfX[i]
      testY = inputdfY[i]
      
      xd, xy = testX.shape
      
      split_indices = [i for i in range(batch_size, xd, batch_size)]

      batchXlist = np.split(testX, split_indices)[:-1]
      batchYlist = np.split(testY, split_indices)[:-1]

      num_batches = math.ceil(xd / batch_size) - 1
      batches = [(batchXlist[i], batchYlist[i]) for i in range(num_batches)]

      total_loss = 0
      protein_pred_tensor = torch.Tensor() 
      protein_true_tensor = torch.Tensor()
      for i in range(num_batches):
        batchX, batchY = batches[i]
        tensorX = torch.tensor(batchX, dtype=torch.float).view(batch_size, 20*self.window_size)
        tensorY = torch.tensor(batchY, dtype=torch.float).view(batch_size, self.window_size)

        # if torch.cuda.is_available():
          # tensorX.cuda()
          # tensorY.cuda()

        # Compute Loss
        y_pred = self.model(tensorX)
        criteria = torch.nn.MSELoss(reduction='sum')
        loss = torch.sqrt(criteria(y_pred, tensorY))
        total_loss += loss
        y_pred_middle = y_pred[:,int(self.window_size/2)]
        y_true_middle = tensorY[:,int(self.window_size/2)]
        protein_pred_tensor=torch.cat((protein_pred_tensor,y_pred_middle))
        protein_true_tensor=torch.cat((protein_true_tensor,y_true_middle))
      avg_loss = total_loss / xd
      protein_list.append((avg_loss,protein_pred_tensor.detach().numpy(),protein_true_tensor.detach().numpy()))
      if single_protein:
        break;
    return protein_list
    
  def predict_on_test_data(self, batch_size = 32, start = 0, single_protein = False):
    return self.predict(self.testdfX, self.testdfY, batch_size, start, single_protein)

  def predict_on_outside_data(self, outside_data_one_hot_df_list, batch_size = 32, start = 0, single_protein = False, max_size=40000, logging = False):
    if logging:
      print("Seperating Labels")
    outsideX, outsideY = self.extractor.get_test_data(outside_data_one_hot_df_list, self.window_size, max_size, logging)
    
    if logging:
      print("Running Predictions")
    return self.predict(outsideX, outsideY, batch_size, start, single_protein)
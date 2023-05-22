# -*- coding: utf-8 -*-

import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.8 # just random number
bias = 0.3 # just random number
X = torch.arange(start=0,end=1,step=0.02).unsqueeze(dim=1)# getting data between 0 to 1 with 0.02 step in 1 dimension

y = weight*X +bias # straight line formula or linear regresion formula

split =int(0.8 * len(X)) # splitting data in 80% to 20% ratio
X_test,y_test  =X[split:],y[split:] # getting 80% data from x
X_train,y_train = X[:split],y[:split]# getting 20% data from x

def plot_prediction(train_data=X_train,train_label=y_train,test_data=X_test,test_label =y_test,prediction = None):
  plt.figure(figsize=(6,5))# setting size of graph
  plt.scatter(train_data,train_label,c="b",s=6,label="training data")# drawing  graph for training data vs training output
  plt.scatter(test_data,test_label,c="r",s=6,label="testing data")# drawing  graph for testing data vs testing output
  if prediction is not None:
    plt.scatter(test_data,prediction,c="g",s=6,label="predicted data") #drawing  graph  for testing data vs predicted output
  plt.show()
plot_prediction()#visualizing data

class LinearReg(nn.Module):# making a nueral network
  def __init__(self):
    super().__init__()
    self.LinearLayer = nn.Linear(in_features=1,out_features=1)# getting random weight and bias
   
  def forward(self,x):
    return self.LinearLayer(x) # forword functino will be called internally it apply linear regresion formula and returns predicted data

model_0 = LinearReg()
with torch.inference_mode():#here requires_grad =False i.e. gradient not being calculated
  y_preds = model_0(X_test)
plot_prediction(prediction=y_preds)# and without  calculation of gradient we are able give proper prediction

loss_fn =nn.L1Loss()# this function helps to calculate mean average between predcted value and result
optimiser = torch.optim.SGD(params = model_0.parameters(),lr=0.02)# help to calculate gradient and update the parameters

torch.manual_seed(42)
epochs =10000
y_pred = 0
for epoch in range(epochs):
  model_0.train() #sets y in training mode
  y_pred = model_0(X_train) # predicting the output
  loss = loss_fn(y_pred,y_train) # calculating mean avg between the y_pred and y_train
  optimiser.zero_grad() # setting grad to zero so that it does not accumulate after every iteration
  loss.backward() # calculate gradient of  with respect to parameter
  optimiser.step()# updates model parameter based on gradient
  model_0.eval()# evalution mode (no gradient store)

with torch.inference_mode():
  y_preds = model_0(X_test)
plot_prediction(prediction=y_preds)


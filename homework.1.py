import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset,DataLoader

def load_csv(csv_path):
   df=pd.read_csv(csv_path)
   print(f'{csv_path}:shape={df.shape}')
   labels=df['label'].values.astype(np.int64)
   pixels=df.drop(columns=['label']).values.astype(np.float32)/255.0
   images=pixels.reshape(-1,1,28,28)
   return TensorDataset(torch.from_numpy(images),torch.from_numpy(labels))

train_dataset = load_csv('/home/lhlbw/homework/data/mnist_train.csv')
test_dataset  = load_csv('/home/lhlbw/homework/data/mnist_test.csv')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)


class SimpleCNN(nn.Module):
   def __init__(self):
       super(SimpleCNN,self).__init__()
       self.conv1=nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
       self.conv2=nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
       self.conv3=nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
       self.fc1=nn.Linear(128*3*3,512)
       self.fc2=nn.Linear(512,10)
       
   def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.avg_pool2d(x,2)
        x=F.relu(self.conv2(x))
        x=F.avg_pool2d(x,2)
        x=F.relu(self.conv3(x))
        x=F.avg_pool2d(x,2)
        x=x.view(-1,128*3*3)
        x=F.relu(self.fc1(x))
        return self.fc2(x)
device=torch.device('cuda')
model=SimpleCNN().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

for epoch in range(10):
   model.train() 
   total_loss=0.0 
   for  images,labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        loss=criterion(model(images),labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        total_loss+=loss.item()
   print(f'Epoch [{epoch+1}/10],Loss:{total_loss/len(train_loader):.5f}')


model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _,predicted=torch.max(model(images).data,1)
        total +=labels.size(0)
        correct +=(predicted ==labels).sum().item()
        
accuracy=100*correct/total
print(f'Test Accuracy:{accuracy:.3f}%')    
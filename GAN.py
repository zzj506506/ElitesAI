
# import d2l
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
from torch import nn
import numpy as np
from torch.nn import init

X=np.random.normal(size=(1000,2))
A=np.array([[1,2],[-0.1,0.5]])
b=np.array([1,2])
data=X.dot(A)+b

# d2l.set_figsize((3.5,2.5))
# d2l.plt.scatter(data[:100,0].asnumpy(),data[:100,1].asnumpy())

plt.figure(figsize=(3.5,2.5))
plt.scatter(data[:100,0],data[:100,1])
plt.show()

print("The covariance matrix is\n%s" % np.dot(A.T, A))

batch_size=8
data_iter=DataLoader(data,batch_size=batch_size)

class net_G(nn.Module):
    def __init__(self):
        super(net_G,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(1,2),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.model(x)
        return x

class net_D(nn.Module):
    def __init__(self):
        super(net_D,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(1,5),
            nn.Tanh(),
            nn.Linear(5,3),
            nn.Tanh(),
            nn.Linear(3,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x=self.model(x)
        return x

# net_G=nn.Sequential(nn.Linear(1,2))
# net_D=nn.Sequential(nn.Linear(1,5),
#                     nn.Tanh(),
#                     nn.Linear(5,3),
#                     nn.Tanh(),
#                     nn.Linear(3,1))

def update_D(X,Z,net_D,net_G,loss,trainer_D):
    batch_size=X.shape[0]
    ones=np.ones((batch_size,),ctx=X.contet)
    zeros = np.zeros((batch_size,), ctx=X.contet)
    real_Y=net_D(X)
    fake_X=net_D(Z)
    fake_Y=net_D(fake_X)
    loss_D=(loss(real_Y,ones)+loss(fake_Y,zeros))/2
    loss_D.backward()
    trainer_D.step()
    return float(loss_D.sum())

def update_G(Z,net_D,net_G,loss,trainer_G):
    batch_size=Z.shape[0]
    ones=np.ones((batch_size,),ctx=Z.context)
    fake_X=net_G(Z)
    fake_Y=net_D(fake_X)
    loss_G=loss(fake_Y,ones)
    loss_G.backward()
    trainer_G.step()
    return float(loss_G.sum())
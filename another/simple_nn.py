import numpy as np
import torch
import torch.nn.functional
import matplotlib.pyplot as plt

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=(x**2)+0.2*torch.rand(x.size())
# y=x.pow(2)+0.2*torch.rand(x.size())

plt.scatter(x.data.numpy(),y.data.numpy())
plt.show()

class simple_net(torch.nn.Module):
    def __init__(self,n_in,n_hid,n_out):
        super(simple_net, self).__init__()
        self.hidden1=torch.nn.Linear(n_in,n_hid)
        # self.hidden2 = torch.nn.Linear(n_hid, n_hid)
        self.out=torch.nn.Linear(n_hid,n_out)

    def forward(self,x):
        x=self.hidden1(x)
        x=torch.nn.functional.relu(x)
        # x = self.hidden2(x)
        # x = torch.nn.functional.relu(x)
        x=self.out(x)
        return x

net=simple_net(1,10,1)
# print(net)

optimizer=torch.optim.SGD(net.parameters(),lr=0.6)
loss_func=torch.nn.MSELoss()
#
plt.ion()
for epoch in range(1000):
    predict=net.forward(x)
    loss=loss_func(predict,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), predict.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'Loss=%.4f'%loss.data.numpy(),
                 fontdict={'size':20,'color':'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()







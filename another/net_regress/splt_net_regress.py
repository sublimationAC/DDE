# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 08:36:24 2019

@author: Pavilion
"""

def train_net_one(train_data_input,train_data_output,name):

    
    
    train_data_input=train_data_input.astype(np.float32)
    train_data_output=train_data_output.astype(np.float32)
    
    num_input=train_data_input.shape[1]
    num_output=train_data_output.shape[1]

    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    print(torch.cuda.device_count())
#    torch.cuda.set_device(6,7)
    
    net=dde_net_fc(num_input,num_input*5,num_output).cuda()
    # print(net)
    
    LR=0.01
    optimizer=torch.optim.SGD(net.parameters(),lr=LR)
#    optimizer= torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
    loss_func=torch.nn.MSELoss()
    #
    
    
# =============================================================================
#     x=torch.tensor(train_data_input)
#     y=torch.tensor(train_data_output)
# =============================================================================

    train_data_num=train_data_input.shape[0]  #10000
    x=torch.Tensor(train_data_input[:train_data_num]).cuda()
    y=torch.Tensor(train_data_output[:train_data_num]).cuda()
    
    torch_dataset = Data.TensorDataset(x, y)
    BATCH_SIZE=20000
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    
    
    loss_plt=[]
    max_epoch=100
    for epoch in range(max_epoch):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            
            predict=net.forward(batch_x)
            loss=loss_func(predict,batch_y)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            loss_plt.append(loss.data.cpu().numpy())
            print('epoch: %d '% epoch ,'step: %d '% step,'Loss=%.4f'%loss_plt[-1])
# =============================================================================
#         if epoch % 5 ==0:
#             print('epoch: %d '% epoch ,'Loss=%.4f'%loss.data.numpy())
# =============================================================================
    torch.save(net, name+'.pkl')  # save entire net
    torch.save(net.state_dict(), name+'params.pkl')   # save only the parameters          
            
    plt.ion()
    plt.cla()
    plt.scatter(np.linspace(0,len(loss_plt),num=len(loss_plt)), loss_plt)
    plt.plot(np.linspace(0,len(loss_plt),num=len(loss_plt)), loss_plt,'r-',lw=1)
    plt.ioff()
    plt.show()

  


def train_net_splt(train_data_input,train_data_output):
    num_exp=46
    num_land=73
    num_angle=3
    num_tslt=3
        
    
    train_data_output_angle=train_data_output[:,0:num_angle]
    train_data_output_tslt=train_data_output[:,num_angle:num_angle+num_tslt]
    train_data_output_exp=train_data_output[:,num_angle+num_tslt:num_angle+num_tslt+num_exp]
    train_data_output_dis=train_data_output[:,num_angle+num_tslt+num_exp:num_angle+num_tslt+num_exp+num_land*2]
    
    train_net_one(train_data_input,train_data_output_angle,'../mid_data/splt_7_17_cf3_norm/net_angle')
    train_net_one(train_data_input,train_data_output_tslt,'../mid_data/splt_7_17_cf3_norm/net_tslt')
    train_net_one(train_data_input,train_data_output_exp,'../mid_data/splt_7_17_cf3_norm/net_exp')
    train_net_one(train_data_input,train_data_output_dis,'../mid_data/splt_7_17_cf3_norm/net_dis')
    
    
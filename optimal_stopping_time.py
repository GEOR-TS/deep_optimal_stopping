import torch
import time
import numpy as np
from torch import nn
from torch.utils import data
from matplotlib import pyplot as plt
#from d2l import torch as d2l

x=torch.arange(-8.0,8.0,0.1,requires_grad=True)
y=torch.relu(x)

K=10
r=0.15
dim=5
N=10
T=15
executive_price=2
tau=torch.zeros(N)
#print(tau)
I=2*torch.ones(N)
#print(I)

s_0=2*torch.ones(dim)
delta=0.1*torch.ones(dim)
sigma=0.2*torch.ones(dim)
corr=0*torch.ones(dim,dim)

all_determine_func=[1]

def Time(n,N):   #将时间长度T给N等分
    t=n*T / N
    return t

#print(time(1,10))

def std_Brown_Motion_path(expiry_time):     #生成标准布朗运动样本路径
    Z_walk = torch.randn(expiry_time)
    B_1=torch.zeros(expiry_time+1)
    for i in range(expiry_time):
        B_1[i+1]=B_1[i]+Z_walk[i]
    return B_1


def multi_sample_Brown_Motion(num_sample,expiry_time,initial_value):   #生成多个布朗运动路径
    B = initial_value*torch.ones(num_sample, expiry_time + 1)
    for k in range(num_sample):
        B[k,:] = B[k,:]+std_Brown_Motion_path(expiry_time)
#        plt.plot(torch.arange(0, N + 1).detach(), B[k, :].detach()) # 把各个布朗运动样本路径画出来
    return B

#B_l=multi_sample_Brown_Motion(5,10,0)
#B_l=multi_sample_Brown_Motion(5,10)
#print(B_l)
#plt.plot(torch.arange(0, 10 + 1).detach(),B_l[0, :].detach())
#plt.show()
#print(B)
#plt.show()

def high_dim_asset_price_path(s_0,delta,sigma,corr,K,dim,N):  #生成多个高维资产价格样本路径
    S=torch.ones(K,dim,N+1)
    for k in range(K):
        for d in range(dim):
            B = multi_sample_Brown_Motion(K,N,0)
            S[k,d,:]=s_0[d]*S[k,d,:]
            for i in range(N):
                S[k, d,i+1] = S[k,d,0] * torch.exp((r - delta[d] - ((sigma[d] ** 2) / 2)) * (Time(i+1,N)) + sigma[d] * B[k, i + 1])
#    print(S)
    return S


#S_0=2
#S=S_0*torch.ones(K,N+1)

def plot_asset_price_path(S,K,dim,initial_point,N):  #对多个高维资产价格样本路径进行作图，不同的分量对应不同的图
    for d in range(dim):
        for k in range(K):
            for i in range(N):
                plt.plot(torch.arange(initial_point, N + 1+initial_point).detach(), S[k,d,:].detach(),color='gray')
        plt.show()
    return 1

#plot_asset_price_path(S,K,dim,N)

#print(S)
#print(corr)

def payoff(r,execute_price,n,N,s):  #n时刻的回报函数
    g=torch.exp(torch.tensor(-r*Time(n,N)))*torch.relu(torch.max(s[:,n]-execute_price))
    return g

#print(S[0,:,5]-0.1)
#print(torch.exp(torch.tensor(-r*time(5,N)))*torch.max(S[0,:,5]-0.1))
#print(payoff(r,0.1,5,N,S[0,:,:]))

def continuation_brown_motion_path(B,k,K,n,N,J):   #某一样本点的多个布朗运动延续样本路径
    B_n =  multi_sample_Brown_Motion(J,N-n,B[k-1,n])
#    plt.plot(torch.arange(0, N + 1-n).detach(), B_n[0, :].detach())
#    plt.plot(torch.arange(0, N + 1-n).detach(), B_n[1, :].detach())
#    plt.plot(torch.arange(0, N + 1 - n).detach(), B_n[2, :].detach())
#    plt.show()
    return B_n

def contiuation_high_dim_asset_path(S,delta,sigma,corr,dim,k,K,n,N,J):  #某一样本点的价格过程的延续样本路径
    S_n= high_dim_asset_price_path(S[k-1,:,n],delta,sigma,corr,J,dim,N-n)
#    plot_asset_price_path(S_n,J,dim,n,N-n)
    return S_n

#B=multi_sample_Brown_Motion(K,N,0)
#B_n=continuation_brown_motion_path(B,3,K,4,N,5)
#S=high_dim_asset_price_path(s_0,delta,sigma,corr,K,dim,N)
#S_n=contiuation_high_dim_asset_path(S,delta,sigma,corr,dim,5,K,4,N,5)

def continuation_value():  #计算延续路径价值
    
    return 1

class stopping_determine_model(nn.Module):     #构建两层神经网络类
    def __init__(self,n,dim,q_1):
        super(stopping_determine_model, self).__init__()
        self.n=n
        self.dim=dim
        self.q_1=q_1
        self.net=nn.Sequential(nn.Linear(self.dim,self.q_1),nn.ReLU(),nn.Linear(self.q_1,1),nn.LogSigmoid())

    def forward(self,x):
        x=self.net(x)
#        x.squeeze(-1)
        return x

def calcu_optimal_time(S_k,all_determine_func,n,N):  #计算n-th的最优停时
    if n==N:
        t=N
        return t
    else:
        t=0
        for m in range(N-n+1):
            if m==0:
                t+=n*all_determine_func[N-n]
            else:
                multiple=1
                for j in range(m):
                    multiple*=(1-all_determine_func[N-n-j].forward(S_k[:,n+j]))
                t+= ((n+m)*all_determine_func[m].forward(S_k[:,n+m])*multiple)
        return t

def n_k_approxi_expect_reward(S,k,K,n,N,all_determine_func):   #计算第n时刻k样本的回报估计
    if n == N:
        reward=payoff(r,executive_price,n,N,S[k-1,:,:])
        return reward
    else:
        reward=payoff(r,executive_price,n,N,S[k-1,:,:])*all_determine_func[N-n].forward(S[k-1,:,n])+payoff(r,executive_price,calcu_optimal_time(S[k-1,:,:],all_determine_func,n+1,N),N,S[k-1,:,:])*(1-all_determine_func[N-n].forward(S[k-1,:,n]))
        return reward

def n_average_approxi_expect_reward(S,K,n,N,all_determine_func):   #第n时刻的平均回报估计
    sum_reward=0
    for k in range(1,K+1):
        sum_reward+=n_k_approxi_expect_reward(S,k,K,n,N,all_determine_func)
    ave_reward=(1.0/K)*sum_reward
    return ave_reward

def n_train_func(all_determine_func,lr,S,K,dim,n,N,q_1,batch_size=5,num_epochs=5):  #n时刻决策函数的训练（N除外）
    data_iter=data.DataLoader(data.TensorDataset(S),batch_size,shuffle=True)
    target_train_obj=stopping_determine_model(n,dim,q_1)
    target_train_net=target_train_obj.net
    optimizer=torch.optim.SGD(target_train_net.parameters(),lr=lr)
    loss=nn.L1Loss()
    all_determine_func.append(target_train_obj)

    rwd=[n_average_approxi_expect_reward(S[0,:,:], batch_size, n, N, all_determine_func)]

    for p in range(num_epochs):
#       start=time.time()
#        for batch_index,S_k in enumerate(data_iter):
#            reward=n_average_approxi_expect_reward(S_k, batch_size, n, N, all_determine_func)
#            l=loss((1/reward),torch.tensor(0))
#            optimizer.zero_grad()
#            l.backward()
#            optimizer.step()
#
#            if (batch_index + 1) * batch_size % 100 == 0:
#                rwd.append(n_average_approxi_expect_reward(S_k, batch_size, n, N, all_determine_func))
#        print('reward(loss): %f, %f sec per epoch'%(rwd[-1],time.time()-start))

        for index, S_k in enumerate(data_iter):
            print('index is')
            print(index)
#        print('\n')
            S_k_inlist = S_k[0]
            reward = n_average_approxi_expect_reward(S_k_inlist, batch_size, n, N, all_determine_func)
            l = loss((1 / reward), torch.tensor(0))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            rd = n_average_approxi_expect_reward(S_k_inlist, batch_size, n, N, all_determine_func)
            print('reward(loss): %f, %f sec per epoch \n' % (rd, time.time() - start))
            if (index + 1) * batch_size % 100 == 0:
                rwd.append(n_average_approxi_expect_reward(S_k_inlist, batch_size, n, N, all_determine_func))

    plt.plot(np.linspace(0, num_epochs, len(rwd)),rwd)
    plt.xlabel('epoch')
    plt.ylabel('reward(loss)')
    plt.show()
    return 1

def optimize_all_determine_func(S,K,dim,N,q_1,all_determine_func):   #倒向迭代（循环）训练所有时刻的决策函数
    F_N = stopping_determine_model(N, dim, dim + 40)
    F_N_net = F_N.net
    data_iteration = data.DataLoader(data.TensorDataset(S), 10, shuffle=True)
    loss = nn.L1Loss()
    optimizer_N = torch.optim.SGD(F_N_net.parameters(), lr=0.005)
    for t in range(2):
        for index, S_k in enumerate(data_iteration):
            S_k_inlist = S_k[0]
            for k in range(10):
                l = loss(F_N.forward(S_k_inlist[k, :, N]), torch.tensor(1))
                optimizer_N.zero_grad()
                l.backward()
                optimizer_N.step()

    all_determine_func[0]=F_N

    for n in range(N+1):
        model=stopping_determine_model(N-n,dim,q_1)
        all_determine_func.append(model)


S=high_dim_asset_price_path(s_0,delta,sigma,corr,K,dim,N)
A=data.TensorDataset(S)
#print(A[0])
#print(S[0,:,:])
#data_iter=data.DataLoader(A,2,shuffle=True)
#for batch_i,S_k in enumerate(data_iter):
#    print(batch_i)
#    print(S_k)
#    print("\n")

F_N=stopping_determine_model(N,dim,dim+40)
F_N_net=F_N.net
data_iteration=data.DataLoader(data.TensorDataset(S), 10, shuffle=True)
#print(data_iteration)
loss=nn.L1Loss()
optimizer_N=torch.optim.SGD(F_N_net.parameters(),lr=0.005)
l=0
#print(S)
for t in range(2):
#    start=time.time()
#    print('epoch is %d'%(t))
#    print('\n')
    for index,S_k in enumerate(data_iteration):
#        print('index is')
#        print(index)
#        print('\n')
        print('S_k is')
        print(S_k)
#        print('\n')
        S_k_inlist= S_k[0]
#        print('S_k_inlist is')
#        print(type(S_k_inlist))
#        print(S_k_inlist)
#        print('\n')
        for k in range(10):
            l = loss(F_N.forward(S_k_inlist[k, :, N]), torch.tensor(1))
            optimizer_N.zero_grad()
            l.backward()
            optimizer_N.step()
#            l = loss(F_N.forward(S_k_inlist[k,:, N]), torch.tensor(1))
#        print('reward(loss): %f, %f sec per epoch \n' % (l, time.time() - start))

all_determine_func[0]=F_N
#print(all_determine_func[0])
target_train_obj=stopping_determine_model(N-1,dim,dim+40)
target_train_net=target_train_obj.net
data_iter = data.DataLoader(data.TensorDataset(S), 5, shuffle=True)
loss=nn.L1Loss()
optimizer=torch.optim.SGD(target_train_net.parameters(),lr=0.001)
all_determine_func.append(target_train_obj)

for p in range(10):
    start = time.time()
    print('epoch is %d'%(p))
    for index, S_k in enumerate(data_iter):
        print('index is')
        print(index)
#        print('\n')
        S_k_inlist=S_k[0]
        reward = n_average_approxi_expect_reward(S_k_inlist, 5, N-1, N, all_determine_func)
        l = loss((1 / reward), torch.tensor(0))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        rd=n_average_approxi_expect_reward(S_k_inlist, 5, N-1, N, all_determine_func)
        print('reward(loss): %f, %f sec per epoch \n' % (rd, time.time() - start))

#print(all_determine_func)
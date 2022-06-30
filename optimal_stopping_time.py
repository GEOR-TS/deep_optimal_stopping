import torch
import numpy as np
from torch import nn
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

def time(n,N):
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

#B_l=a_multi_sample_Brown_Motion(5,10,0)
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
                S[k, d,i+1] = S[k,d,0] * torch.exp((r - delta[d] - ((sigma[d] ** 2) / 2)) * (time(i+1,N)) + sigma[d] * B[k, i + 1])
#    print(S)
    return S

#S=high_dim_asset_price_path(s_0,delta,sigma,corr,K,dim,N)
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
    g=torch.exp(torch.tensor(-r*time(n,N)))*torch.relu(torch.max(s[:,n]-execute_price))
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

class n_stopping_determine_model(nn.Module):     #构建两层神经网络类
    def __init__(self,n,dim,q_1):
        super(n_stopping_determine_model, self).__init__()
        self.n=N
        self.dim=dim
        self.q_1=q_1
        self.F=nn.Sequential(nn.Linear(self.dim,self.q_1),nn.ReLU(),nn.Linear(self.q_1,1),nn.LogSigmoid())

    def forward(self,x):
        x=self.F(x)
        return x

all_determine_func=[1]

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

def n_k_approxi_expect_reward(S,k,K,n,N,all_determine_func):
    if n == N:
        reward=payoff()
        return reward
    else:
        reward=payoff(r,executive_price,n,N,S[k-1,:,:])*all_determine_func[N-n].forward(S[k-1,:,n])+payoff(r,executive_price,calcu_optimal_time(S[k-1,:,:],all_determine_func,n+1,N),N,S[k-1,:,:])*(1-all_determine_func[N-n].forward(S[k-1,:,n]))
        return reward

def n_average_approxi_expect_reward(S,K,n,N,all_determine_func):
    sum_reward=0
    for k in range(1,K+1):
        sum_reward+=n_k_approxi_expect_reward(S,k,K,n,N,all_determine_func)
    ave_reward=(1.0/K)*sum_reward
    return ave_reward


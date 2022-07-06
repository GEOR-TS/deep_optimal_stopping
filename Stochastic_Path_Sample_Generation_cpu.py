#import torch                             ####本程序使用pypy3.8解释器运行会比使用python解释器运行要快20倍，500*5的总样本数生成只需要9秒就能运行完成
import time
import numpy as np
#from matplotlib import pyplot as plt
import pandas as pd

time_0=time.time()

r=0.15
T=3
K=10      #样本路径数
dim=5
N=9
J=10
K_L=10
K_U=10

executive_price=2
#print(tau)
I=2*np.ones(N)
#print(I)

s_0=2*np.ones(dim)
delta=0.1*np.ones(dim)
sigma=0.2*np.ones(dim)
corr=0*np.ones((dim,dim))


def Time(n,N):   #将时间长度T给N等分
    t=n*T / N
    return t

#print(time(1,10))

def std_Brown_Motion_path(expiry_time):     #生成标准布朗运动样本路径
    Z_walk = np.random.randn(expiry_time)
    B_1=np.zeros(expiry_time+1)
    for i in range(expiry_time):
        B_1[i+1]=B_1[i]+Z_walk[i]
    return B_1


def multi_sample_Brown_Motion(num_sample,expiry_time,initial_value):   #生成多个布朗运动路径
    B = initial_value*np.ones((num_sample, expiry_time + 1))
    for k in range(num_sample):
        B[k,:] = B[k,:]+std_Brown_Motion_path(expiry_time)
#        plt.plot(torch.arange(0, N + 1).detach(), B[k, :].detach()) # 把各个布朗运动样本路径画出来
    return B

#B_l=multi_sample_Brown_Motion(5,10,0)
#print(B_l)
#plt.plot(torch.arange(0, 10 + 1).detach(),B_l[0, :].detach())
#plt.show()
#print(B)
#plt.show()

def high_dim_asset_price_path(s_0,delta,sigma,corr,K,dim,N):  #生成多个高维资产价格样本路径
    S=np.ones((K,dim,N+1))
    for k in range(K):
        for d in range(dim):
            B = multi_sample_Brown_Motion(K,N,0)
            S[k,d,:]=s_0[d]*S[k,d,:]
            for i in range(N):
                S[k, d,i+1] = S[k,d,0] * np.exp((r - delta[d] - ((sigma[d] ** 2) / 2)) * (Time(i+1,N)) + sigma[d] * B[k, i + 1])
#    print(S)
    return S

def generate_high_dim_asset_path_file_seperately(s_0,delta,sigma,corr,K,dim,N):    #产生路径存入pandas进而存入csv，然后产生dim这么多的资产样本
    for d in range(dim):
        #产生一个资产的K个样本路径
        B_d=multi_sample_Brown_Motion(K,N,0)
        S_d=np.ones((K,N+1))
        for i in range(N+1):
            S_d[:,i]=s_0[d]*np.exp(B_d[:,i])
        #存入pandas.Series
        data_d=pd.DataFrame(S_d,columns=np.arange(0,N+1),index=np.arange(1,K+1))
        file_path=r'Asset_'+str(d+1)+'.csv'
        data_d.to_csv(file_path)
#        print(data_d)
#        print(S_d)
    return 1

def load_asset_path_file(dim,K,N):    #加载csv文件，返回完整的所有资产的样本路径
    S=np.ones((K,dim,N+1))
    for d in range(dim):
        file_path=r'Asset_'+str(d+1)+'.csv'
        data_d=pd.DataFrame(pd.read_csv(file_path,index_col=0))
        print('index: %d'%(d))
#        print(data_d)
        S_d=data_d.to_numpy()
        S[:,d,:]=S_d
#        print(S_d)
#        print('\n')
    return S
#S_0=2
#S=S_0*torch.ones(K,N+1)

#def plot_asset_price_path(S_d,K,dim,initial_point,N):  #对多个高维资产价格样本路径进行作图，不同的分量对应不同的图
#    for k in range(K):
#        plt.plot(np.arange(initial_point, N + 1+initial_point), S_d[k,:],color='gray')
#    plt.show()
#    return 1

#print(torch.cuda.is_available())
#S=high_dim_asset_price_path(s_0,delta,sigma,corr,K,dim,N)



generate_high_dim_asset_path_file_seperately(s_0,delta,sigma,corr,K,dim,N)
print('use of time: %f'%(time.time()-time_0))
#plot_asset_price_path(S_d,K,dim,0,N)

#load_asset_path_file(dim,K,N)

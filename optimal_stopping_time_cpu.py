import torch
import time
import numpy as np
from torch import nn
from torch.utils import data
from matplotlib import pyplot as plt
from scipy.stats import norm
import pandas as pd
import visdom
#from d2l import torch as d2l

#'--fp16'

time_0=time.time()                             #记录程序开始的时间，为后面测算运行时间做准备
viz=visdom.Visdom(env="Optimal_Stopping_cpu")  #建立visdom环境，初始化，然后可以通过终端打开网页进行实时训练监控

#x=torch.arange(-8.0,8.0,0.1,requires_grad=True)
#y=torch.relu(x)

K=1000000                          #训练样本路径总数K
r=0.15                             #内在资产回报率
dim=5                              #资产组合维数
N=9                                #期权行权时长的等分数，表示有N个可能的行权时间
T=15                               #行权最大时长，即过期时间
J=10                               #延续路径样本数
K_L=10                             #用于估计期望回报下界的样本路径数
K_U=10                             #用于估计期望回报上界的样本路径数
executive_price=2                  #行权价格
#print(tau)
I=2*torch.ones(N)
#print(I)

s_0=2*torch.ones(dim)              #0时刻的资产组合的初始价格向量
delta=0.1*torch.ones(dim)          #资产组合的各资产分红
sigma=0.2*torch.ones(dim)          #资产组合的各资产波动率
corr=0*torch.ones(dim,dim)         #资产组合之间的相关系数矩阵

all_determine_func=[1]             #用于存放所有时刻决策函数神经网络模型
tau=[]                             #用于存放所有时刻的最优停时

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
        plt.plot(torch.arange(0, N + 1).detach(), B[k, :].detach()) # 把各个布朗运动样本路径画出来
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
    print(S)
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

def payoff(r,execute_price,n,N,s):  #n时刻的回报函数(注意s是某一特定样本)
    g=torch.exp(torch.tensor(-r*Time(n,N)))*torch.relu(torch.max(s[:,n]-execute_price))
    return g

#print(S[0,:,5]-0.1)
#print(torch.exp(torch.tensor(-r*time(5,N)))*torch.max(S[0,:,5]-0.1))
#print(payoff(r,0.1,5,N,S[0,:,:]))

def continuation_brown_motion_path(B,k,K,n,N,J):   #某一样本点的多个布朗运动延续样本路径
    B_n =  multi_sample_Brown_Motion(J,N-n,B[k-1,n])
    plt.plot(torch.arange(0, N + 1-n).detach(), B_n[0, :].detach())
    plt.plot(torch.arange(0, N + 1-n).detach(), B_n[1, :].detach())
    plt.plot(torch.arange(0, N + 1 - n).detach(), B_n[2, :].detach())
    plt.show()
    return B_n

def contiuation_high_dim_asset_path(S,delta,sigma,corr,dim,k,K,n,N,J):  #某一样本点的价格过程的延续样本路径
    S_n= high_dim_asset_price_path(S[k-1,:,n],delta,sigma,corr,J,dim,N-n)
    plot_asset_price_path(S_n,J,dim,n,N-n)
    return S_n

#B=multi_sample_Brown_Motion(K,N,0)
#B_n=continuation_brown_motion_path(B,3,K,4,N,5)
#S=high_dim_asset_price_path(s_0,delta,sigma,corr,K,dim,N)
#S_n=contiuation_high_dim_asset_path(S,delta,sigma,corr,dim,5,K,4,N,5)

class stopping_determine_model(nn.Module):     #构建两层神经网络类
    def __init__(self,n,dim,q_1,N=N):           #初始化模型的各重要参数和多层感知网络
        super(stopping_determine_model, self).__init__()
        self.N=N
        self.n=n
        self.dim=dim
        self.q_1=q_1
        self.net=nn.Sequential(nn.Linear(self.dim,self.q_1),nn.ReLU(),nn.Linear(self.q_1,1),nn.Sigmoid())

    def forward(self,x):     #进行前馈计算/前向传播
        if n==self.N:       #N时刻默认输出恒等于1
            return 1
        else:
            x=self.net(x)
#            x.squeeze(-1)
            return x

    def decision(self,x):    #根据sigmoid的决策概率值以0.5为分界得到决策
        if self.net(x)<0.5:
            return 0
        else:
            return 1

    def rate_of_success(self):
        pass

def n_k_approxi_expect_reward(S,k,K,n,N,all_determine_func):   #计算第n时刻k样本的回报估计
    if n == N:
        reward=payoff(r,executive_price,n,N,S[k-1,:,:])
        return reward
    else:
        reward=payoff(r,executive_price,n,N,S[k-1,:,:])*all_determine_func[N-n].forward(S[k-1,:,n])+payoff(r,executive_price,n_k_calcu_optimal_time(S[k-1,:,:],all_determine_func,n+1,N),N,S[k-1,:,:])*(1-all_determine_func[N-n].forward(S[k-1,:,n]))
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
    optimizer=torch.optim.Adam(target_train_net.parameters(),lr=lr)
#    loss=nn.L1Loss()
    all_determine_func.append(target_train_obj)

    rwd=[n_average_approxi_expect_reward(S[0,:,:], batch_size, n, N, all_determine_func)]        #用于记录每次分好的batch的最后一个小批量样本的回报函数

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
            l = 0-reward
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            rd = n_average_approxi_expect_reward(S_k_inlist, batch_size, n, N, all_determine_func)
            print('reward(loss): %f, %f sec per epoch \n' % (rd, time.time() - start))
            if (index + 1) * batch_size % 100 == 0:
                rwd.append(n_average_approxi_expect_reward(S_k_inlist, batch_size, n, N, all_determine_func))

    plt.plot(np.linspace(0, num_epochs, len(rwd)),rwd)
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.show()
    return 1

def optimize_all_determine_func(S,K,dim,N,q_1,all_determine_func):   #倒向迭代（循环）训练所有时刻的决策函数
    F_N = stopping_determine_model(N, dim, dim + 40)
    F_N_net = F_N.net
    data_iteration = data.DataLoader(data.TensorDataset(S), 5, shuffle=True)    #batch_size可修改
    loss = nn.L1Loss()
    optimizer_N = torch.optim.SGD(F_N_net.parameters(), lr=0.005)   #学习率可修改
    for t in range(4):                                              #epoch值可修改
        for index, S_k in enumerate(data_iteration):                #S_k是列表里装着一个batch样本组成的矩阵
            S_k_inlist = S_k[0]
            for k in range(5):
                l = loss(F_N.forward(S_k_inlist[k, :, N]), torch.tensor(1))
                optimizer_N.zero_grad()
                l.backward()
                optimizer_N.step()

    all_determine_func[0]=F_N

    for n in range(N+1):
        model=stopping_determine_model(N-n,dim,q_1)
        all_determine_func.append(model)

def n_k_calcu_optimal_time(S_k, all_determine_func, n, N):  # 计算n-th的最优停时(这里是得到最优决策函数后得到，为后面生成新的样本集计算上下界和置信区间服务)
    if n == N:                                          #注意S_k代表意思是将某一样本输入得到对应的所有时刻对应的最优停时tau^k_n，来计算相应回报函数值和延续价值
        t = N
        return t
    else:
        t = 0
        for m in range(N - n + 1):
            if m == 0:
                t += n * all_determine_func[N - n].forward(S_k[:,n+m])
            else:
                multiple = 1
                for j in range(m):
                    multiple *= (1 - all_determine_func[N - n - j].forward(S_k[:, n + j]))
                t += ((n + m) * all_determine_func[N-n-m].forward(S_k[:, n + m]) * multiple)
        return t

def calcu_all_optimal_time(S_k,all_determine_func,N):    #计算所有时刻对应的最优停时并放到一个列表中储存,返回类型为list
    tau=[]
    for n in range(N+1):
        t_n=n_k_calcu_optimal_time(S_k,all_determine_func,n,N)
        tau.append(t_n)
    return tau

#J_n_k=high_dim_asset_price_path(S[k-1,:,n],delta,sigma,corr,J,dim,N-n) 生成延续路径

def n_k_conti_path_sample_set_dim_expansion(J_n_k,J,dim,n,N):    #向前增加0使得新矩阵跟我们N时刻的矩阵在n方向分量维数相同，使得格式统一便于使用前面已设函数
    New_matrix=torch.zeros(J,dim,N+1)
    for m in range(N-n+1):
        New_matrix[:,:,n+m]=J_n_k[:,:,m]
    return New_matrix

def n_k_continuation_value(J_n_k,all_determine_func, n, N, J):  # 计算延续路径价值(n小于N)，J_n_k代表新生成第k个样本n时刻的J个延续样本路径的样本集(注意这里的J_n_k是维数扩张后的样本集)
    Val = 0
    New_matrix=n_k_conti_path_sample_set_dim_expansion(J_n_k,J,dim,n,N)
    for j in range(J):
        New_n_k_j=New_matrix[j,:,:]
        tau_next_j = n_k_calcu_optimal_time(New_n_k_j, all_determine_func, n+1, N)
        Val+=tau_next_j
    ave_Val=Val/J
    return ave_Val

def extra_modify_global_optimal_time(S_k,tau,all_determine_func,N):       #通过大小判断（根据论文最后计算全局最优停时的部分）修改全局最优停时tau_0,注意必须在全部tau算出来后进行
    payoff_0=payoff(r,executive_price,0,N,S_k)    #第k个样本
    optimal_time=0
    if payoff_0<n_k_continuation_value(S,all_determine_func,1,N,K):   #注意这里我们借用延续价值函数来计算n=1时刻的回报函数的蒙特卡洛估计，而S的样本量够大因此可以认为是确定的近似估计
        f_0=0                                                         #我们不以tau列表作为参数，我们直接使用全局设定的tau列表进行修改，因为tau列表是直接前面生成好，不需要像all_determine_func一样迭代计算
        all_determine_func[N]=f_0
        optimal_time=tau[1]
    else:
        f_0=1
        all_determine_func[N]=f_0
        optimal_time=tau[0]                     #刚开始就立刻停止
    tau[0]=optimal_time
    return tau

def LowerBound_and_variance(New_Sample,all_determine_func,K_L,N):     #用新样本集计算下界和渐近方差(以list返回)
    sum=0
    payoff_Assemble=[]                   #将payoff用list全部记录下来方便后面求方差时调用
    square_sum=0
    #计算下界
    for k in range(K_L):
        k_tau=calcu_all_optimal_time(New_Sample[k,:,:],all_determine_func,N)
        k_tau=extra_modify_global_optimal_time(New_Sample[k:,:],k_tau,all_determine_func,N)
        k_payoff=payoff(r,executive_price,k_tau[0],N,New_Sample[k,:,:])
        payoff_Assemble.append(k_payoff)
        sum+=k_payoff
    ave_sum=sum/K_L
    #计算无偏方差
    for k in range(K_L):
        square_sum+=(payoff_Assemble[k]-ave_sum)**2
    var=square_sum/(K_L-1)
    return [ave_sum,var]

def UpperBound_and_variance(New_Sample,all_determine_func,delta,sigma,corr,dim,K_U,J,N):    #用新样本集计算上界和渐近方差，为统一记号方便计算和理解，我们令n=0的delta_M=0，我们先用集合将所有delta_M算出来后再进行求和（以list返回）
    sum=0
    max_diff_Assemble=[]               #将最大差用list记录下来方便后面计算方差时调用
    square_sum=0
    #计算上界
    for k in range(K_U):
        delta_M_k = torch.zeros(N + 1)
        M_k=torch.zeros(N+1)
        payoff_k=torch.zeros(N+1)
        for n in range(N+1):
            # 计算n_k处payoff
            payoff_n_k = payoff(r, executive_price, n, N, New_Sample[k, :, n])
            payoff_k[n] = payoff_n_k
            if n ==0:
                delta_M_k[n]=0
            else:
                # 计算C_k_n
                conti_path_sample_k_n=contiuation_high_dim_asset_path(New_Sample,delta,sigma,corr,dim,k,K_U,n,N,J)
                Expand_path=n_k_conti_path_sample_set_dim_expansion(conti_path_sample_k_n,J,dim,n,N)
                C_k_n=n_k_continuation_value(Expand_path,all_determine_func,n,N,J)
                #计算C_k_(n-1)
                conti_path_sample_k_n_former = contiuation_high_dim_asset_path(New_Sample, delta, sigma, corr, dim, k, K_U, n-1, N, J)
                Expand_path_former = n_k_conti_path_sample_set_dim_expansion(conti_path_sample_k_n_former, J, dim, n-1, N)
                C_k_n_former = n_k_continuation_value(Expand_path_former, all_determine_func, n, N, J)
                #根据New_sample在n_k位置的值进行计算
                delta_M_k_n=all_determine_func[N-n].forward(New_Sample[k,:,n])*payoff_n_k+(1-all_determine_func[N-n].forward(New_Sample[k,:,n]))*C_k_n - C_k_n_former
                delta_M_k[n]=delta_M_k_n
            #计算M_k_n
            M_k_n=0
            for m in range(n):
                M_k_n+=delta_M_k[m]
            M_k[n]=M_k_n
        #计算sum
        max_difference=torch.max(payoff_k-M_k)                  #计算n从0到N中的最大差
        max_diff_Assemble.append(max_difference)
        sum+=max_difference
    ave_sum=sum/K_U
    #计算无偏方差
    for k in range(K_U):
        square_sum+=(max_diff_Assemble[k]-ave_sum)**2
    var=square_sum/(K_U-1)
    return [ave_sum,var]

def Point_est_and_Confidence_interval(alpha,LowBd_with_var,UppBd_with_var,K_L,K_U):        #计算点估计和置信区间(以list返回)
    #计算点估计
    Point_est = (LowBd_with_var[0] + UppBd_with_var[0]) / 2
    #计算正态分布alpha/2分位数(下侧分位数)
    z_alpha_half=norm.isf(q=alpha/2)
    #计算置信区间
    Interval_L=LowBd_with_var[0]-z_alpha_half*(LowBd_with_var[1]/torch.sqrt(torch.tensor(K_L)))
    Interval_U=UppBd_with_var[0]+z_alpha_half*(UppBd_with_var[1]/torch.sqrt(torch.tensor(K_U)))
    return [Point_est,[Interval_L,Interval_U]]
"""
def combination_five_asset_to_one(assert1,assert2,assert3,assert4,assert5,sam_num):           
    #这个函数将五个assert文件汇聚成一个文件，并将其整理为连续的格式
    #首先要使用pd.read_csv将五个文件输出为assert1,assert2,assert3,assert4,assert5
    #函数最终将整理好的文件return，格式为pd.DataFrame
    arr1=np.array(range(sam_num*5),dtype=int,ndmin=2).T
    arr2=np.empty([sam_num*5,10],dtype=float)
    arr_sum=np.concatenate([arr1,arr2],axis=1)
    assert_final=pd.DataFrame(arr_sum,columns=['rank',0,1,2,3,4,5,6,7,8,9])
    assert_final.set_index('rank',inplace=True)
    assert1.set_index('Unnamed: 0',inplace=True)
    assert2.set_index('Unnamed: 0',inplace=True)
    assert3.set_index('Unnamed: 0',inplace=True)
    assert4.set_index('Unnamed: 0',inplace=True)
    assert5.set_index('Unnamed: 0',inplace=True)
    for i in range(sam_num*5):
        temp=i%5
        n=int(i/5)
        if temp==0:
            assert_final.iloc[i] = assert1.iloc[n]
        elif temp==1:
            assert_final.iloc[i] = assert2.iloc[n]
        elif temp==2:
            assert_final.iloc[i] = assert3.iloc[n]
        elif temp==3:
            assert_final.iloc[i] = assert4.iloc[n]
        elif temp==4:
            assert_final.iloc[i] = assert5.iloc[n]
    return assert_final
"""
"""
def load_asset_path_file(dim,K,N,control_string='Asset_'):    #加载csv文件，返回完整的所有资产的样本路径(注意，文件控制字符串在函数中进行更改)
    S=np.ones((K,dim,N+1))                                    #这部分缺陷是加载速度过慢，故不用此方法进行输入
    for d in range(dim):
        file_path=control_string+str(d+1)+'.csv'
        data_d=pd.DataFrame(pd.read_csv(file_path,index_col=0))
        print('index: %d'%(d))
#        print(data_d)
        S_d=data_d.to_numpy()
        S[:,d,:]=S_d
#        print(S_d)
#        print('\n')
    return S
"""
class MyIterableDataset(data.IterableDataset):       #special for example 1 with 5 dimensional assets    构造了迭代器输入数据的迭代类，此部分代码与迭代器机制和python实现有关，可不具体了解，有兴趣可搜索网上资料
    def __init__(self, file_path_1,file_path_2,file_path_3,file_path_4,file_path_5,K,dim,N,chunksize=100000):      #chunksize是每次迭代的样本块所含样本数
        self.chunksize=chunksize
        self.file_iter_1 = pd.read_csv(file_path_1, iterator=True, chunksize=chunksize,index_col=0)        #使用了pandas的csv输入方法，加上iteration后默认返回迭代格式数据
        self.file_iter_2 = pd.read_csv(file_path_2, iterator=True, chunksize=chunksize,index_col=0)
        self.file_iter_3 = pd.read_csv(file_path_3, iterator=True, chunksize=chunksize,index_col=0)
        self.file_iter_4 = pd.read_csv(file_path_4, iterator=True, chunksize=chunksize,index_col=0)
        self.file_iter_5 = pd.read_csv(file_path_5, iterator=True, chunksize=chunksize,index_col=0)
        self.file_assemble_iter = np.ones((chunksize, dim, N + 1))    #将五个不同的资产组合样本块成一个5维资产组合样本块，这里是初始化这样一个矩阵
        self.chunk_num = int(K / chunksize)           #计算块数

    def __iter__(self):                            #编写迭代方式
        for chunk_num in range(self.chunk_num):
            for data_1 in self.file_iter_1:
                self.file_assemble_iter[:,0,:]=data_1.to_numpy()
            for data_2 in self.file_iter_2:
                self.file_assemble_iter[:,1,:]=data_2.to_numpy()
            for data_3 in self.file_iter_3:
                self.file_assemble_iter[:,2,:]=data_3.to_numpy()
            for data_4 in self.file_iter_4:
                self.file_assemble_iter[:,3,:]=data_4.to_numpy()
            for data_5 in self.file_iter_5:
                self.file_assemble_iter[:,4,:]=data_5.to_numpy()
            yield torch.from_numpy(self.file_assemble_iter).float()


#        with open(self.file_path, 'r') as file_obj:
#            for line in file_obj:
#                line_data = line.strip('\n').split(',')
#                yield torch.from_numpy(np.array(line_data, dtype='int')) # 这里按照自己的代码看格式哈


#dataloader = DataLoader(MyDataset(), batch_size=5)

#for i, item in enumerate(dataloader):
#    print(i, item)

#以下为代码测试实例

#S=high_dim_asset_price_path(s_0,delta,sigma,corr,K,dim,N)
#A=data.TensorDataset(S)
#print(A[0])
#print(S[0,:,:])
#data_iter=data.DataLoader(A,2,shuffle=True)
#for batch_i,S_k in enumerate(data_iter):
#    print(batch_i)
#    print(S_k)
#    print("\n")

#S_pd=load_asset_path_file(dim,K,N,control_string='Asset_')

S_iter=MyIterableDataset('E:\Sample\\1million\Asset_1.csv','E:\Sample\\1million\Asset_2.csv','E:\Sample\\1million\Asset_3.csv','E:\Sample\\1million\Asset_4.csv','E:\Sample\\1million\Asset_5.csv',1000000,5,9)          #迭代输入数据

#S_iter=combination_five_asset_to_one(pd.read_csv('E:\Sample\\1million\Asset_1.csv'),pd.read_csv('E:\Sample\\1million\Asset_2.csv'),pd.read_csv('E:\Sample\\1million\Asset_3.csv'),pd.read_csv('E:\Sample\\1million\Asset_4.csv'),pd.read_csv('E:\Sample\\1million\Asset_5.csv'),K)
#S_iter=S_iter.to_numpy()

n=0
S=torch.ones(K,dim,N+1)
for S_t in S_iter:             #将迭代格式的数据转化成一个完整的大矩阵
    S[n:100000+n,:,:]=S_t
    n+=100000
#    print('index : %d'%(n))
#    print(S_t)
print('n: %d'%(n))

all_determine_func[0]=stopping_determine_model(N,dim,dim+40)   #initialize FN  N时刻初始化

#print(all_determine_func[0])
target_train_obj=stopping_determine_model(N-1,dim,dim+40)      #实例化N-1时刻的决策函数神经网络模型
target_train_net=target_train_obj.net                          #标记该模型的网络部分
#data_iteraion_2 = data.DataLoader(S_iter, 10000)
data_iteration_2=data.DataLoader(data.TensorDataset(S), 10000, shuffle=True)            #将数据重新随机小批量分划，输出一个迭代格式的小批量样本数据集，10000表示batch_size
#loss=nn.L1Loss()
#optimizer=torch.optim.SGD(target_train_net.parameters(),lr=0.005)
optimizer=torch.optim.Adam(target_train_net.parameters())     #使用Adam优化方法
all_determine_func.append(target_train_obj)                 #将N-1的模型添加进list中

for epoch in range(10):                 #进行10个周期的重复训练
    start = time.time()
    print('epoch is %d'%(epoch))
    for index, S_k in enumerate(data_iteration_2):     #遍历一个周期中的所有小批量样本块
        print('index is: %d'%(index))
#        print(index)
#        print('\n')
        S_k_inlist=S_k[0]          #由于迭代出来的格式是放在了list中，直接取出来
        reward = 0- n_average_approxi_expect_reward(S_k_inlist, 10000, N-1, N, all_determine_func)          #用回报函数的平均值作为优化目标函数
        optimizer.zero_grad()           #每次训练时都将上次的梯度清零
        reward.backward()           #反向计算
        optimizer.step()            #将网络中所有参数更新
        rd=n_average_approxi_expect_reward(S_k_inlist, 10000, N-1, N, all_determine_func)   #为打印数值而再计算一次

        print('reward: %f, %f sec per epoch \n' % (rd, time.time() - start))
        viz.line(X=np.array([index+100*epoch]), Y=np.array([rd.detach().numpy()]), win='tarin_reward',      #用visdom进行每一时刻的数值追踪作图
                 opts={'title': 'train_reward', 'legend': ['train']}, update='append')

#print(all_determine_func)

# 1 查看网络第一层(即第一个全连接层)的参数
#print(all_determine_func[1].net[0].state_dict())
#print('\n')
# 2 查看网络第三层(即第二个全连接层)偏置参数的类型
#print(type(all_determine_func[1].net[2].bias))
#print('\n')
# 3 查看网络第三层(即第二个全连接层)偏置参数
#print(all_determine_func[1].net[2].bias)
#print('\n')
# 4 查看网络第三层(即第二个全连接层)偏置参数的值
#print(all_determine_func[1].net[2].bias.data)
#print('\n')
# 5 查看网络第一层(即第一个全连接层)权重参数
#print(all_determine_func[1].net[0].weight)
#print('\n')
# 6 查看网络第二层
#print(all_determine_func[1].net[1])

#测试有无gpu
print(torch.cuda.is_available())
print(torch.cuda.device_count())
#查看pytorch版本
print(torch.__version__)
print('use of time: %f'%(time.time()-time_0))        #程序最终用时
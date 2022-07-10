import torch
import time
import numpy as np
from torch import nn
from torch.utils import data
from torch.nn import functional
from matplotlib import pyplot as plt
from scipy.stats import norm
import pandas as pd
#import gym

time_0=time.time()

K=1000000
r=0.15
dim=5
N=9
T=15
J=10
K_L=10
K_U=10
executive_price=2
#print(tau)
I=2*torch.ones(N)
#print(I)

s_0=2*torch.ones(dim)
delta=0.1*torch.ones(dim)
sigma=0.2*torch.ones(dim)
corr=0*torch.ones(dim,dim)


def Time(n,N):   #将时间长度T给N等分
    t=n*T / N
    return t

"""
class RF(object):
    def __init__(self,env):
        # 环境的状态和动作维度
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        # 存放每个episode的s,a,r
        self.ep_obs,self.ep_act,self.ep_r = [],[],[]
        # 初始化神经网络
        self.net = Net(obs_dim = self.obs_dim,act_dim = self.act_dim).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.time_step = 0


    # 选择动作的函数
    def choose_action(self,obs):
        obs = torch.FloatTensor(obs).to(device)   # 转换为torch的格式
        action = self.net.forward(obs)
        with torch.no_grad(): # 不进行参数的更新
            action = F.softmax(action,dim=0).cuda().data.cpu().numpy()
        action = np.random.choice(range(action.shape[0]),p=action)   # 根据softmax输出的概率来选择动作

        return action

    # 存储一个episode的状态、动作和回报的函数
    def store_transition(self,obs,act,r):
        self.ep_obs.append(obs)
        self.ep_act.append(act)
        self.ep_r.append(r)

    # 更新策略网络的函数
    def learn(self):
        self.time_step += 1  # 记录走过的step
        # 记录Gt的值
        discounted_ep_rs = np.zeros_like(self.ep_r)
        running_add = 0
        # 计算未来总收益
        for t in reversed(range(0,len(self.ep_r))):  # 反向计算
            running_add = running_add * GAMMA + self.ep_r[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)  # 减均值
        discounted_ep_rs /= np.std(discounted_ep_rs)  # 除以标准差
        discounted_ep_rs = torch.FloatTensor(discounted_ep_rs).to(device)

        # 输出网络计算出的每个动作的概率值
        act_prob = self.net.forward(torch.FloatTensor(self.ep_obs).to(device))
        # 进行交叉熵的运算
        neg_log_prob = F.cross_entropy(input = act_prob,target=torch.LongTensor(self.ep_act).to(device),reduction = 'none')
        # 计算loss
        loss = torch.mean(neg_log_prob * discounted_ep_rs)

        # 反向传播优化网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每次学习后清空s,r,a的数组
        self.ep_r,self.ep_act,self.ep_obs = [],[],[]   

def execute():
    # 初始化RF类
    agent = RF(env)

    # 进行训练
    for episode in range(EPISODE):
        obs = env.reset()
        for step in range(STEP):
            # 与环境的交互
            action = agent.choose_action(obs)
            next_obs,reward,done,_ = env.step(action)
            # 存储一个episode中每个step的s,a,r
            agent.store_transition(obs,action,reward)
            # 进入下一个状态
            obs = next_obs
            # 每个episode结束再进行训练(MC)
            if done:
                agent.learn()
                break
        # 每100个episode进行测试
        if episode % 100 == 0:
            avg_reward = test_episode(env,agent)
            print('Episode: ',episode,'Test_reward: ',avg_reward) """

class stopping_determine_model(nn.Module):     #构建两层神经网络类
    def __init__(self,n,dim,q_1,N=N):
        super(stopping_determine_model, self).__init__()
        self.N=N
        self.n=n
        self.dim=dim
        self.q_1=q_1
        self.net=nn.Sequential(nn.Linear(self.dim,self.q_1),nn.ReLU(),nn.Linear(self.q_1,self.q_1),nn.ReLU(),nn.Linear(self.q_1,self.q_1),nn.ReLU(),nn.Linear(self.q_1,1),nn.Sigmoid())

    def forward(self,x):
        if n==self.N:
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



class Policy_Reinforce():
    def __init__(self,K,dim,n,N):
        self.dim=dim
        self.n=n
        pass

    def choose_action(self,S_k):

        obs = torch.FloatTensor(obs).to(device)   # 转换为torch的格式

        action = self.net.forward(obs)
        with torch.no_grad(): # 不进行参数的更新
            action = F.softmax(action,dim=0).cuda().data.cpu().numpy()
        action = np.random.choice(range(action.shape[0]),p=action)   # 根据softmax输出的概率来选择动作
        pass
        return action


class MyIterableDataset(data.IterableDataset):       #special for example 1 with 5 dimensional assets
    def __init__(self, file_path_1,file_path_2,file_path_3,file_path_4,file_path_5,K,dim,N,chunksize=100000):
        self.chunksize=chunksize
        self.file_iter_1 = pd.read_csv(file_path_1, iterator=True, chunksize=chunksize,index_col=0)
        self.file_iter_2 = pd.read_csv(file_path_2, iterator=True, chunksize=chunksize,index_col=0)
        self.file_iter_3 = pd.read_csv(file_path_3, iterator=True, chunksize=chunksize,index_col=0)
        self.file_iter_4 = pd.read_csv(file_path_4, iterator=True, chunksize=chunksize,index_col=0)
        self.file_iter_5 = pd.read_csv(file_path_5, iterator=True, chunksize=chunksize,index_col=0)
        self.file_assemble_iter = np.ones((chunksize, dim, N + 1))
        self.chunk_num = int(K / chunksize)

    def __iter__(self):
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

def payoff(r,execute_price,n,N,s):  #n时刻的回报函数(注意s是某一特定样本)
    g=torch.exp(torch.tensor(-r*Time(n,N)))*torch.relu(torch.max(s[:,n]-execute_price))
    return g

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



S_iter=MyIterableDataset('E:\Sample\\1million\Asset_1.csv','E:\Sample\\1million\Asset_2.csv','E:\Sample\\1million\Asset_3.csv','E:\Sample\\1million\Asset_4.csv','E:\Sample\\1million\Asset_5.csv',1000000,5,9)


n=0
S=torch.ones(K,dim,N+1)
for S_t in S_iter:
    S[n:100000+n,:,:]=S_t
    n+=100000
#    print('index : %d'%(n))
#    print(S_t)
print('n: %d'%(n))

#初始化第N时刻的决策函数
all_determine_func=[stopping_determine_model(N,dim,dim+40)]


target_train_obj=stopping_determine_model(N-1,dim,dim+40)
target_train_net=target_train_obj.net
#data_iteraion_2 = data.DataLoader(S_iter, 10000)
data_iteration_2=data.DataLoader(data.TensorDataset(S), 10000, shuffle=True)
loss=nn.L1Loss()
optimizer=torch.optim.SGD(target_train_net.parameters(),lr=0.003)

all_determine_func.append(target_train_obj)

for p in range(2):
    start = time.time()
#    print('epoch is %d'%(p))
    for index, S_k in enumerate(data_iteration_2):
        print('index is')
        print(index)
#        print('\n')
        S_k_inlist=S_k[0]
        #MonteCarlo估计回报函数

#        reward = n_average_approxi_expect_reward(S_k_inlist, 5000, N-1, N, all_determine_func)

        ###注意把reward的循环和entropy的循环（k）个分量用向量操作进行，减少循环使用
        reward_SampleVect=torch.zeros(10000)
        decision_SampleVect=torch.zeros(10000)   #这里的动作（决策）由每个样本输入网络后得到的估计值以0.5作为界来得到的0，1决策
        prob_SampleVect=torch.zeros(10000)

        n=0

        for k in range(10000):  #batch_size=5000
            reward_SampleVect[k] = payoff(r, executive_price, N-1, N, S_k_inlist[k , :, :]) * all_determine_func[N - (N-1)].forward(S_k_inlist[k , :, N-1]) + \
                    payoff(r, executive_price,n_k_calcu_optimal_time(S_k_inlist[k , :, :], all_determine_func, (N-1) + 1, N), N ,S_k_inlist[k , :, :]) * (1 - all_determine_func[N - (N-1)].forward(S_k_inlist[k , :, N-1]))

#            decision_SampleVect[k] = all_determine_func[1].decision(S_k_inlist[k,:,N-1])

            if payoff(r,executive_price,N-1,N,S_k_inlist[k,:,:])>=payoff(r,executive_price,N,N,S_k_inlist[k,:,:]):
                decision_SampleVect[k]=1
            else:
                decision_SampleVect[k]=0

            prob_SampleVect[k]=all_determine_func[1].forward(S_k_inlist[k,:,N-1])

            if all_determine_func[1].decision(S_k_inlist[k,:,N-1])==int(reward_SampleVect[k].item()>=payoff(r,executive_price,N,N,S_k_inlist[k,:,:]).item()):
                n+=1
            else:
                n+=0

#            print('N-1 decision is : %d , reward: %f , N reward %f , use of time: %d \n' % (decision_SampleVect[k].item(),reward_SampleVect[k].item(),payoff(r,executive_price,N,N,S_k_inlist[k,:,:]).item(),time.time()-start))

#        reward_SampleVect-=torch.mean(reward_SampleVect)      #将回报函数减去平均值，获得一个幅度的显正负的值来进行更新
        ave_reward=torch.mean(reward_SampleVect)

        cross_entropy= functional.binary_cross_entropy(prob_SampleVect,decision_SampleVect)
        ave_entropy=torch.mean(cross_entropy)

        print('rate of success: %f, ave reward: %f, ave cross entropy loss: %f' % (n / 10000 , ave_reward.item(), ave_entropy.item()))

        l = torch.mean(reward_SampleVect*cross_entropy)
#        l=ave_reward*ave_entropy

        optimizer.zero_grad()
        l.backward()
        optimizer.step()
#        rd=n_average_approxi_expect_reward(S_k_inlist, 5000, N-1, N, all_determine_func)
#        print('N-1 decision is : %d , reward: %f , N reward %f , use of time: %d'%(all_determine_func[1].decision(S[])))

print('finished, total use of time: %f'%(time.time()-time_0))


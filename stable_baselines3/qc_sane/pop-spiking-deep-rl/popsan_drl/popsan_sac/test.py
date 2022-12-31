import itertools

class B():
    def __init__(self,l,m,n):
        self.ben=l
        self.bdec=m
        self.bsnn=n
        
class A():
    def __init__(self):
        for actor_index in range(5):
            popsan_params_actor_index="""B(actor_index*10,actor_index*100,actor_index*1000)"""
            # print("O",popsan_params_actor_index)
            exec("self.pop%d = %s" % (actor_index + 1, popsan_params_actor_index));
            
def create_other(a):
    for j in range(5):
        temp="""itertools.chain(eval("a.pop%d.ben"%(j+1)),eval("a.pop%d.bsnn"%(j+1)),eval("a.pop%d.bdec"%(j+1)))"""
        exec("pop_params%d = %s" % (j + 1, temp));
        print(eval("pop_params%d"%(j+1)))
        # show(a)

def show(a):
    for j in range(5):
        print(eval("a.pop%d"%(j+1)))

a=A()
print(a)
create_other(a)
quantiles=[0.1,0.5,0.9]
q1=10
q2=20
for critic_idx in range(2):
    exec("qf%d_losses = %s" % (critic_idx + 1, []));            
    for i, quantile in enumerate(quantiles):
        error = 100 - eval("q%d"%(critic_idx+1))
        loss = max(quantile*error, (quantile-1)*error)
        exec("qf%d_losses.append(%s)" % (critic_idx + 1, loss));	
        print(loss)
    print(eval("qf%d_losses" % (critic_idx + 1)))	

import numpy as np
q_info = dict()   
for critic_idx in range(2):

    exec("q_info['Q%dVals'] = %s" % (critic_idx + 1,"qf%d_losses" % (critic_idx + 1)));
    # print(np.array([critic_idx*10,40]))
print("###",q_info)	
from torch.optim import Adam
import torch
for actor_idx in range(2):
    exec("popsan_mean_optimizer%d = %s" % (actor_idx + 1,""" Adam([torch.tensor(eval("qf%d_losses" % (actor_idx + 1)))], lr=0.1)"""));
    Adam([torch.tensor(eval("qf%d_losses"%(actor_idx+1)))], lr=0.1).zero_grad()

    # print(eval("popsan_mean_optimizer%d"%(actor_idx+1)))
    
def tryi():
    return 40,30

for actor_index in range(4):
    exec("loss_pi%d,inf%d = %s"%(actor_index+1,actor_index+1,tryi()))
print(loss_pi1,loss_pi2,inf1,inf2)	

def tryi2():
    l=[40,30,50,60]
    return *l,2000
b=tryi2()
print(b)

class C():
    def __init__(self,l):
        for actr in range(3):
            exec("self.q%d = %s"%(actr+1,l[actr]))

c=C([42,10,20])
print(c.q1,c.q2,c.q3)
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    print(np.shape(x),x[:,1])
    sc = ax.scatter(x[:,0], x[:,1])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

path="result_walk/usable/15_snn_selu_quan_3actor_bst procd/"
fold="params/spike-sac_sac-popsan-Walker2d-v2-encoder-dim-10-decoder-dim-10/"
data=pd.read_csv(path+fold+"data0_traincsv.csv",low_memory=False)

with open(path+"terminal",'r')as f:
    alll=f.read()
t=alll.split("Model")
agent=[]
for tem in t:
    if "Steps" in tem:
        k=tem.split("Selected aCtro")[1].split("\n")[0].strip()
        agent.append(k)


label=[1]  #t=-1
#label.extend([-1]*(len(data)-1))#t=0 & onward
#label.extend(["1"]*10000) #starting 10k actor1 (t+1)
k=0
time=data["t"]
#print(time[:50])
#print(len(time))
for i in time[1:]: #time-1 already covered
    if (i+1)<10000:
        label.extend(["1"])
    else:
        label.extend(agent[((i+1)//10000)-1])


data["agent"]=label

drop_nn_data=data.dropna()

state=[i[1:-1].split() for i in drop_nn_data["o2"]]
#np.shape(state)
row=np.shape(state)[0]
clmn=np.shape(state)[1]
#row,clmn
obs=[[float(state[k][col]) for col in range(clmn)] for k in range(row)]
y=drop_nn_data["agent"][:int(2e5)]
target_ids=[1,2,3]
target_names=y.unique()
for per in [50,75,100,500,1000]:
    print("\nTSNining:::: ofr perplexity",per)
    X_embedded = TSNE(random_state=100,n_components=2,perplexity=per).fit_transform(obs[:int(2e5)])
    #fashion_scatter(X_embedded, y)
    print("\n\nploting ...")
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b'
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(X_embedded[y == str(i), 0], X_embedded[y == str(i), 1], c=c, label=label)
    plt.legend()
    plt.savefig(path+str(per)+"_tsne.png",dpi=400)

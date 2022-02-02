import numpy as np
import pandas as pd
from scipy.stats import norm


f1 = open("Sample input and output for HMM/Input/data.txt",'r')
f2 = open("Sample input and output for HMM/Input/parameters.txt.txt",'r')
classNames = ["El Nino", "La Nina"]
num_array = np.array([[float(num) for num in f1.read().split()]])
state_no = int((f2.readline().split())[0])
trans_mat1 = []
for i in range(state_no):
    tmp = [float(num) for num in f2.readline().split()]
    trans_mat1.append(tmp)
df = pd.DataFrame(num_array)
trans_mat1 = np.array(trans_mat1)
means1 = [float(num) for num in f2.readline().split()]
stds1 = [np.sqrt(float(num)) for num in f2.readline().split()]
#means1,stds1,trans_mat1,num_array


def printarr(arr):
    for val in arr:
        print(val, end = " ")
    print()


def get_stationary_dist(mat):
    arr = mat.T[:-1]
    arr = np.r_[arr,[np.ones(arr.shape[1])]]
    for i in range(arr.shape[0]-1):
        for j in range(arr.shape[1]):
            if i==j:
                arr[i][j] -= 1
    b = np.zeros((arr.shape[0],1))
    b[b.shape[0]-1][0] = 1
    
    ans = np.linalg.solve(arr,b)
    return ans

st_ds = get_stationary_dist(trans_mat1).T
stat_dis = st_ds
#print(st_ds)
st_ds = np.array([np.log2(val) for val in st_ds])

#st_ds
def get_probs(num_array,means1,stds1,vit=1):
    probs = []
    for x in num_array[0]:
        vals = []
        for i in range(len(means1)):
            #val = 1/(stds1[i]*np.sqrt(2*np.pi))*np.exp(-.5*((x-means1[i])**2/stds1[i]))
            val = norm.pdf(x,loc=means1[i],scale=stds1[i])
            if(vit==1): val = np.log2(val)
            vals.append(val)
        
        probs.append(vals)
    probs = np.array(probs)
    return probs
probs = get_probs(num_array,means1,stds1)
#probs.shape
def get_class(st_ds,t_m,probs,classNames):
    classes = []
    t_m = np.array([np.log2(val) for val in t_m])
    for i in range(1,probs.shape[0]+1):
        temp = []
        cls = []
        for j in range(st_ds.shape[1]):
            mx = None
            idx = -1
            for k in range(st_ds.shape[1]):
                tmp = st_ds[0][k] + t_m[k][j] + probs[i-1][j]
                if mx == None:
                    mx = tmp
                    idx = k
                else :
                    if mx<tmp :
                        mx = tmp
                        idx = k
            cls.append(idx)
            temp.append(mx)
        st_ds = np.array([[float(val) for val in temp]])
        classes.append(cls)
    
    idx = None
    mx = None
    for i in range(st_ds.shape[1]):
        x = st_ds[0][i]
        if mx == None:
            mx = x
            idx = i
        else :
            if mx<x :
                mx = x
                idx = i
    ans = []
    ans.append(idx)
    for x in reversed (classes):
        if len(ans) == 0:
            ans.append(np.argmax(x))
        else : 
            ans.append(x[ans[len(ans)-1]])
        
    ans = ans[:-1]
    classes = []
    for x in reversed(ans):
        classes.append(classNames[x])
    return classes
clss = get_class(st_ds,trans_mat1,probs,classNames)
clss
f3 = open("Sample input and output for HMM/Output/states_Viterbi_wo_learning.txt",'r')
lines = [line.replace('"',"") for line in f3.read().splitlines()]
cnt = 0
print("Predicted","\t","Actual")
for i in range(len(clss)):
    print(clss[i],"\t",lines[i])
    if(clss[i]==lines[i]): 
        cnt += 1
        #print(1)
    #else: print(0)
print("accuracy : ",cnt/len(clss)*100,"%\n")

'''np.random.seed(0)
trans_mat2 = np.random.rand(trans_mat1.shape[0],trans_mat1.shape[1])
means2 = np.random.rand(len(means1))
stds2 = np.random.rand(len(means2))
trans_mat2,means2,stds2'''



def set_class(t_m,probs):
    
    
    forward = np.zeros((probs.shape[0], t_m.shape[0]))
    backward = np.zeros((probs.shape[0], t_m.shape[0]))
    pix = np.zeros((probs.shape[0], t_m.shape[0]))
    pixx = np.zeros((probs.shape[0]-1, t_m.shape[0]* t_m.shape[0]))
    st_ds = get_stationary_dist(trans_mat1).T

    for i in range(probs.shape[0]):

        st_ds = (st_ds@t_m)*probs[i]
           
        st_ds = np.array([val/np.sum(st_ds) for val in st_ds])
        forward[i] = st_ds
    st_ds = np.ones((1,t_m.shape[0]))
    backward[probs.shape[0]-1] = st_ds
    for i in range(probs.shape[0]-2,-1,-1):


        st_ds = (t_m@(st_ds*probs[i+1]).T).T
           
        st_ds = np.array([val/np.sum(st_ds) for val in st_ds])
        backward[i] = st_ds
        

    #np.savetxt("forwardMatrix.txt",forward,delimiter=' ')
    #np.savetxt("backwardMatrix.txt",backward,delimiter=' ')

    for i in range(probs.shape[0]):
        pix[i] = forward[i]*backward[i]
        pix[i] = np.array([val/np.sum(pix[i]) for val in pix[i]])
        if i>0:
            fr = np.array([forward[i-1]]).T
            #print("b",fr)
            fr =  (fr * t_m)
            #print("a",fr)
            pixx[i-1] = (fr*(backward[i]*probs[i])).flatten()
            pixx[i-1] = np.array([val/np.sum(pixx[i-1]) for val in pixx[i-1]])
    
    bb = np.sum(pixx,axis=0)
    #print(bb.shape)
    avg_probs = np.sum(pix,axis = 0)
    #print(avg_probs)
    prob_rains = np.sum(pix*(num_array.T),axis = 0)
    #print(prob_rains)
    avg_rains = prob_rains/avg_probs
    #print(avg_rains)
    std_probs = np.sum(pix*(num_array.T-avg_rains)**2,axis = 0)
    #print(std_probs)
    std_rains = np.sqrt(std_probs/avg_probs)
    #print(std_rains)
    n = (int)(np.sqrt(bb.shape[0]))
    ret = np.reshape(bb,(n,n))
    tr_mat = []
    for i in range(ret.shape[0]) :
        tr_mat.append([val/np.sum(ret[i]) for val in ret[i]])
    ret = np.array(tr_mat)
    #print(ret.shape)
    return ret, avg_rains, std_rains
i = 0
while i<10:
    print("Iteration :" , i+1)
    probs = get_probs(num_array,means1,stds1,vit=0)
    ret, avg_rains, std_rains = set_class(trans_mat1,probs)
    trans_mat1 = ret
    means1 = avg_rains
    stds1 = std_rains
    #print(ret, avg_rains, std_rains)
    for vals in trans_mat1:
        printarr(vals)
    printarr(means1)
    printarr(stds1)
    i += 1


probs = get_probs(num_array,means1,stds1)
st_ds = get_stationary_dist(trans_mat1).T
#print(st_ds)
clss = get_class(st_ds,trans_mat1,probs,classNames)
clss
f4 = open("Sample input and output for HMM/Output/states_Viterbi_after_learning.txt",'r')
lines = [line.replace('"',"") for line in f4.read().splitlines()]
cnt = 0
print("\nPredicted","\t","Actual")
for i in range(len(clss)):
    print(clss[i],"\t",lines[i])
    if(clss[i]==lines[i]): 
        cnt += 1
        #print(1)
    else: print(clss[i]," ",lines[i],0)
print("accuracy : ",cnt/len(clss)*100,"%")
print()
stds1 = [val*val for val in stds1]
for vals in trans_mat1:
    printarr(vals)
printarr(means1)
printarr(stds1)
for val in stat_dis:
    printarr(val)
'''f3 = open("Sample input and output for HMM/Output/states_Viterbi_after_learning.txt",'r')
lines = [line.replace('"',"") for line in f3.read().splitlines()]
cnt = 0
for i in range(len(clss)):
    print(clss[i]," ",lines[i],end=" ")
    if(clss[i]==lines[i]): 
        cnt += 1
        print(1)
    else: print(0)
#cnt = np.sum(clss==lines)
print("accuracy : ",cnt/len(clss)*100,"%")'''
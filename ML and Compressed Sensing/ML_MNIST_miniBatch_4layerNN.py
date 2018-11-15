# -*- coding: utf-8 -*-
##
### Making two hidden layers rather one hidden layer. 
##  Trying out mini-batch gradient descent


import pandas as pd
import numpy as np
import time as time
import sklearn as lrn
import math
import random as rand
import sklearn.preprocessing as lrnpr
import scipy.io
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.utils import shuffle
t0 = time.time()
filename = "C:\\Users\\Jyler\\Documents\\DataSci_ML\\trainset5000.csv"
testfile = "C:\\Users\\Jyler\\Documents\\DataSci_ML\\testset100.csv"
crsVal = "C:\\Users\\Jyler\\Documents\\DataSci_ML\\CrossValset100.csv"

df_init = pd.read_csv(filename)
#print(np.shape(df_init),type(df_init), df_init[0:5,402:407])

############ Testing dataset below
#dftst_init = pd.read_csv(testfile)

dftestX_ = dftst_init.drop('label',axis=1).values
dftesty = dftst_init['label'].values[:,np.newaxis]
dftestX = dftestX_.transpose()
########## 

dfcrs_init = pd.read_csv(crsVal)
dfcrsX_ = dfcrs_init.drop('label',axis=1).values
dfcrsy = dfcrs_init['label'].values[:,np.newaxis]
dfcrsX = dfcrsX_.transpose()

dfy = df_init['label'].values[:,np.newaxis] ## 42000x1
dfX_ = df_init.drop('label',axis=1).values ## 42000 x 784 when using full MNIST dataset.
dfX = dfX_.transpose() ## 784 x ...

m = np.shape(dfy)[0]

num_labels = 10
nhid = 700 # should be roughly 550s +250s+250s = 1050s ~~ 18mins
nhid2 = 25 
nout = 10
nvis = 784 #?


### The vector below is missing the bias term for the input layer. 
### Unrolls every theta matrix into a vector of paramters, with views on where ...
### each matrix and its respective bias vector is. 
# nvis*nhid is shape of vis_to_hid, nhid is shape of hid_bias
# nhid*nhid2 is shape of hid_to_hid2, nhid2 is shape of hid2_bias
# nhid2*nout is shape of hid2_to_out, nout is shape of out_bias
######### nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout + nout
params = np.random.normal(0.0,0.1,(nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout + nout,1));
print('Parameter vector shape =',np.shape(params))
#######
#print(np.shape(dfy))

def makepic(X,y,im_num):
#    print(np.shape(X),'<<<')
    pic1D = X[im_num,:]
    pic2D = pic1D.reshape(28,28)
    pixmax = np.max(pic1D)
    pixmin = np.min(pic1D)
    pic2D_normed = (pic2D - pixmin)/(pixmax - pixmin)
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.imshow(pic2D_normed, cmap='gray')
    print("number shown =",y[im_num][0])
    
#makepic(dfX_,dfy,100)
############3
def sigmoid(z):
    #r = 1.0/(1+np.exp(-z))
    r = expit(z)
    return r

def sigmoidGradient(z):
    r = (sigmoid(z))*(1-sigmoid(z))
    return r

############ Gradient descent with momentum.
def DescentMomentum(params,gradparams,v_t1,p_mtum,alpha):
    
    v_t2 = p_mtum*v_t1 + alpha*gradparams
    updated_params = params - v_t2
    return (updated_params,v_t2)
###########################

### Samples main dataset to produce mini-batches. i.e. 
### input == dataset and labels, mini-batch size. 
### Output = minibatch dataframe and corresponding labels.
def sample(Xshuff,y):
    # X is array 784x m m = num of training pictures
    # y is array m x 1
    n = np.shape(X)[1] - 1
    s = 100
    skip = sorted(rand.sample(range(1,n+1),n-s))
    #dfbatchX = 
########################
#def minibatch(df,)
####
#########################
def numerGrad(params,X,y,lda,num_labels):
    numGrad = np.zeros([np.size(params),1])
    number_vect = np.arange(0,np.size(params),1,dtype=int)
    for i in range(20):
        basis_vect = (number_vect == i)[:,np.newaxis]
        #print('bas',np.shape(basis_vect),'number_vect',np.shape(number_vect))
        eps = (1e-4)*basis_vect
        #print(eps[0:3],'||',(params-eps)[0:3])
        #break
        loss1 = CostFunction(params-eps,X,y,lda,num_labels)
        loss2 = CostFunction(params+eps,X,y,lda,num_labels)
        numGrad[i] = (loss2 - loss1)/(2*(1e-4))
    return numGrad
#########################
## Forward Propagation, Costfunction and regularization
## lda = lambda for regularization, num_labels = number of labels.
def CostFunction(params,X,y,lda,num_labels):
    mbatch = np.size(y)
    ### Reconstruct theta matrices from the parameters vector.
    vis_to_hid = params[:nvis*nhid].reshape((nhid,nvis)) ## theta1 excluding bias term
    hid_bias = params[nvis*nhid:nvis*nhid+nhid] ##  bias term for vis_to_hid
    hid_to_hid2 = params[:nhid*nhid2].reshape((nhid2,nhid))
    hid2_bias = params[nvis*nhid + nhid + nhid*nhid2:nvis*nhid + nhid + nhid*nhid2 + nhid2]
    hid2_to_out = params[nvis*nhid + nhid + nhid*nhid2 + nhid2:nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout].reshape((nout,nhid2)) ## 
    out_bias = params[nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout:nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout + nout] ## bias for ^
    #print('vistohid = ',np.shape(vis_to_hid),'hid_bias=',np.shape(hid_bias),'\nhid_to_hid2=',np.shape(hid_to_hid2)
    #    ,'hid2_bias=',np.shape(hid2_bias),'hid2_to_out=',np.shape(hid2_to_out))
    #return None
    
    J = 0;
    Z_2 = np.dot(vis_to_hid,X) + hid_bias ##
    A_2 = sigmoid(Z_2);
    Z_3 = np.dot(hid_to_hid2,A_2) + hid2_bias
    A_3 = sigmoid(Z_3)
    Z_4 = np.dot(hid2_to_out,A_3) + out_bias; ## 
    A_4 = sigmoid(Z_4); ## 
    
    
    tempJ = 0;
    #tempReg = 0;
    for i in range(num_labels):
        tempy = (y==i)

        tempReg = ((lda)/(2*mbatch))*((np.sum(np.multiply(vis_to_hid,vis_to_hid))+np.sum(np.multiply(hid_to_hid2,hid_to_hid2)))
                    + np.sum(np.multiply(hid2_to_out,hid2_to_out)));
#        tempJ = tempJ + (-1/m)*(np.dot((tempy.transpose()),(np.log(((h_out[i]).tranpose())))) + (np.dot(((1-tempy).transpose()),(np.log((1-h_out[i]).transpose())))))
        tempJ = tempJ + (-1/mbatch)*(np.dot(tempy.transpose(),(np.log(((A_4[i]).transpose())))) + (np.dot((1-tempy).transpose(),(np.log(((1-A_4[i]).transpose()))))))

    J = tempJ + tempReg
    

    return J[0]  
#print('# of pics =',m)\
#print(CostFunction(params,dfX,dfy,0,10))
## lda = lambda for regularization, num_labels = number of labels.
## Back prop for gradient. I should probably verify this with a derivative function.
def gradCostFunction(params,X,y,lda,num_labels):
    mbatch = np.size(y)
    ### Reconstruct theta matrices from the parameters vector.
    vis_to_hid = params[:nvis*nhid].reshape((nhid,nvis)) ## theta1 excluding bias term
    hid_bias = params[nvis*nhid:nvis*nhid+nhid] ##  bias term for vis_to_hid
    hid_to_hid2 = params[:nhid*nhid2].reshape((nhid2,nhid))
    hid2_bias = params[nvis*nhid + nhid + nhid*nhid2:nvis*nhid + nhid + nhid*nhid2 + nhid2]
    hid2_to_out = params[nvis*nhid + nhid + nhid*nhid2 + nhid2:nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout].reshape((nout,nhid2)) ## 
    out_bias = params[nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout:nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout + nout] ## bias for ^

    ### Construct thetaGradient matrices from a gradient parameters vector.
    paramsGrad = np.zeros([nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout + nout,1])
    tempGrad_1 = paramsGrad[:nvis*nhid].reshape((nhid,nvis)) ## theta1 excluding bias term
    tempGrad_1bias = paramsGrad[nvis*nhid:nvis*nhid+nhid] ##  bias term for vis_to_hid
    tempGrad_2 = paramsGrad[:nhid*nhid2].reshape((nhid2,nhid))
    tempGrad_2bias = paramsGrad[nvis*nhid + nhid + nhid*nhid2:nvis*nhid + nhid + nhid*nhid2 + nhid2]
    tempGrad_3 = paramsGrad[nvis*nhid + nhid + nhid*nhid2 + nhid2:nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout].reshape((nout,nhid2)) ## 
    tempGrad_3bias = paramsGrad[nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout:nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout + nout] ## bias for ^

    Z_2 = np.dot(vis_to_hid,X) + hid_bias ##
    A_2 = sigmoid(Z_2);
    Z_3 = np.dot(hid_to_hid2,A_2) + hid2_bias
    A_3 = sigmoid(Z_3)
    Z_4 = np.dot(hid2_to_out,A_3) + out_bias; ## 
    A_4 = sigmoid(Z_4); ## 
    
    for t in range(mbatch):
        tempy_ = np.arange(0,num_labels,1).transpose()
        tempy = (tempy_ == y[t])
        
        delta4 = A_4[:,t][:,np.newaxis] - tempy[:,np.newaxis] #
        
        delta3 = np.dot(hid2_to_out.transpose(),delta4)*sigmoidGradient(Z_3[:,t][:,np.newaxis]) # 
        delta2 = np.dot(hid_to_hid2.transpose(),delta3)*sigmoidGradient(Z_2[:,t][:,np.newaxis])

    ### The bias term needs to be included below here. All it is is bias_L(i) += delta_L(i+1)

        tempGrad_1bias += (1/mbatch)*delta2; #  Multiplying by (1/m) here rather than later because *= return a type error. But, multiplication is distributive anyways.
        tempGrad_2bias += (1/mbatch)*delta3; #  See ^ for why (1/m)
        tempGrad_3bias += (1/mbatch)*delta4;
        tempGrad_1 += (1/mbatch)*(delta2)*(X[:,t][:,np.newaxis].transpose()); #
        tempGrad_2 += (1/mbatch)*(delta3)*(A_2[:,t].transpose()); #
        tempGrad_3 += (1/mbatch)*(delta4)*(A_3[:,t].transpose());
    tempGrad_1 += ((lda/mbatch)*vis_to_hid) ## Regularize.
    tempGrad_2 += ((lda/mbatch)*hid_to_hid2) ## Regularize.
    tempGrad_3 += ((lda/mbatch)*hid2_to_out)
    
    return paramsGrad        


def predictf(params,X):
    
    if np.shape(np.shape(X))[0] == 1:
        X = X[:,np.newaxis]
    else:
        pass
    
    vis_to_hid = params[:nvis*nhid].reshape((nhid,nvis)) ## theta1 excluding bias term
    hid_bias = params[nvis*nhid:nvis*nhid+nhid] ##  bias term for vis_to_hid
    hid_to_hid2 = params[:nhid*nhid2].reshape((nhid2,nhid))
    hid2_bias = params[nvis*nhid + nhid + nhid*nhid2:nvis*nhid + nhid + nhid*nhid2 + nhid2]
    hid2_to_out = params[nvis*nhid + nhid + nhid*nhid2 + nhid2:nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout].reshape((nout,nhid2)) ## 
    out_bias = params[nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout:nvis*nhid + nhid + nhid*nhid2 + nhid2 + nhid2*nout + nout] ## bias for ^
    
    Z_2 = np.dot(vis_to_hid,X) + hid_bias ##
    A_2 = sigmoid(Z_2);
    Z_3 = np.dot(hid_to_hid2,A_2) + hid2_bias
    A_3 = sigmoid(Z_3)
    Z_4 = np.dot(hid2_to_out,A_3) + out_bias; ## 
    A_4 = sigmoid(Z_4); ## 
    
    mx = np.argmax(A_4,axis=0)[:,np.newaxis]

    return mx

def minim_batch(params,df,num_epochs,batch_size,opt_plt=False,lda=0,momen=0.9):
    print('momentum=',momen,'lda=',lda, 'num_epochs=',num_epochs,'batch_size=',batch_size)
    v_t1 = 0
    #### Option plot true --> plot cost vs num_iters
    if opt_plt:
        y_vals = np.zeros([num_epochs,1])
        x_vals = np.zeros([num_epochs,1])

        for ep in range(num_epochs):
            dfshuff = shuffle(df)
            dfshuffX_ = dfshuff.drop('label',axis=1).values
            dfshuffY = dfshuff['label'].values[:,np.newaxis]
            dfshuffX = dfshuffX_.transpose()
            
            print('__________', 'epoch =',ep+1)
            print(np.shape(dfshuffY))
            print(np.shape(dfshuffX))            
            for i in range(0,m,batch_size):
                dfXbatch = dfshuffX[:,i:i+batch_size]
                dfYbatch = dfshuffY[i:i+batch_size]
                #print(np.shape(dfXbatch))
                gp1 = gradCostFunction(params,dfXbatch,dfYbatch,lda,num_labels);
                params, v_t1 = DescentMomentum(params,gp1,v_t1,momen,0.03)  ## last number is the learning rate
                p1 = CostFunction(params,dfXbatch,dfYbatch,lda,num_labels);
            
            ## Plot cost per epoch
            y_vals[ep] = p1
            x_vals[ep] = ep+1
        ####
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel("number of epochs")
        ax1.set_ylabel("J(theta)")
        ax1.plot(x_vals,y_vals)
        ###
    else:
        for i in range(num_iters):
            gp1 = gradCostFunction(params,X,y,lda,num_labels);
            params, v_t1 = DescentMomentum(params,gp1,v_t1,momen,0.5)
            p1 = CostFunction(params,X,y,lda,num_labels);

    return params, p1
###
def verif(X,y):
    testm = np.shape(y)[0]
    num = 0
    for i in range(testm):
        if abs(X[i][0] - y[i][0])<(1e-4):
            num+=1
        else:
            pass
    err = (1 - (num/testm))*100
    return err
####################
#makepic(dfX_,dfy,35) ### dfX_ has shape 5000x784, dfX has shape 784x5000
params_min, mincost = minim_batch(params,df_init,num_epochs=14,batch_size=100,opt_plt=True,lda=0.1,momen=0.9)

###### Checking back prop
#apl = gradCostFunction(params,dfX,dfy,0,num_labels)
#banana = numerGrad(params,dfX,dfy,0,num_labels)
#orange = np.concatenate((apl,banana),axis=1)
#print(orange[0:10])
################################### Back prop works as it should.

#print(np.shape(dfX[:,35]))
print('____Now test and verify_____')
predX = predictf(params_min,dfcrsX)
predtst = predictf(params_min,dftestX)
print('final cost=',mincost)
print('% error, cross val set=',verif(predX,dfcrsy),'%')
print('% error, cross val set=',verif(predtst,dftesty),'%')

print(time.time()-t0,'s')
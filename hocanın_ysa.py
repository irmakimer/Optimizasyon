import numpy as np
import math
from datasetSpiral import ti,yi
#----------------------------------------
def exp(x):
    return np.array([math.exp(i) for i in x])
#-----------------------------------------
def tanh(x):
    if isinstance(x,float):
        result = (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    else:
        result = ((np.array(exp(x))-np.array(exp(-x)))/(np.array(exp(x))+np.array(exp(-x)))).reshape(-1,1)
        return result
#------------------------------------------------------------
def MISOYSAmodelIO(ti,Wg,bh,Wc,bc):
    S=Wg.shape[0]
    yhat=[]
    for t in ti:
        t=t.reshape(-1,1)
        nn=Wc.dot(tanh(Wg.dot(t)+bh))+bc
        yhat.append(nn[0][0])
    return yhat
#----------------------------------------------------------------
def error(Wg,bh,Wc,bc,ti,yi):
    yhat=MISOYSAmodelIO(ti, Wg, bh, Wc, bc)
    return np.array(yi)-np.array(yhat)
#-----------------------------------------------------
def findJacobian(traininginput , Wg,bh,Wc,bc):
    R=traininginput.shape[1]
    S =Wg.shape[0]
    numofdata=len(traininginput)
    J= np.matrix(np.zeros((numofdata,S*(R+2)+1)))
    for i in range(0,numofdata):
        for j in range(0,S*R):
            k=np.mod(j,S)
            m=int(j/S)
            J[i,j]=-Wc[0,k]*traininginput[i,m]*(1-tanh(Wg[k,:].dot(traininginput[i])+bh[k])**2)
        for j in range(S*R,S*R+S):
            J[i,j]=-Wc[0,j-S*R]*(1-tanh(Wg[j-S*R,:].dot(traininginput[i])+bh[j-S*R])**2)
        for j in range(S*R+S,S*(R+2)):
            J[i,j]=-tanh(Wg[j-(R+1)*S,:].dot(traininginput[i])+bh[j-(R+1)*S])
        J[i,S*(R+2)]=-1
    return J
#---------------------------------------------------------
def Matrix2Vector(Wg,bh,Wc,bc):
    x=np.array([],dtype=float).reshape(0,1)
    for i in range(0,Wg.shape[1]):
        x=np.vstack((x,Wg[:,i].reshape(-1,1)))
    x=np.vstack((x,bh.reshape(-1,1)))
    x=np.vstack((x,Wc.reshape(-1,1)))
    x=np.vstack((x,bc.reshape(-1,1)))
    x=x.reshape(-1,)
    return x
#--------------------------------------------------------------
def Vector2Matrix(z,S,R):
    Wgz=np.array([],dtype=float.reshape(S,0))
    for i in range(0,R):
        T=(z[i*S:(i+1)*S]).reshape(S,1)
        
        
        

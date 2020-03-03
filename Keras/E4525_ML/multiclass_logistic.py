import datetime

import numpy as np
import numpy.random as random
import scipy.optimize as optimize
from sklearn.preprocessing import LabelEncoder
import sklearn.utils.extmath as extmath

def logisticClassProbability(X,b,W):
    logits=extmath.safe_sparse_dot(X,W.T,dense_output=True)+b
    elogits=np.exp(logits-logits.max(axis=1)[:,np.newaxis]) # make calculation more stable numerically
    elogits_sum=elogits.sum(axis=1)
    class_probs=elogits/elogits_sum[:,np.newaxis]
    return class_probs
def logisticLoss(X,Z,b,W):
    class_probs=logisticClassProbability(X,b,W)
    loss= np.mean(-(Z*np.log(np.maximum(class_probs,1e-10))).sum(axis=1))
    return loss
def logisticGradient(X,Z,b,W):
    class_probs=logisticClassProbability(X,b,W)
    delta=Z-class_probs
    return -delta.sum(axis=0),-extmath.safe_sparse_dot(delta.T,X,dense_output=True)


def val_func(x0,X,Z,penalty):
    D=X.shape[1]
    K=Z.shape[1]
    b=x0[:K]
    W=x0[K:].reshape((K,D))
    loss=logisticLoss(X,Z,b,W)
    if penalty>0:
        loss+=0.5*penalty*(W**2).sum()
    return loss

def grad_func(x0,X,Z,penalty):
    D=X.shape[1]
    K=Z.shape[1]
    b=x0[:K]
    W=x0[K:].reshape((K,D))
    gradb,gradW=logisticGradient(X,Z,b,W)
    if penalty>0:
        gradW+=penalty*W
    return np.concatenate([gradb,gradW.ravel()])

def optimize_logistic_weights_scipy(X,Z,b,W,penalty=0,
                                    method="newton-cg",
                                    tol=1e-16,
                                    max_iter=100):
    D=X.shape[1]
    K=Z.shape[1]
    x0=np.concatenate((b,W.ravel()))
   
    fit=optimize.minimize(val_func, x0, args=(X,Z,penalty),jac=grad_func,
             method=method,   
             tol=tol)
    x1=fit.x
    b=x1[:K]
    W=x1[K:].reshape((K,D))
    return b,W
def report_function(e,params,X,Z,X_val,Z_val,penalty):
    D=X.shape[1]
    K=Z.shape[1]
    b=params[:K]
    W=params[K:].reshape((K,D))
    N=min(1000,X.shape[0])
    perm=np.random.choice(X.shape[0],N)
    loss=val_func(params,X[perm],Z[perm],penalty)
    class_probs=logisticClassProbability(X[perm],b,W)
    Y_pred=class_probs.argmax(axis=1) 
    Y=Z[perm].argmax(axis=1)
    train_accuracy=np.mean(Y_pred==Y)
    date_str=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    msg=f"{date_str}|"
    msg+=f"\t{e}: TRAIN loss {loss:.4f},  acc {train_accuracy:.4f}"
    if not(X_val is None): # if we have a valuation set we can report
                        # how well we are doing out of sample
        N=min(1000,X_val.shape[0])
        perm=np.random.choice(X_val.shape[0],N)
        val_e=val_func(params,X_val[perm],Z_val[perm],penalty)
        class_probs=logisticClassProbability(X_val[perm],b,W)
        Y_pred=class_probs.argmax(axis=1)          
        Y_val=Z_val[perm].argmax(axis=1)     
        val_accuracy=np.mean(Y_pred==Y_val)
        msg+=f" || VAL loss {val_e:.4f}, acc {val_accuracy:.4f}"
    print(msg)


### Logistic Regression using Stochastic Gradient Descent
def optimize_logistic_weights(X,Z,b,W,
                            X_val=None,
                            Z_val=None,
                            penalty=0,
                            learning_rate=0.01,
                            tol=1e-8,
                            max_iter=1000,
                            batch_size=100,
                            verbose=True
                              ):
    
    D=X.shape[1]
    K=Z.shape[1]
    x=np.concatenate((b,W.ravel()))
    if Z_val is not None:
        Y_val=Z_val.argmax(axis=1)
    N=X.shape[0]
    l0=val_func(x,X,Z,penalty)
    for e in range(max_iter):
        if (e%(max_iter//10)==0 and verbose):
            report_function(e,x,X,Z,X_val,Z_val,penalty)
        perm=random.permutation(N)
        for i in range(0,N,batch_size):
            Xb=X[perm[i:i+batch_size]]
            Zb=Z[perm[i:i+batch_size]]
            p=Xb.shape[0]/N*penalty
            grad=grad_func(x,Xb,Zb,p)
            x-=learning_rate*grad
        l=val_func(x,X,Z,penalty)
       
        d=np.abs(l-l0)
        if d<tol*l0:
            break
        l0=l  
    if verbose:
        report_function(e,x,X,Z,X_val,Z_val,penalty)
    b=x[:K]
    W=x[K:].reshape((K,D))
    return b,W   

class LogisticGDClassifier:
    def __init__(self,
        penalty=0,
        learning_rate=0.005,
        batch_size=100,
        tol=1e-4,
        max_iter=100,
        verbose=True
    ):
        self.penalty=penalty
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.tol=tol
        self.max_iter=max_iter
        self.verbose=verbose
    def fit(self,X,Y,X_val=None,Y_val=None):
        # X_val and Y_val are only used to report accuracy during optimization
        # they do not affect the fitted W,b parameters
        
        
        if X.ndim==1:
            X=X.reshape(1,-1)
        N,D=X.shape
       
        self.encoder=LabelEncoder()
        y=self.encoder.fit_transform(Y)      
        K=len(self.encoder.classes_)
        Z=np.zeros((N,K),dtype=int)
        Z[np.arange(N),y]=1
        
        
        if not(X_val is None):
            N_val=X_val.shape[0]
            y_val=self.encoder.transform(Y_val)  
            Z_val=np.zeros((N_val,K),dtype=int)
            Z_val[np.arange(N_val),y_val]=1
        else: 
            Z_val=None

        b_guess=np.zeros(K)
        W_guess=random.normal(0,1,(K,D))/np.sqrt(D)
        
        self.b,self.W=optimize_logistic_weights(X,Z,b_guess,W_guess,
                                            X_val=X_val,
                                            Z_val=Z_val,
                                            penalty=self.penalty,
                                            learning_rate=self.learning_rate,
                                            batch_size=self.batch_size,
                                            tol=self.tol,
                                            max_iter=self.max_iter,
                                            verbose=self.verbose
                                            )
        
    def predict(self,X):
        if X.ndim==1:
            X=X.reshape(1,-1)
        N,D=X.shape   
        class_probs=logisticClassProbability(X,self.b,self.W)        
        y=class_probs.argmax(axis=1)
        return self.encoder.inverse_transform(y)
    def predict_proba(self,X):
        if X.ndim==1:
            X=X.reshape(1,-1)
        N,D=X.shape   
        class_probs=logisticClassProbability(X,self.b,self.W)
        return class_probs 
        

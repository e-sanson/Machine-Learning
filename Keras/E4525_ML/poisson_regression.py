import datetime

import numpy as np
import numpy.random as random
import scipy.special as special
import scipy.optimize as optimize

def poisson_lambda(theta,X,offset): 
    nu=np.dot(X,theta.T) +offset[:,np.newaxis]
    #print("nu",nu.shape)
    l=np.exp(nu)
    return l



def PoissonError(theta0,X,Y,offset):
    K=Y.shape[1]
    theta=theta0.reshape(K,-1)
    l=poisson_lambda(theta,X,offset)
    loss=l-Y*np.log(l)
    #print("l",l.shape,"Y",Y.shape,"loss",loss.shape)
    return loss.mean()

def PoissonErrorGradient(theta0,X,Y,offset): 
    K=Y.shape[1]
    theta=theta0.reshape(K,-1)
    l=poisson_lambda(theta,X,offset)
    dloss=l-Y
    ##print("l",l.shape,"Y",Y.shape,"dloss",dloss.shape)
    grad=np.dot(dloss.T,X)
    #print("grad",grad.shape)
    return grad/len(X)/K

def PoissonErrorGradientFlattened(theta,X,Y,offset):
    return  PoissonErrorGradient(theta,X,Y,offset).ravel()

class PoissonRegression:
    def __init__(self,offset_col=None):
            self.offset_col=offset_col
    def get_offset(self,X):
        X=X.reshape(len(X),-1)
        ones=np.ones((len(X),1))
        print(ones.shape,X.shape)
        if self.offset_col!=None:
            x_used=np.c_[ones,X[:,:self.offset_col],X[:,self.offset_col+1:]]
            offset=np.log(X[:,self.offset_col])
        else:
            print(ones.shape,X.shape)
            x_used=np.c_[ones,X]
            offset=np.zeros((len(X)))
        return x_used,offset
    def fit(self,X,Y):
        x_used,offset=self.get_offset(X)
        N,D=x_used.shape
        if len(Y.shape)!=2:
            Y=np.reshape(Y,(N,-1))
            #print("Y",Y.shape)
        K=Y.shape[1]
        theta0=np.zeros((K,D))
        #print("theta",theta0.shape)
        fit=optimize.minimize(PoissonError,theta0.ravel(),jac=PoissonErrorGradientFlattened,args=(x_used,Y,offset),method="bfgs")
        #print(fit)
        self.theta=fit.x.reshape(K,-1)
        return self
    def predict(self,X):
        x_used,offset=self.get_offset(X)
        l=poisson_lambda(self.theta,x_used,offset)
        return l

def count_error(l,Y):
    err=np.sum(np.abs(l-Y))
    return err/len(Y)

def report_function(e,params,X,Y,offset,X_val,Y_val,offset_val):
    N=X.shape[0]
    perm=np.random.choice(X.shape[0],N)
    loss=PoissonError(params,X[perm],Y[perm],offset[perm])
    l=poisson_lambda(params,X[perm],offset[perm])
    err=(l-Y[perm])**2/l
    train_dispersion=np.mean(err)
    count_err=count_error(np.round(l),Y)
    date_str=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    msg=f"{date_str}|"
    msg+=f"\t{e}: TRAIN loss {loss:.4f},  disp {train_dispersion:.4f}, err {count_err:.4f}"
    if not(X_val is None):
        perm=np.random.choice(X_val.shape[0],N)
        val_loss=PoissonError(params,X_val[perm],Y_val[perm],offset_val[perm])
        l=poisson_lambda(params,X_val[perm],offset_val[perm])
        err=(l-Y_val[perm])**2/l
        val_dispersion=np.mean(err)
        val_err=count_error(np.round(l),Y)
        msg+=f" || VAL loss {val_loss:.4f}, disp {val_dispersion:.4f}, err {val_err:.4f}"
    print(msg)


### Logistic Regression using Stochastic Gradient Descent
def optimize_poisson(X,Y,offset,params,
                            X_val=None,
                            Y_val=None,
                            offset_val=None,
                            penalty=0.0,
                            learning_rate=0.01,
                            tol=1e-8,
                            max_iter=1000,
                            batch_size=100,
                            verbose=True
                              ):
    
    #params=params.ravel()
    N=X.shape[0]
    l0=PoissonError(params,X,Y,offset)
    for e in range(max_iter):
        if (e%(max_iter//10)==0 and verbose):
            report_function(e,params,X,Y,offset,X_val,Y_val,offset_val)
        perm=random.permutation(N)
        for i in range(0,N,batch_size):
            Xb=X[perm[i:i+batch_size]]
            Yb=Y[perm[i:i+batch_size]]
            expb=offset[perm[i:i+batch_size]]
            grad=PoissonErrorGradient(params,Xb,Yb,expb)
            params-=learning_rate*grad
            params[1:]*=(1-penalty)
        l=PoissonError(params,X,Y,offset)
       
        d=np.abs(l-l0)
        if d<tol*l0:
            break
        l0=l  
    if verbose:
        report_function(e,params,X,Y,offset,X_val,Y_val,offset_val)
    
    return params

class PoissonRegressionSGD:
    def __init__(self,
        exposure_col=None,
        penalty=0,
        learning_rate=0.005,
        batch_size=100,
        tol=1e-4,
        max_iter=1000,
        verbose=True
    ):
        self.offset_col=exposure_col # offset is log of exposure
        self.penalty=penalty
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.tol=tol
        self.max_iter=max_iter
        self.verbose=verbose
    def get_offset(self,X):
        X=X.reshape(len(X),-1)
        ones=np.ones((len(X),1))
        if self.offset_col!=None:
            x_used=np.c_[ones,X[:,:self.offset_col],X[:,self.offset_col+1:]]
            print("x_used",x_used.shape)
            offset=np.log(X[:,self.offset_col])
        else:
            x_used=np.c_[ones,X]
            offset=np.zeros((len(X)))
        return x_used,offset
    def fit(self,X,Y,X_val=None,Y_val=None):
        x_used,offset=self.get_offset(X)
        N,D=x_used.shape
        if len(Y.shape)!=2:
            Y=np.reshape(Y,(N,-1))
        K=Y.shape[1]
        #theta0=np.random.normal(0,1/np.sqrt(D),D)
        theta0=np.zeros((K,D))
        
        if not(X_val is None):
            x_val_used,offset_val=self.get_offset(X_val)
        else: 
            x_val_used=None
            offset_val=None
        
        self.theta=optimize_poisson(x_used,Y,offset,theta0,
                                            X_val=x_val_used,
                                            Y_val=Y_val,
                                            offset_val=offset_val,
                                            penalty=self.penalty,
                                            learning_rate=self.learning_rate,
                                            batch_size=self.batch_size,
                                            tol=self.tol,
                                            max_iter=self.max_iter,
                                            verbose=self.verbose
                                            )
        #self.theta=theta.reshape((K,D))
        return self
    def predict(self,X):
        x_used,offset=self.get_offset(X)
        l=poisson_lambda(self.theta,x_used,offset)
        return np.round(l)


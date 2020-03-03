import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin,RegressorMixin

epsilon=1e-16

def regression_loss(G,H):
    return -0.5*G**2/max(H,epsilon)

def binomial_loss(G,H):
    return -G*np.log(max(G/H,epsilon))

def crossentropy_loss(G,H):
    return - np.sum(G *np.log(np.maximum(G/H,epsilon)))

def gini_loss(G,H):  
    return - np.sum(G**2)/max(H,epsilon)


def compute_cuts(X,G,H,loss_func):
    T=len(X)
    result=[]
    G1=G.sum(axis=0)
    H1=H.sum()
    G0=np.zeros_like(G1)
    H0=0.0
    indexes=X.argsort()   
    loss=loss_func(G1,H1)
    X_last=X[indexes[0]]
    result.append((X_last*0.99,loss))
    i=0
    while 1:
        while (i<T) and (X[indexes[i]]==X_last):
            index=indexes[i]
            G1-=G[index]
            H1-=H[index]
            G0+=G[index]
            H0+=H[index]
            i+=1
        loss_left=loss_func(G0,H0)
        loss_right=loss_func(G1,H1)
        loss=(loss_left+loss_right)
        cut=(X_last+X[indexes[min(i,T-1)]])/2
        result.append((cut,loss))
        if i>=T:
            break
        X_last=X[indexes[i]]
    return np.array(result) 

def select_cut(X,G,H,loss_func):
    T=len(X)
    G1=G.sum(axis=0)
    H1=H.sum()
    G0=0.0
    H0=0.0    
    indexes=X.argsort()   
    loss0=loss_func(G1,H1)   
    X_last=X[indexes[0]]   
    min_loss=loss0
    min_cut=X_last    
    i=0
    while 1:
        while (i<T-1) and (X[indexes[i]]==X_last):
            index=indexes[i]
            G1-=G[index]
            H1-=H[index]
            G0+=G[index]
            H0+=H[index]
            i+=1
        loss_left=loss_func(G0,H0)
        loss_right=loss_func(G1,H1)
        loss=(loss_left+loss_right)
        X_next=X[indexes[i]]
        if loss<min_loss:
            min_loss=loss
            min_cut=(X_last+X_next)/2
        if i>=T-1:
            break
        X_last=X[indexes[i]]
    return min_cut,min_loss-loss0      

def select_branch(X,G,H,max_features,loss_func):
    N,D=X.shape
    min_var="Bad"
    min_loss=1e9
    min_cut=1e9
    variables=np.random.choice(np.arange(D),max_features,replace=False)
    for var in variables:
        cut,loss=select_cut(X[:,var],G,H,loss_func)
        if loss<min_loss:
            min_loss=loss
            min_var=var
            min_cut=cut
    return min_var,min_cut,min_loss

# this returns a tree 
# every branch contains
#   1. Variable to branch on
#   2. threshold
#   3. Left and Right brachces of tree.
# Terminal nodes (with no further branches) return the sample counts for each target label within that branch.
def build_tree(X,G,H,depth,max_features,gamma,loss_func):
    G1=G.sum(axis=0)
    H1=max(H.sum(),1e-16)
    N=len(G)
    if N<2  or depth==0:
        return G1/H1 # we return minus the squared gradient
    var,X_cut,loss_decrease=select_branch(X,G,H,max_features,loss_func)
    left_idx=X[:,var]<X_cut
    right_idx=np.logical_not(left_idx)
    if -2*loss_decrease<=gamma: # we do not a brunch in the loss those not decrease enough
                                # the factor of 2 is to match behavior of xgboost classifier
                                # but does not match the derivation on xgboost paper.
        return  G1/H1
    X_left=X[left_idx]
    X_right=X[right_idx]
    G_left=G[left_idx]
    G_right=G[right_idx]
    H_left=H[left_idx]
    H_right=H[right_idx]
    left=build_tree(X_left,G_left,H_left,depth-1,max_features,gamma,loss_func)
    right=build_tree(X_right,G_right,H_right,depth-1,max_features,gamma,loss_func)
    
    return ( var,X_cut,
             (left,right)
             )

def value_point(x,tree):  
    if not(isinstance(tree,tuple)):
        return tree
    var,threshold,children=tree
    x_val=x[var]
    branch=1
    if x_val<threshold:
        branch=0
    return value_point(x,children[branch])
        
def value_points(X,nclasses,tree):
    results=np.empty((len(X),nclasses),dtype=np.float)
    for i1 in range(len(X)):
        results[i1]=value_point(X[i1],tree)
    return results

class RegressionTree(BaseEstimator,RegressorMixin):
    def __init__(self,max_depth=3,max_features=None,gamma=0):
        self.max_depth=max_depth
        self.max_features=max_features
        self.gamma=max(gamma,epsilon)
    def fit(self,X,Y,sample_weight=None): 
        if sample_weight is None:
            sample_weight=np.ones(len(Y))
        max_features=self.max_features
        if max_features is None:
            max_features=X.shape[1]
        G=sample_weight*Y
        H=sample_weight
        self.tree=build_tree(X,G,H,self.max_depth,max_features,self.gamma,regression_loss)
        return self
    def predict(self,X):
        return value_points(X,1,self.tree).ravel()

class ClassificationTree(BaseEstimator,ClassifierMixin):
    def __init__(self,max_depth=3,loss=gini_loss,max_features=None,gamma=0):
        self.max_depth=max_depth
        self.loss=loss
        self.max_features=max_features
        self.gamma=max(gamma,epsilon)
    def fit(self,X,Y,sample_weights=None):
        # default sample weights is all points equally sampled
        if sample_weights is None:
            sample_weights=np.ones(len(Y))
        max_features=self.max_features
        if max_features is None:
            max_features=X.shape[1]
        # one-hot encode labels
        self.classes_, y = np.unique(Y, return_inverse=True)
        N=len(X)
        K=len(self.classes_)
        Z=np.zeros((N,K),dtype=np.int)
        Z[np.arange(N),y]=1
        
        G = sample_weights[:,np.newaxis]*Z
        H = sample_weights
        self.tree=build_tree(X,G,H,self.max_depth,max_features,self.gamma,self.loss)
        return self
    def predict_proba(self,X):
        #if not(self.features is None):
        #    X=X[:,self.features]
        return value_points(X,len(self.classes_),self.tree)
    def predict(self,X):
        Z=self.predict_proba(X)
        return self.classes_[np.argmax(Z,axis=1)]

class RandomForest(BaseEstimator,ClassifierMixin):
    def __init__(self,max_depth=3,n_estimators=10,loss=gini_loss,max_features=None,gamma=0):
        self.max_depth=max_depth
        self.n_estimators=n_estimators
        self.loss=loss
        self.max_features=max_features
        self.gamma=gamma
    def fit(self,X,Y):
        self.trees=[]
        N,K=X.shape
        for i1 in range(self.n_estimators):
            tree=ClassificationTree(max_depth=self.max_depth,
                              max_features=self.max_features,
                              gamma=self.gamma,
                              loss=self.loss)
            indexes=np.random.choice(N,N,replace=True)
            X_used=X[indexes]
            Y_used=Y[indexes]
            tree.fit(X_used,Y_used)
            self.trees.append(tree)
        self.classes_=self.trees[0].classes_
        return self
    def predict_proba(self,X):
        p=np.zeros((len(X),len(self.classes_)),dtype=np.float)
        for tree in self.trees:
            p+=tree.predict_proba(X)
        return p/self.n_estimators
    def predict(self,X):
        Z=self.predict_proba(X)
        return self.classes_[np.argmax(Z,axis=1)]

def logistic_loss(eta,X,Y):
    l=np.log(1+np.exp(eta)) - eta*Y
    return l.sum()

def Y_hat(eta):
    return 1.0/(1.0+np.exp(-eta))

def logistic_gradient(Y_hat,Y):    
    return (Y_hat-Y)

def logistic_curvature(Y_hat,Y):
    return Y_hat*(1-Y_hat)

class GradientTree:
    def __init__(self,max_features, max_depth=3,gamma=0):
        self.max_depth=max_depth
        self.max_features=max_features
        self.gamma=max(gamma,epsilon)
    def fit(self,X,G,H): 
        self.tree=build_tree(X,G,H,
                                  depth=self.max_depth,
                                  max_features=self.max_features,
                                  gamma=self.gamma,
                                  loss_func=regression_loss
                                 )
        return self
    def __call__(self,X):
        return value_points(X,1,self.tree).ravel()



def predict_additive_model(X,models,eta0):
    eta=eta0*np.ones(len(X))
    for model in models:
        eta-=model(X)
    return eta


class LogitBoostedTree(BaseEstimator,ClassifierMixin):
    def __init__(self,n_estimators=10,max_depth=3,max_features=None,gamma=0,verbose=0):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.max_features=max_features
        self.gamma=gamma
        self.verbose=verbose
    def fit(self,X,Y):
        max_features=self.max_features
        if max_features is None:
            max_features=X.shape[1]
        Y1=Y.mean()
        self.eta0=np.log ( Y1/(1-Y1))
        #print(self.eta0)
        eta=self.eta0*np.ones(len(Y))
        #print("eta",eta[:10])
        if self.verbose:
            print("step = 0 loss =",logistic_loss(eta,X,Y))
        self.trees=[]
        for i in range(self.n_estimators):
            Y_pred=Y_hat(eta)
            #print("Y_pred",Y_pred[:20])
            G=logistic_gradient(Y_pred,Y)
            H=logistic_curvature(Y_pred,Y)
            #print("G = ",G[:10],"\nH=",H[:10])
            gt=GradientTree(max_features=max_features,
                            max_depth=self.max_depth,
                            gamma=self.gamma)
            gt.fit(X,G,H)
            deta=gt(X)
            #print(deta[:10])
            eta-=deta
            if self.verbose:
                print("step =",i+1 ," loss =",logistic_loss(eta,X,Y))
            self.trees.append(gt)
        return self
    def predict_proba(self,X):
        eta=predict_additive_model(X,self.trees,self.eta0)
        return Y_hat(eta)
    def predict(self,X):
        return self.predict_proba(X)>0.5


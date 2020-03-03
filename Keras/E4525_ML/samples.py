import numpy as np
import numpy.random as random



# data is in two concentric ellipsis, plus some noise
def ellipsoid(N,R0=4,R1=6,theta0=np.pi/4):
    
    DR=[1.,-1.]
    dR=0.75
    dtheta=np.pi/10
    

    X = np.zeros((N*2,2))
    y = np.zeros(N*2, dtype='uint8')
    for j in range(2):
      ix = range(N*j,N*(j+1))
      rA=R0+DR[j]+np.random.randn(N)*dR
      rB=R1+DR[j]+np.random.randn(N)*dR
      theta = np.linspace(0,np.pi*2,N)+np.random.randn(N)*dtheta 
      X[ix] = np.c_[rA*np.cos(theta)*np.cos(theta0)+rB*np.sin(theta)*np.sin(theta0), 
                    -rA*np.cos(theta)*np.sin(theta0)+rB*np.sin(theta)*np.cos(theta0)]
      y[ix] = j
    return X,y

def gaussian_mixture(N_samples,pi, mu,sigma):
    K,D=mu.shape
    # Z is a hot encoded multinomial with K possible classes
    # Z is N x K 
    Z=np.random.multinomial(1,pi,N_samples)
    gaussians=np.empty((K,N_samples,D))
    for k in range(K):
          gaussian=np.random.multivariate_normal(mu[k],sigma[k],size=N_samples)
          gaussians[k]=gaussian
    # Z.T[...,np.newaxis] will be 3 x N_samples x 1
    # Gaussian is                 3 x N_samples x D
    # P_{k,i,d } =  Z_{i,k} * G_{k,i,d}
    prod=Z.T[...,np.newaxis]*gaussians
    X=np.sum(prod,axis=0)
    return X,Z

def generate_logistic_multinomial(X,W,b):
    X1=np.c_[np.ones(len(X)),X]
    bW=np.c_[b,W]
    nu=np.dot(X1,bW.T)
    enu=np.exp(nu)
    enu_sum=enu.sum(axis=1)
    pi=enu/enu_sum[:,np.newaxis]
    Z=np.empty_like(pi)
    for i1 in range(len(pi)):
        Z[i1]=random.multinomial(1,pi[i1],1)
    return Z
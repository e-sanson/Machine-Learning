import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def decision_boundary_2d(model,X,Y,ax,cmap=plt.cm.Spectral,alpha=0.5,slack=0.05):
    x1=X[:,0]
    x2=X[:,1]
    dx1=x1.max()-x1.min() 
    x1_min, x1_max = x1.min() - slack*dx1, x1.max() + slack*dx1
    dx2=x2.max()-x2.min()
    x2_min, x2_max =x2.min() - slack*dx2, x2.max() + slack*dx2
    hx1 = dx1/101
    hx2=dx2/101
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, hx1),
                     np.arange(x2_min, x2_max, hx2))

    XX=np.c_[xx1.ravel(),xx2.ravel()]
    yy = model.predict(XX)
    ndim=yy.ndim
    if ndim>XX.ndim: # one hot encoded, find index of prediction
        yy=yy.argmax(axis=ndim-1)
        Y=Y.argmax(axis=1)
    else:
        encoder=LabelEncoder()
        Y=encoder.fit_transform(Y)
        yy=encoder.transform(yy)
        
    yy =yy.reshape(xx1.shape)

    ax.contourf(xx1, xx2, yy, cmap=cmap, alpha=alpha)
    ax.scatter(x1, x2, c=Y, s=20,edgecolors='k' , cmap=cmap)
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

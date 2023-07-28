import numpy as np
from keras.models import Model
from keras.layers import *
import scipy.io
import numpy as np
from keras import backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def yc_patch(A,l1,l2,o1,o2):

    n1,n2=np.shape(A);
    tmp=np.mod(n1-l1,o1)
    if tmp!=0:
        #print(np.shape(A), o1-tmp, n2)
        A=np.concatenate([A,np.zeros((o1-tmp,n2))],axis=0)

    tmp=np.mod(n2-l2,o2);
    if tmp!=0:
        A=np.concatenate([A,np.zeros((A.shape[0],o2-tmp))],axis=-1);


    N1,N2 = np.shape(A)
    X=[]
    for i1 in range (0,N1-l1+1, o1):
        for i2 in range (0,N2-l2+1,o2):
            tmp=np.reshape(A[i1:i1+l1,i2:i2+l2],(l1*l2,1));
            X.append(tmp);
    X = np.array(X)
    return X[:,:,0]

def yc_patch_inv(X1, n1, n2, l1, l2, o1, o2):
    tmp1 = np.mod(n1 - l1, o1)
    tmp2 = np.mod(n2 - l2, o2)
    if (tmp1 != 0) and (tmp2 != 0):
        A = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))
        mask = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))

    if (tmp1 != 0) and (tmp2 == 0):
        A = np.zeros((n1 + o1 - tmp1, n2))
        mask = np.zeros((n1 + o1 - tmp1, n2))

    if (tmp1 == 0) and (tmp2 != 0):
        A = np.zeros((n1, n2 + o2 - tmp2))
        mask = np.zeros((n1, n2 + o2 - tmp2))

    if (tmp1 == 0) and (tmp2 == 0):
        A = np.zeros((n1, n2))
        mask = np.zeros((n1, n2))

    N1, N2 = np.shape(A)
    ids = 0
    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            # print(i1,i2)
            #       [i1,i2,ids]
            A[i1:i1 + l1, i2:i2 + l2] = A[i1:i1 + l1, i2:i2 + l2] + np.reshape(X1[:, ids], (l1, l2))
            mask[i1:i1 + l1, i2:i2 + l2] = mask[i1:i1 + l1, i2:i2 + l2] + np.ones((l1, l2))
            ids = ids + 1

    A = A / mask;
    A = A[0:n1, 0:n2]
    return A

def BlockA(inpt_img,D1, dropout):
    C1 = Conv2D(D1, 3, padding = 'same')(inpt_img)
    C1 = BatchNormalization()(C1)
    C1 = Activation('relu')(C1)
    C1 = Dropout(dropout)(C1)
    return C1

def DenseBlock(x0, n_layers, m,dropout):
    skip_connection_list = []
    for j in range(n_layers):
        xl = BlockA(x0, m, dropout=dropout)
        x0 = concatenate([x0, xl])
        skip_connection_list.append(xl)

    for i in range(0, len(skip_connection_list)):
        if i == 0:
            out = skip_connection_list[i]
        else:
            out = concatenate([out, skip_connection_list[i]])

    return out

def TD(inp,filters,stride):
    
    x = Conv2D(filters, 3, padding = 'same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(stride)(x)
    return x

def TU(inpt_img,stride,D1, dropout):
    C1 = Conv2DTranspose(D1, 3, padding = 'same', strides=stride)(inpt_img)
    C1 = BatchNormalization()(C1)
    C1 = Activation('relu')(C1)
    C1 = Dropout(dropout)(C1)
    return C1

def SK(inputs, m=2, r=8, L=32, kernel=4):
    d = max(int(kernel / r), L)
    
    out = Conv2D(kernel, 1, strides=1, padding='same')(inputs)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    x1 = Conv2D(kernel, 3, strides=1, padding='same')(out)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    _x1 = GlobalAveragePooling2D()(x1)

    x2 = Conv2D(kernel, 5, strides=1, padding='same')(out)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    _x2 = GlobalAveragePooling2D()(x2)

    U = Add()([_x1, _x2])

    z = Dense(d, activation='relu')(U)
    z = Dense(kernel*2)(z)

    z = Reshape([1, 1, kernel, m])(z)
    scale = Softmax()(z)

    x = Lambda(lambda x: tf.stack(x, axis=-1))([x1, x2])
    r = multiply([scale, x])
    r = Lambda(lambda x: K.sum(x, axis=-1))(r)
    return r

def cseis():
    from matplotlib.colors import ListedColormap
    import numpy as np
    seis=np.concatenate(
(np.concatenate((0.5*np.ones([1,40]),np.expand_dims(np.linspace(0.5,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose(),
np.concatenate((0.25*np.ones([1,40]),np.expand_dims(np.linspace(0.25,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose(),
np.concatenate((np.zeros([1,40]),np.expand_dims(np.linspace(0,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose()),axis=1)
    return ListedColormap(seis)

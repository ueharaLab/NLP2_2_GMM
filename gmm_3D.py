import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import pickle
from scipy.stats import multivariate_normal

with open('parameters.pickle', mode='rb') as f:
    parameters=pickle.load(f) 
gamma=parameters[0]
pi=parameters[1]
mean=parameters[2]
cov=parameters[3]

#関数に投入するデータを作成
x = np.arange(-1.5,2.5,0.1)
y = np.arange(-2.5,2,0.1)
X, Y = np.meshgrid(x, y)
#print(X)
#print(Y)

z = np.c_[X.ravel(),Y.ravel()]
#print(z)
#print(z.shape)

'''
#二次元正規分布の確率密度を返す関数
def gaussian(x,mu,sigma):
    #分散共分散行列の行列式
    det = np.linalg.det(sigma)
    print(det)
    #分散共分散行列の逆行列
    inv = np.linalg.inv(sigma)   
    n=len(x)
    print(inv)
    print(np.dot(np.dot((x - mu),inv),(x - mu).T))
    return np.exp(-np.diag(np.dot(np.dot((x - mu),inv),(x - mu).T)/2.0)) / (np.sqrt((2 * np.pi) ** n * det))
'''
def gaussian(x,mu,sigma):
    return multivariate_normal(mu,sigma).pdf(x)
    
def gmm_pdf(x):
    K = len(pi)
    pdf = 0.0
    for j in range(K):
        pdf += pi[j] * gaussian(x, mean[j], cov[j])
    return pdf


#2変数の平均値を指定
#mu = np.array([0,0])
#2変数の分散共分散行列を指定
#sigma = np.array([[1,0.],[0.,1]])

Z = gmm_pdf(z)
#print(Z.shape)
shape = X.shape
Z = Z.reshape(shape)

#二次元正規分布をplot
fig = plt.figure(figsize = (50, 50))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
plt.show()

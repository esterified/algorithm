
"""
Created on Sat Jun 17 14:07:09 2017

@author: ESTERIFIED
"""

from scipy import optimize,meshgrid
import numpy as np
import matplotlib.cm as com
import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.pyplot import plot,grid,scatter,xlabel,ylabel,figure,subplot,clabel,contour
from numpy.linalg import norm
from numpy import linspace
from mpl_toolkits.mplot3d import Axes3D 

#Regularization Parameter:

# X = (hours sleeping, hours studying), y = Score on test

X = np.array(([5,1], [5,1], [10,2], [6,1.5]), dtype=float)
y = np.array(([75], [92], [93], [70]), dtype=float)
# Normalize
X = X/np.amax(X,axis=0)
y = y/100 #Max test score is 100

class neural(object):
        
    def __init__(self):
        self.Lambda = 0.0001
        self.l1=2
        self.l2=4
        self.l3=1
        self.w1=np.random.randn(self.l1,self.l2)
        self.w2=np.random.randn(self.l2,self.l3)
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def forward(self, x):
        self.z2 = np.dot(x, self.w1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        yhat = self.sigmoid(self.z3) 
        return yhat
      
    def sigmoidprime(self,x):
        #Gradient of sigmoid
        return np.exp(-x)/((1+np.exp(-x))**2)
    #Need to make changes to costFunction and costFunctionPrim:

    def costfunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.w1**2)+np.sum(self.w2**2))
        return J

    def costfunctionprime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yhat = self.forward(X)
        delta3 = np.multiply(-(y-self.yhat), self.sigmoidprime(self.z3))
        #Add gradient of regularization term:
        dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.w2
        delta2 = np.dot(delta3, self.w2.T)*self.sigmoidprime(self.z2)
        #Add gradient of regularization term:
        dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.w1
        return dJdW1, dJdW2    
#    def costfunction(self,x,y):
#        self.yhat=self.forward(x)
#        return 0.5*sum((y-self.yhat)**2)
#    def costfunctionprime(self,x,y):
#        self.yhat = self.forward(x)
#        delta3 = np.multiply(-(y-self.yhat), self.sigmoidprime(self.z3))
#        dJdW2 = np.dot(self.a2.T, delta3)
#        delta2 = np.dot(delta3, self.w2.T)*self.sigmoidprime(self.z2)
#        dJdW1 = np.dot(x.T, delta2)  
#        return dJdW1, dJdW2
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.w1.ravel(), self.w2.ravel()))
        return params 
    def setParams(self,params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.l2 * self.l1
        self.w1 = np.reshape(params[W1_start:W1_end], (self.l1 , self.l2))
        W2_end = W1_end + self.l2*self.l3
        self.w2 = np.reshape(params[W1_end:W2_end], (self.l2, self.l3))
    def computegradients(self, X, y):
        dJdW1, dJdW2 = self.costfunctionprime(X, y)
        return np.concatenate((dJdW1.ravel(), \
        dJdW2.ravel()))
    def computeNumericalGradient(self, X, y): #if needed
        paramsInitial = self.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            self.setParams(paramsInitial + perturb)
            loss2 = self.costfunction(X, y)
            
            self.setParams(paramsInitial - perturb)
            loss1 = self.costfunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        self.setParams(paramsInitial)

        return numgrad 
        
class Train_n(object):
    def __init__(self,g):
        self.N=g
    def callbackF(self, params):
        self.N.setParams(params)
        #storing costfunction corresponding to iteration just , \
        #for the ease of plotting in to the variable j 
        self.J.append(self.N.costfunction(self.X, self.y))   
        #self.testJ.append(self.N.costfunction(self.testX, self.testy)) 
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costfunction(X, y)
        grad = self.N.computegradients(X,y)
        
        return cost,grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        self.testJ=[]
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X,y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
    
        
n=neural()
c=n.computegradients(X, y)
d=n.computeNumericalGradient(X, y)
#print a,'\n',b 
#
testi=norm(c-d)/norm(c+d)

g=Train_n(n)
g.train(X,y)

#Plot after BFGS iteration completion
fig = figure('cost vs iterations')
plot(g.J)
grid(1)
xlabel('Iterations')
ylabel('Cost')
#Plot projections of our new data:
fig = figure('projections')
subplot(1,2,1)
scatter(X[:,0], y)
grid(1)
xlabel('Hours Sleeping')
ylabel('Test Score')

subplot(1,2,2)
scatter(X[:,1], y)
grid(1)
xlabel('Hours Studying')
ylabel('Test Score')


#Test network for various combinations of sleep/study:
hoursSleep = linspace(0, 10, 10)
hoursStudy = linspace(0, 5, 10)

#Normalize data (same way training data way normalized)
hoursSleepNorm = hoursSleep/10.
hoursStudyNorm = hoursStudy/5.

#Create 2-d versions of input for plotting
a, b  = meshgrid(hoursSleepNorm, hoursStudyNorm)

#Join into a single input matrix:
allInputs = np.zeros((a.size, 2))
allInputs[:, 0] = a.ravel()
allInputs[:, 1] = b.ravel()
allOutputs = n.forward(allInputs)
#Contour Plot:
yy = np.dot(hoursStudy.reshape(10,1), np.ones((1,10)))
xx = np.dot(hoursSleep.reshape(10,1), np.ones((1,10))).T
figure('contour')
CS = contour(xx,yy,100*allOutputs.reshape(10, 10))
clabel(CS, inline=1, fontsize=10)
xlabel('Hours Sleep')
ylabel('Hours Study')

fig = figure()

ax=fig.gca(projection='3d')

ax.scatter(10*X[:,0], 5*X[:,1], 100*y, c='k', alpha = 1, s=100)
ax.grid(1)

surf = ax.plot_surface(xx, yy, 100*allOutputs.reshape(10, 10),cmap=com.jet, alpha = 0.7,rstride=1,cstride=1)

ax.set_xlabel('Hours Sleep')
ax.set_ylabel('Hours Study')
ax.set_zlabel('Test Score')
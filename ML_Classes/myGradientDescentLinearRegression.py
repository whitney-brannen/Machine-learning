import numpy as np
import pandas as pd
from scipy.stats import f
import random as r
import matplotlib.pyplot as plt

class myGradientDescentLinearRegression:
    
    jStop = 0.00001
    alpha = 0.05
    maxIterations = 10000
    
    def __init__(self, data):
        
        # initialize
        self.data = data
        self.x = self.data.iloc[:,0]
        self.y = self.data.iloc[:,1]
        self.n = len(self.x)
        self.dfn = None
        self.dfd = None
        # linear equation values
        self.theta0 = None
        self.theta1 = None
        self.deltaJ = None
        self.ybar = self.y.mean()
        # stats
        self.Rsquared = None
        self.Fstat = None
        self.Pvalue = None
        # sum of squares and mean squares
        self.ssTot = None
        self.ssReg = None
        self.ssErr = None
        self.msTot = None
        self.msReg = None
        self.msErr = None
    
    def fit(self):
        self.calc_df()
        self.gradient_descent()
        self.calc_ssTot()
        self.calc_ssErr()
        self.calc_ssReg()
        self.calc_msTot()
        self.calc_msReg()
        self.calc_msErr()
        return print(f"Data read: n = {self.n}")
        
    def transform(self):
        self.calc_Fstat()
        self.calc_Rsquared()
        self.calc_Pvalue()
        return print(f"y={round(self.theta0,3)} + {round(self.theta1,3)}x\nIntercept θ0: {round(self.theta0,3)}\nCoefficient θ1: {round(self.theta1,3)}\nR2: {round(self.Rsquared,3)}\nF statistic: {round(self.Fstat,3)}\nP value: {self.Pvalue}")
    
    def fit_transform(self):
        self.fit()
        self.trasnform()
        
    def calc_df(self):
        observations,groups = self.data.shape
        self.dfn = groups - 1
        self.dfd = (observations) - groups
        return self.dfn,self.dfd
    
    def gradient_descent(self):
        self.theta1 = 0
        self.theta0 = 0
        self.calc_deltaJ()
        iterations = 0
        while self.deltaJ > myGradientDescentLinearRegression.jStop:
            self.calc_newTheta0()
            self.calc_newTheta1()
            self.calc_deltaJ()
            iterations += 1
            if iterations == myGradientDescentLinearRegression.maxIterations:
                break
            
    def calc_deltaJ(self):
        m = self.n
        self.calc_ssErr()
        ri2 = self.ssErr
        self.deltaJ = (1/(2*m)) * ri2 
    
    def calc_newTheta0(self):
        sumOf = 0
        for i in range(self.n):
            sumOf += (self.theta0 + self.theta1*self.x[i] - self.y[i])
        partialDerivativeTheta0 = ( 1/(self.n) ) * sumOf
        newTheta0 = self.theta0 - myGradientDescentLinearRegression.alpha*(partialDerivativeTheta0)
        self.theta0 = newTheta0
    
    
    def calc_newTheta1(self):
        sumOf = 0
        for i in range(self.n):
            sumOf += (self.theta0 + self.theta1*self.x[i] - self.y[i])*self.x[i]
        partialDerivativeTheta1 = ( 1/(self.n) ) * sumOf
        newTheta1 = self.theta1 - myGradientDescentLinearRegression.alpha*(partialDerivativeTheta1)
        self.theta1 = newTheta1
    
    def calc_yhat(self,xval):
        return self.theta0 + (self.theta1 * xval)
      
    def calc_ssTot(self):
        ssTot = 0
        for i in self.y:
            ssTot += (i - self.ybar)**2
        self.ssTot = ssTot
        
    def calc_ssReg(self):
        ssReg = 0
        for i in self.x:
            ssReg += (self.calc_yhat(i) - self.ybar)**2
        self.ssReg = ssReg
    
    def calc_ssErr(self):
        residualSquared = 0
        for i in range(len(self.y)):
            yi = self.y[i]
            yihat = self.calc_yhat(self.x[i])
            residualSquared += (yi - yihat)**2
        self.ssErr = residualSquared
    
    def calc_msTot(self):
        self.msTot = self.ssTot / (self.n - 1)
    
    def calc_msReg(self):
        self.msReg = self.ssReg / 1
    
    def calc_msErr(self):
        self.msErr = self.ssErr / (self.n - 2)
    
    def calc_Fstat(self):
        self.Fstat = self.msReg / self.msErr
    
    def calc_Rsquared(self):
        self.Rsquared = self.ssReg / self.ssTot
    
    def calc_Pvalue(self):
        self.Pvalue = f.sf(self.Fstat, self.dfn, self.dfd)
import numpy as np
import pandas as pd
from scipy.stats import f
import random as r
import matplotlib.pyplot as plt

class myAnalyticalLinearRegression:
    
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
        self.calc_theta1()
        self.calc_theta0()
        self.calc_ssTot()
        self.calc_ssReg()
        self.calc_ssErr()
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
    
    def calc_theta1(self):
        cov = self.x.cov(self.y)
        varx = self.x.var() 
        self.theta1 = cov/varx
        
    def calc_theta0(self):
        xbar = self.x.mean()
        slope = xbar * self.theta1
        self.theta0 = self.ybar - slope
        
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
        return self.msTot
    
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
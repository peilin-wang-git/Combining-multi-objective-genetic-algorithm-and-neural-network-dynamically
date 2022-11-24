import numpy as np


# deltaY=Y2-Y1=dYFreq-0.3*deltaR_Q+0.15*deltaZL+0.05*deltaV>1e-7
# Y=YF-0.3*R_Q+0.15ZL-0.05V YMIN
# dYFreq/dx=0 in [FREQ-toleF,FREQ+toleF],-1e11 in (,FREQ-toleF],+1e11 in [FREQ+tolef,)
# #1 dy/dxi=(dy(xi+step)-dy(xi-step))/2step
# #2 dy/dxi=(dy(xi+step)-dy(xi))/step
# x_(i+1)=x_i-alpha*grad(y)
class yfunc(object):
    def __init__(self,yClass):
        super().__init__()
        self.exact=yClass()
    def gaussian(self,x,mu,sigma):
        f_x = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-np.power(x-mu, 2.)/(2*np.power(sigma,2.)))
        return(f_x)
    
    def Y(self,parx,*coffs):
        #print("yfunc",coffs)
        #print(len(coffs))
        return self.exact.Y(parx,*coffs)
    def pGrad(self,parx,*coffs):
        return self.exact.pGrad(parx,*coffs)

class myYFunc00(object):
    def __init__(self):
        super().__init__()
    def Y(self,x,*coffs):
        ndims=len(x)
        sum=0
        for i in range(ndims):
            sum+=x[i]**2
        return sum
    def pGrad(self,x,*coffs):
        ndims=len(x)
        pgrad = np.zeros(ndims) 
        for i in range(ndims):
            pgrad[i]=2*x[i]  
        return pgrad
class myYFunc01(object):
    def __init__(self):
        super().__init__()
        self.xdim=4
        self.step=0.01
        self.demandF = 500
        self.tolef = 0.0
        self.g1 = 0.15
        self.defnor=[1,1,1,1,1]
    def Y(self,x,*coffs):
        if len(coffs)==0:
            nor=self.defnor
        else:
            nor=coffs[0]
        b=-0.3*x[1]/nor[1]-0.15*x[2]/nor[2]-0.05*x[4]/nor[4]
        a0=((x[0]-self.demandF)**2)*self.g1    
        return b+a0
    def pGrad(self,x0,*coffs):
        if len(coffs)==0:
            nor=self.defnor
        else:
            nor=coffs[0]
        ndims=len(x0)
        pgrad = np.zeros(ndims)        
        for j in range(ndims):
            xi=x0
            xi[j] = x0[j] + self.step
            pgrad[j] = (self.Y(xi, nor)-self.Y(x0,nor)) / self.step
            #print("pgrad_norm[", j, "]", pgrad[j], file=self.log, flush=True)
        return pgrad

class myYFunc02(object): #去除了Freq相关函数
    def __init__(self):
        super().__init__()
        self.xdim=3
        self.step=0.01
        self.tolef = 0.0
        self.defnor=[1,1,1,1,1]
    def Y(self,x,*coffs):
        #print("exact",coffs)
        if len(coffs)==0:
            nor=self.defnor
        else:
            nor=coffs[0]
        b=-0.6*x[1]/nor[1]-0.3*x[2]/nor[2]-0.1*x[4]/nor[4]
        return b
    def pGrad(self,x0,*coffs):
        
        
        if len(coffs)==0:
            nor=self.defnor
        else:
            nor=coffs[0]
        ndims=len(x0)
        pgrad = np.zeros(ndims)        
        for j in range(ndims):
            xi=x0
            xi[j] = x0[j] + self.step
            pgrad[j] = (self.Y(xi, nor)-self.Y(x0,nor)) / self.step
            #print("pgrad_norm[", j, "]", pgrad[j], file=self.log, flush=True)
        return pgrad

class myYFunc03(object): #A*(500-Freq)**2
    def __init__(self):
        super().__init__()
        self.xdim=3
        self.step=0.01
        self.demandF = 500
        self.tolef = 0.0
        self.g1 = 1
        self.defnor=[1,1,1,1,1]
    def Y(self,x,*coffs):
        
        if len(coffs)==0:
            nor=self.defnor
        else:
            nor=coffs[0]
        a0=((x[0]-self.demandF)**2)*self.g1  
        return a0
    def pGrad(self,x0,*coffs):
        if len(coffs)==0:
            nor=self.defnor
        else:
            nor=coffs[0]
        ndims=len(x0)
        pgrad = np.zeros(ndims)        
        for j in range(ndims):
            xi=x0
            xi[j] = x0[j] + self.step
            pgrad[j] = (self.Y(xi, nor)-self.Y(x0,nor)) / self.step
            #print("pgrad_norm[", j, "]", pgrad[j], file=self.log, flush=True)
        return pgrad
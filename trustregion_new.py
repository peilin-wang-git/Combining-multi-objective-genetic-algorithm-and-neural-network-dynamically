import os
import sys
import shutil
import re
import result
import copy
import numpy as np
import worker
import yfunction


##V2 rewrite
class trustregion(object):
    def __init__(self, worker, x0):
        super().__init__()
        # deltaY=Y2-Y1=dYFreq-0.3*deltaR_Q+0.15*deltaZL+0.05*deltaV>1e-7
        # Y=YF-0.3*R_Q+0.15ZL-0.05V YMIN
        # dYFreq/dx=0 in [FREQ-toleF,FREQ+toleF],-1e11 in (,FREQ-toleF],+1e11 in [FREQ+tolef,)
        # #1 dy/dxi=(dy(xi+step)-dy(xi-step))/2step
        # #2 dy/dxi=(dy(xi+step)-dy(xi))/step
        # x_(i+1)=x_i-alpha*grad(y)
        
        ###
        self.tolerance = 1e-6
        self.maxcount = 50
        self.alpha = 0.001  # learning rate
        self.step = 0.01

        ##y函数
        self.yFunc=yfunction.yfunc(yfunction.myYFunc01)

        ##归一化X
        self.xrangep=0.1  #X取值的最大宽度
        self.xnorm0=0
        self.realx0=x0
        ###OTHERS
        self.w = worker
        self.results=result.result
        logpath=os.path.join(self.w.getResultDir(),"result.log")
        self.log = open(logpath, "w",)
        self.runcount=0

        ##########INITIALIZED############       

    def normx2realx(self,normx):
        rx=np.zeros_like(normx)
        for i in range(len(rx)):
            rx[i]=self.realx0*(1+(normx[i]-0.5)*self.xrangep)
        return rx
    
    def realx2normx(self,realx):
        nx=np.zeros_like(realx)
        for i in range(len(nx)):
            nx[i]=(realx-self.realx0)/(realx*self.xrangep)+0.5
            pass

    def genRandomStartPs(self,N,dims,ubounds,lbounds):
        stps=np.zeros(self.startpoints)
        res=np.zeros((N,dims))
        tmp=np.zeros(N)        
        np.random.seed()
        for i in range(dims):
            d=(ubounds[i]-lbounds[i])/N
            for j in range(N):
                tmp[j]=np.random.random()*d+d*j+lbounds[i]
            np.random.shuffle(tmp)
            for j in range(N):
                res[j][i]=tmp[j]
        return res



    def Y(self, pam,normal):
        b=-0.3*pam[1]/normal[1]-0.15*pam[2]/normal[2]-0.05*pam[4]/normal[4]
        b1=pam[1]/normal[1]
        b2=pam[2]/normal[2]
        b3=pam[4]/normal[4]
        a0=((pam[0]-self.demandF)**2)*self.g1    
        return b+a0

    def pGradFromR0_norm(self, x0, step, nor):
        ndims=len(x0)
        pgrad = np.zeros(ndims)        
        for j in range(ndims):
            xi=x0.copy()
            xi[j] = x0[j] + step
            rg=self.w.runWithx(xi,"dx"+str(j))
            pgrad[j] = (self.yFunc.Y(rg, nor)-self.yFunc.Y(r0,nor)) / step
            print("pgrad_norm[", j, "]", pgrad[j], file=self.log, flush=True)
        return pgrad

    def computeGradBWFile(self, path0, path1):
        r0 = result.readModeResult(path0, 1)
        r1 = result.readModeResult(path1, 1)
        grad = self.deltaY(r1, r0) / self.step
        print(grad)
        print(r0, r1)
        return grad

    def gradient(self,x0,r0,nor):
        ndims=len(x0)
        pgrad = np.zeros(ndims)        
        for j in range(ndims):
            xi=x0.copy()
            xi[j] = x0[j] + step
            rg=self.w.runWithx(xi,"dx"+str(j))
            pgrad[j] = (self.yFunc.Y(rg, nor)-self.yFunc.Y(r0,nor)) / step
        return grad
    
    def do(self):
        self.runcount=0
        #print("run", i, file=self.log, flush=True)
        #得到x0
        x0=self.x0
        print("x0=", x0, file=self.log, flush=True)
        r0=self.w.runWithx(x0,str(runcount))
        roo=r0
        nor=r0
        print("r0=", r0, file=self.log, flush=True)
        x1=np.zeros(len(x0))
        v=np.zeros(len(x0))
        s=np.zeros(len(x0))
        eps=np.zeros(len(x0))
        for i in range(len(x0)):
            eps[i]=self.epsilon
        dx=np.zeros(len(x0))
        success = False
        while self.runcount <= self.maxcount:
            print("run", runcount, file=self.log, flush=True)

            print("pgrad_norm=", pgrad, file=self.log, flush=True)

            #TRUST REGION FRAMEWORK
            #DELTAK=(UBOUND-LBOUND)*SIGMA/2
            #F(X)=Y(R(X))
            #M(XK+p)=F(XK)+gTp
            #gT=F(XK+DX)-F(XK)/DX
            #rK=(F(XK)-F(XK+pK))/(M(XK)-M(XK+pK))
            #IF rK<0 X
            #IF rk<1/4


            print("x0=", x0, file=self.log, flush=True)
            print("x1=", x1, file=self.log, flush=True)
            r1=self.w.runWithx(x1,str(i))
            print("r1=", r1, file=self.log, flush=True)
            dy = self.yFunc.Y(r1,nor)-self.yFunc.Y(r0,nor)
            print("dy=", dy, file=self.log, flush=True)
            y=self.yFunc.Y(r1,nor)
            print("y=", y, file=self.log, flush=True)
            x0 = x1
            r0 = r1
            if abs(dy) < self.tolerance:
                success = True
                break
            self.runcount += 1
        if success == True:
            print(
                "Succeed", "x=", x0, "r0=", roo, "rfin=", r1, file=self.log, flush=True
            )
        else:
            print(
                "reached maxAttemptCount",
                "x0=",
                x0,
                "r0=",
                roo,
                "rfin=",
                r1,
                file=self.log,
                flush=True,
            )
        return self.runcount


    def start(self):
        self.do()



import os
import sys
import shutil
import re
import result
import copy
import numpy as np
import cstmanager
import yfunction
import projectutil

class myAlg02_POP(object):
    def __init__(self, manager, params, log_obj):
        super().__init__()
        self.w = manager
        self.results=result.result
        self.log=log_obj
        self.runcount=0
        self.params=params
    def do(self):
        print(self.params)
        for i in range(5):
            xi=projectutil.convert_json_params_to_list(self.params)
            if (not self.params[i]["description"].find("常数")>=0):
                xi[i]+=1
            self.log.logger.info("changed paramname %s, paramvalue %s"% (self.params[i]['name'],self.params[i]['value']))
            
            self.w.addTask(xi,"dx"+str(i))  
        self.w.start()
        self.w.synchronize()
        rg=self.w.getFullResults()
        ###SORT RESULTS
        rg=sorted(rg,key=lambda x: x['name'])
        #############
        print(rg)
    def start(self):
        self.do()

##V2 rewrite for parallel
class myAlg01_pillbox(object):
    def __init__(self, manager, x0):
        super().__init__()
        # deltaY=Y2-Y1=dYFreq-0.3*deltaR_Q+0.15*deltaZL+0.05*deltaV>1e-7
        # Y=YF-0.3*R_Q+0.15ZL-0.05V YMIN
        # dYFreq/dx=0 in [FREQ-toleF,FREQ+toleF],-1e11 in (,FREQ-toleF],+1e11 in [FREQ+tolef,)
        # #1 dy/dxi=(dy(xi+step)-dy(xi-step))/2step
        # #2 dy/dxi=(dy(xi+step)-dy(xi))/step
        # x_(i+1)=x_i-alpha*grad(y)
        self.x0=x0
        ###
        self.tolerance = 1e-6
        self.maxcount = 50
        self.alpha = 10  # learning rate
        self.step = 0.01
        
        ##self.startpoints=5 #额外随机初值
        ##self.startvarient=0.001 #不要取太大 表示在初值上下浮动0.1%
        ##y函数 YFunc02 #去除了Freq相关函数的YFUNC01
        self.yFunc=yfunction.yfunc(yfunction.myYFunc02)
        #y函数 YFunc03 用于锁频
        self.yFunc2=yfunction.yfunc(yfunction.myYFunc03)
        ###OTHERS
        self.w = manager
        self.results=result.result
        logpath=os.path.join(self.w.getResultDir(),"result.log")
        self.log = open(logpath, "w")
        self.runcount=0

        ##########INITIALIZED############       





    def pGradFromR0(self, x0, r0, step):
        pgrad = np.zeros(len(indc))
        
        for j in range(len(indc)):
            ndims=len(x0)
            pgrad = np.zeros(ndims)        
            for j in range(ndims):
                xi=x0
                xi[j] = x0[j] + step
                rg=self.w.runWithx(xi,"dx"+str(j))
                pgrad[j] = (self.yFunc.Y(rg)-self.yFunc.Y(r0)) / step
                print("pgrad[", j, "]", pgrad[j], file=self.log, flush=True)
        return pgrad

    def pGradFromR0_norm(self, x0,r0, step, nor):
        ndims=len(x0)
        pgrad = np.zeros(ndims)        
        for j in range(ndims):
            xi=x0.copy()
            xi[j] = x0[j] + step
            rg=self.w.runWithx(xi,"dx"+str(j))
            pgrad[j] = (self.yFunc.Y(rg, nor)-self.yFunc.Y(r0,nor)) / step
            print("pgrad_norm[", j, "]", pgrad[j], file=self.log, flush=True)
        return pgrad


    #####
    def pGradFromXYR0_norm(self, x0,yFunc,r0, step, nor):
        ndims=len(x0)
        pgrad = np.zeros(ndims)        
        for j in range(ndims):
            xi=x0.copy()
            xi[j] = x0[j] + step
            rg=self.w.runWithx(xi,"dx"+str(j))
            pgrad[j] = (yFunc.Y(rg, nor)-yFunc.Y(r0,nor)) / step
        return pgrad
    
    def pGradFromXYR0_norm_parallel(self, x0,yFunc,r0, step, nor):
        ndims=len(x0)
        pgrad = np.zeros(ndims)        
        for j in range(ndims):
            xi=x0.copy()
            xi[j] = x0[j] + step

            self.w.addTask(xi,"dx"+str(j))  
        self.w.start()
        self.w.synchronize()#同步 很重要
        rg=self.w.getFullResults()
        ###SORT RESULTS
        rg=sorted(rg,key=lambda x: x['name'])
        #############
        mresult=[]
        for irg in rg:
            pgrad[j] = (yFunc.Y(irg['value'], nor)-yFunc.Y(r0,nor)) / step
        return pgrad

    def pGradiFromXiYR0_norm(self,x0,i,yFunc,r0,step,nor):

        xi=x0.copy()
        xi[i] = x0[i] + step
        rg=self.w.runWithx(xi,"dx"+str(i))
        pgradi = (yFunc.Y(rg, nor)-yFunc.Y(r0,nor)) / step
        return pgradi
    def computeGradBWFile(self, path0, path1):
        r0 = result.readModeResult(path0, 1)
        r1 = result.readModeResult(path1, 1)
        grad = self.deltaY(r1, r0) / self.step
        print(grad)
        print(r0, r1)
        return grad

    def findMainFactorForFreq(self,x0,r0,step):
        ndims=len(x0)
        rgs=[]
        #find Freq INDEX
        findex=0
        for j in range(ndims):
            xi=x0.copy()
            xi[j] = x0[j] + step
            pg=(self.w.runWithx(xi,"dx"+str(j))[findex]-r0[findex])/step
            rgs.append(pg) #rgs[i]=p(Freq)/p(xi)
        max=0
        maxj=0
        for j in range(len(x0)):
            if abs(rgs[j])>max:
                max=abs(rgs[j])
                maxj=j
        return maxj

    def do(self):
        ###Y=Y(R(X))####
        self.runcount=0
        #print("run", i, file=self.log, flush=True)
        #得到x0
        x0=self.x0
        print("x0=", x0, file=self.log, flush=True)
        r0=self.w.runWithx(x0,str(self.runcount))
        roo=r0
        nor=r0
        print("r0=", r0, file=self.log, flush=True)
        x1=np.zeros(len(x0))
        dx=np.zeros(len(x0))
        success = False
        #mainFactor=x?
        #subFactor= 
        #寻找决定频率参数
        #dindex=self.findMainFactorForFreq(x0,r0,self.step)
        #期望返回1:(R)
        #已知dindex=1提高效率
        dindex=1
        print("dindex=", dindex, file=self.log, flush=True)
        # x0=[ 46.9556537  230.50712379 259.98077285 397.7191117 ]
        #x0[1]=230.50712379
        #先锁定频率
        freqLock=True
        fcount=0
        fcountmax=20
        unitxd=np.zeros(len(x0))
        unitxd[dindex]=1
        if freqLock!=True:
            while fcount<=fcountmax:
                print("processFreq")
                print("frun", fcount, file=self.log, flush=True)
                pgrad=self.pGradiFromXiYR0_norm(x0,dindex,self.yFunc2,r0,self.step,nor)
                print("x0=", x0, file=self.log, flush=True)
                dfreq=self.yFunc2.Y(r0,nor)
                x1=x0-dfreq/pgrad*unitxd
                r1=self.w.runWithx(x1,"Freq"+str(fcount))
                print("r1=", r1, file=self.log, flush=True)
                dy=self.yFunc2.Y(r1,nor)-self.yFunc2.Y(r0,nor)
                r0=r1
                x0=x1
                if abs(dy)/500 < self.tolerance:
                    success = True
                    freqLock=True
                    break            
                fcount+=1
               
        print("run", self.runcount, file=self.log, flush=True)
        success=False
        while self.runcount <= self.maxcount:
            print("run", self.runcount, file=self.log, flush=True)
            

            pgrad=self.pGradFromXYR0_norm_parallel(x0,self.yFunc2,r0,self.step,nor)
            print("pgrad_norm=", pgrad, file=self.log, flush=True)
            x1=x0-pgrad*self.alpha
            r1=self.w.runWithx(x1,str(self.runcount))
            
            dfreq=self.yFunc2.Y(r1,nor)
            #dfreq=self.yFunc2.Y(r1,nor)-0
            pgrad2=self.pGradiFromXiYR0_norm(x1,dindex,self.yFunc2,r1,self.step,nor)
            x1=x1-dfreq/pgrad2*unitxd #往回修正x1 锁频率

            print("x0=", x0, file=self.log, flush=True)
            print("x1=", x1, file=self.log, flush=True)
            r1=self.w.runWithx(x1,str(self.runcount))
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



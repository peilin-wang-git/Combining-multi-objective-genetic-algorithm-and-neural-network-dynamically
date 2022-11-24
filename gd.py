import os
import sys
import shutil
import re
import result
import copy
import numpy as np
import cstworker
def gaussian(x,mu,sigma):
    f_x = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-np.power(x-mu, 2.)/(2*np.power(sigma,2.)))
    return(f_x)

##V2 with multiple start point
class gredientDecend(object):
    def __init__(self, worker, conf0):
        super().__init__()
        # deltaY=Y2-Y1=dYFreq-0.3*deltaR_Q+0.15*deltaZL+0.05*deltaV>1e-7
        # Y=YF-0.3*R_Q+0.15ZL-0.05V YMIN
        # dYFreq/dx=0 in [FREQ-toleF,FREQ+toleF],-1e11 in (,FREQ-toleF],+1e11 in [FREQ+tolef,)
        # #1 dy/dxi=(dy(xi+step)-dy(xi-step))/2step
        # #2 dy/dxi=(dy(xi+step)-dy(xi))/step
        # x_(i+1)=x_i-alpha*grad(y)
        self.tolerance = 1e-6
        self.demandF = 500
        self.tolef = 0.0
        self.g1 = 0.15
        self.maxcount = 50
        self.alpha = 0.1  # learning rate
        self.step = 0.01


        self.startpoints=5 #额外随机初值
        self.startvarient=0.001 #不要取太大 表示在初值上下浮动0.1%
        ###OTHERS
        self.w = worker
        self.conf0 = conf0
        logpath=os.path.join(self.w.resultDir,"result.log")
        self.log = open(logpath, "w")

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
    def editSP(self, inE, newX, indc):
        outE = inE
        for i in range(len(indc)):
            outE[indc[i]]["value"] = newX[i]
        return outE

    def deltaY(self, pam0, pam1):
        # (freq0,R_Q0,zl0,Q0,vot0,tl0)
        # deltaY=Y2-Y1=dYFreq-0.3*deltaR_Q+0.15*deltaZL+0.05*deltaV>1e-7
        b = (
            -0.3 * (pam1[1] - pam0[1])
            - 0.15 * (pam1[2] - pam0[2])
            - 0.05 * (pam1[4] - pam0[4])
        )
        a1 = (pam1[0] - self.demandF)**2*self.g1
        a0 = (pam0[0] - self.demandF)**2*self.g1

        return b + a1 - a0

    def deltaY_Norm(self, pam0, pam1, nor):
        # (freq0,R_Q0,zl0,Q0,vot0,tl0)
        # deltaY=Y2-Y1=dYFreq-0.3*deltaR_Q+0.15*deltaZL+0.05*deltaV>1e-7
        b = (
            - 0.3 * (pam1[1] - pam0[1]) / nor[1]
            - 0.15 * (pam1[2] - pam0[2]) / nor[2]
            - 0.05 * (pam1[4] - pam0[4]) / nor[4]
        )

        a1 = (pam1[0] - self.demandF)**2*self.g1
        a0 = (pam0[0] - self.demandF)**2*self.g1

        return b + a1 - a0

    def deltaY_Norm_3point(self,pam0, pam1,pam2,nor):
        # (freq0,R_Q0,zl0,Q0,vot0,tl0)
        # deltaY=Y2-Y1=dYFreq-0.3*deltaR_Q+0.15*deltaZL+0.05*deltaV>1e-7
        b = (
            - 0.3 * (4*pam1[1] - 3*pam0[1]-pam2[1]) / nor[1]
            - 0.15 * (4*pam1[2] - 3*pam0[2]-pam2[2]) / nor[2]
            - 0.05 * (4*pam1[4] - 3*pam0[4]-pam2[4]) / nor[4]
        )

        a1 = (pam1[0] - self.demandF)**2*self.g1
        a0 = (pam0[0] - self.demandF)**2*self.g1

        return b + a1 - a0

    def deltaY_Norm_5point(self,pam0, pam1,pam2,pam3,pam4,nor):
        # (freq0,R_Q0,zl0,Q0,vot0,tl0)
        # deltaY=Y2-Y1=dYFreq-0.3*deltaR_Q+0.15*deltaZL+0.05*deltaV>1e-7
        b = (
            - 0.3 * (48*pam1[1] - 25*pam0[1]-36*pam2[1]+16*pam3[1]-3*pam4[1]) / nor[1]
            - 0.15 * (48*pam1[2] - 25*pam0[2]-36*pam2[2]+16*pam3[2]-3*pam4[2]) / nor[2]
            - 0.05 * (48*pam1[4] - 25*pam0[4]-36*pam2[4]+16*pam3[4]-3*pam4[4]) / nor[4]
        )

        a1 = (pam1[0] - self.demandF)**2*self.g1
        a0 = (pam0[0] - self.demandF)**2*self.g1

        return b + a1 - a0

    def Y(self, pam,normal):
        b=-0.3*pam[1]/normal[1]-0.15*pam[2]/normal[2]-0.05*pam[4]/normal[4]
        b1=pam[1]/normal[1]
        b2=pam[2]/normal[2]
        b3=pam[4]/normal[4]
        a0=((pam[0]-self.demandF)**2)*self.g1    
        return b+a0

    def pGradFromR0(self, x0, r0, step, lse, indc,resultDir):
        pgrad = np.zeros(len(indc))
        
        for j in range(len(indc)):
            tl = copy.deepcopy(lse)
            tl[indc[j]]["value"] = x0[j] + step
            pa = self.w.createCSTBatch(tl, tl[indc[j]]["name"],resultDir)
            self.w.runcst(pa)
            p = os.path.join(resultDir, tl[indc[j]]["name"])
            rg = result.readModeResult(p, 1)

            pgrad[j] = self.deltaY(r0, rg) / step
            print("pgrad[", j, "]", pgrad[j], file=self.log, flush=True)
        return pgrad

    def pGradFromR0_norm(self, x0, r0, step, lse, indc, nor,resultDir):
        pgrad = np.zeros(len(indc))
        
        for j in range(len(indc)):
            tl = copy.deepcopy(lse)
            tl[indc[j]]["value"] = x0[j] + step
            pa = self.w.createCSTBatch(tl, tl[indc[j]]["name"],resultDir)
            self.w.runcst(pa)
            p = os.path.join(resultDir, tl[indc[j]]["name"])
            rg = result.readModeResult(p, 1)

            pgrad[j] = self.deltaY_Norm(r0, rg, nor) / step
            print("pgrad_norm[", j, "]", pgrad[j], file=self.log, flush=True)
        return pgrad

    def pGradFromR0_norm_3point(self, x0, r0, step, lse, indc, nor,resultDir):
        pgrad = np.zeros(len(indc))
        
        for j in range(len(indc)):
            tl = copy.deepcopy(lse)
            tl2=copy.deepcopy(lse)
            tl[indc[j]]["value"] = x0[j] + step
            tl2[indc[j]]["value"] = x0[j] + 2*step
            pa = self.w.createCSTBatch(tl, tl[indc[j]]["name"]+str(1),resultDir)
            pa2 = self.w.createCSTBatch(tl2, tl2[indc[j]]["name"]+str(2),resultDir)
            self.w.runcst(pa)
            self.w.runcst(pa2)
            p = os.path.join(resultDir, tl[indc[j]]["name"]+str(1))
            p2 = os.path.join(resultDir, tl2[indc[j]]["name"]+str(2))
            rg = result.readModeResult(p, 1)
            rg2= result.readModeResult(p2, 1)
            pgrad[j] = self.deltaY_Norm_3point(r0, rg,rg2, nor) / (step*2)
            print("pgrad_norm_3point[", j, "]", pgrad[j], file=self.log, flush=True)
        return pgrad
    def pGradFromR0_norm_5point(self, x0, r0, step, lse, indc, nor,resultDir):
        pgrad = np.zeros(len(indc))
        for j in range(len(indc)):
            tl = copy.deepcopy(lse)
            tl2=copy.deepcopy(lse)
            tl3=copy.deepcopy(lse)
            tl4=copy.deepcopy(lse)
            tl[indc[j]]["value"] = x0[j] + step
            tl2[indc[j]]["value"] = x0[j] + 2*step
            tl3[indc[j]]["value"] = x0[j] + 3*step
            tl4[indc[j]]["value"] = x0[j] + 4*step
            pa = self.w.createCSTBatchByName(tl, tl[indc[j]]["name"]+str(1),resultDir)
            pa2 = self.w.createCSTBatchByName(tl2, tl2[indc[j]]["name"]+str(2),resultDir)
            pa3 = self.w.createCSTBatchByName(tl3, tl3[indc[j]]["name"]+str(3),resultDir)
            pa4 = self.w.createCSTBatchByName(tl4, tl4[indc[j]]["name"]+str(4),resultDir)
            self.w.runcst(pa)
            self.w.runcst(pa2)
            self.w.runcst(pa3)
            self.w.runcst(pa4)
            p = os.path.join(resultDir, tl[indc[j]]["name"]+str(1))
            p2 = os.path.join(resultDir, tl2[indc[j]]["name"]+str(2))
            p3 = os.path.join(resultDir, tl3[indc[j]]["name"]+str(3))
            p4 = os.path.join(resultDir, tl4[indc[j]]["name"]+str(4))
            rg = result.readModeResult(p, 1)
            rg2= result.readModeResult(p2, 1)
            rg3= result.readModeResult(p3, 1)
            rg4= result.readModeResult(p4, 1)
            pgrad[j] = self.deltaY_Norm_5point(r0, rg,rg2,rg3,rg4,nor) / (step*4)
            #print("pgrad_norm_3point[", j, "]", pgrad[j], file=self.log, flush=True)
        return pgrad

    def computeGradBWFile(self, path0, path1):
        r0 = result.readModeResult(path0, 1)
        r1 = result.readModeResult(path1, 1)
        grad = self.deltaY(r1, r0) / self.step
        print(grad)
        print(r0, r1)
        return grad

    def do(self,resultDir,iconf,nordata):
        i=0
        print("run", i, file=self.log, flush=True)
        pa = self.w.createCSTBatch(iconf, str(i),resultDir)
        self.w.runcst(pa)
        p = os.path.join(resultDir, str(i))
        r0 = result.readModeResult(p, 1)
        r1 = r0
        if nordata[0]=="self":
            nor = r0  # 归一化系数
        elif nordata[0]=="ext":
            nor=nordata[1]
        print("r0=", r0, file=self.log, flush=True)
        rori = r0
        # PDS
        lse = iconf
        pt = []
        for j in range(len(lse)):
            if lse[j]["fixed"] == False:
                pt.append(j)
        x0 = np.zeros(len(pt))
        x1 = np.zeros(len(pt))
        for j in range(len(pt)):
            x0[j] = lse[pt[j]]["value"]
        i += 1
        print("x0=", x0, file=self.log, flush=True)
        success = False
        while i <= self.maxcount:
            print("count", i, file=self.log, flush=True)
            pgrad = self.pGradFromR0_norm_3point(x0, r0, self.step, lse, pt, nor,resultDir)
            print("pgrad_norm=", pgrad, file=self.log, flush=True)
            x1 = x0 - pgrad * self.alpha #梯度下降
            print("x0=", x0, file=self.log, flush=True)
            print("x1=", x1, file=self.log, flush=True)
            lse = self.editSP(lse, x1, pt)
            pa = self.w.createCSTBatch(lse, str(i),resultDir)
            self.w.runcst(pa)
            p = os.path.join(resultDir, str(i))
            r1 = result.readModeResult(p, 1)
            print("r1=", r1, file=self.log, flush=True)
            dy = self.deltaY_Norm(r0, r1,nor)
            print("dy=", dy, file=self.log, flush=True)
            y=self.Y(r1,nor)
            print("y=", y, file=self.log, flush=True)
            x0 = x1
            r0 = r1
            if abs(dy) < self.tolerance:
                success = True
                break
            i += 1
        if success == True:
            print(
                "Succeed", "x=", x0, "r0=", rori, "rfin=", r1, file=self.log, flush=True
            )
        else:
            print(
                "reached maxAttemptCount",
                "x0=",
                x0,
                "r0=",
                rori,
                "rfin=",
                r1,
                file=self.log,
                flush=True,
            )
        return i
    def start(self):
        # PDS
        lse = self.conf0
        pt = []
        for j in range(len(lse)):
            if lse[j]["fixed"] == False:
                pt.append(j)
        ubounds=[]
        lbounds=[]
        for j in range(len(pt)):
            lbounds.append(lse[pt[j]]['value']*(1-self.startvarient))
            ubounds.append(lse[pt[j]]['value']*(1+self.startvarient))
        stps=self.genRandomStartPs(self.startpoints,len(pt),ubounds=ubounds,lbounds=lbounds)
        x0 = np.zeros(len(pt))
        for j in range(len(pt)):
            x0[j] = lse[pt[j]]["value"]
        stps=np.insert(stps,0,x0,axis=0)#至少有一个初值点
        print("startpoints", file=self.log, flush=True)
        print(stps, file=self.log, flush=True)
        cs=[]
        ys=[]
        #产生归一化目标值数据
        nresultDir = os.path.join(self.w.resultDir,"norm")
        pa = self.w.createCSTBatch(lse,"0",nresultDir)
        self.w.runcst(pa)
        p = os.path.join(nresultDir, "0")
        nom=[]
        nom.append("ext") #nom[0]="ext" 外部提供y归一化data "self"y归一化data由第0次计算得到
        nom.append(result.readModeResult(p, 1)) #归一化y 保证计算结束比较不同初始点y函数是一致的
        #产生随机初始点并开始计算
        for i in range(len(stps)):
            iresultDir = os.path.join(self.w.resultDir,str(i))
            iconf0=self.editSP(lse,stps[i],pt)
            print("---------------------------------------------------------", file=self.log, flush=True)
            print("beginning with startpoint",i,"is",stps[i], file=self.log, flush=True)
            uc=self.do(iresultDir,iconf0,nom)
            cs.append(uc)
        print("---------------------------------------------------------", file=self.log, flush=True)
        ymax=0
        ix=0
        #比较最好的结果
        for i in range(len(stps)):
            iresultDir= os.path.join(self.w.resultDir,str(i),cs[i])
            ires=result.readModeResult(iresultDir,1)
            iy=self.Y(ires,nom[1])            
            if iy>ymax:
                ymax=iy
                ix=i

        print("Best result y is",ymax,"with startpoint",ix, file=self.log, flush=True)



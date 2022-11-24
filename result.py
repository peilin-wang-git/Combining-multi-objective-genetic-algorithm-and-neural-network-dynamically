import os
import sys
class result(object):
    def __init__(self):
        super().__init__()
        self.list=[]


    
def readModeResult(resultPath,modecount):
    #readFREQ
    filename='Mode'+str(modecount)+'Frequency.txt'
    fullpath=os.path.join(resultPath,filename)    
    with open(fullpath,'r') as f:
        lines=f.readlines()
        freq=float(lines[2].split()[1])
        f.close()

    #readR/Q
    filename='Mode'+str(modecount)+'R_Q.txt'
    fullpath=os.path.join(resultPath,filename)
    with open(fullpath,'r') as f:
        lines=f.readlines()
        R_Q=float(lines[2].split()[1])
        f.close()
    
    #readZL
    filename='Mode'+str(modecount)+'ShuntImpedance.txt'
    fullpath=os.path.join(resultPath,filename)
    with open(fullpath,'r') as f:
        lines=f.readlines()
        zl=float(lines[2].split()[1])
        f.close()

    #readQ
    filename='Mode'+str(modecount)+'Q-Factor.txt'
    fullpath=os.path.join(resultPath,filename)
    with open(fullpath,'r') as f:
        lines=f.readlines()
        Q=float(lines[2].split()[1])
        f.close()

    #readVotage
    filename='Mode'+str(modecount)+'Voltage.txt'
    fullpath=os.path.join(resultPath,filename)
    with open(fullpath,'r') as f:
        lines=f.readlines()
        vot=float(lines[2].split()[1])
        f.close()
    #readTotalLoss
    filename='Mode'+str(modecount)+'TotalLoss.txt'
    fullpath=os.path.join(resultPath,filename)
    with open(fullpath,'r') as f:
        lines=f.readlines()
        tl=float(lines[2].split()[1])
        f.close()

    return freq,R_Q,zl,Q,vot,tl

#print(readResult("C:\\Users\\ykk48\\Documents\\CST\\Result\\0",1))
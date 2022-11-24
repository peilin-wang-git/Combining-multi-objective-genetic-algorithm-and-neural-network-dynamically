import os,json,re,copy,sys,shutil
import result
import numpy as np
import time
import subprocess
import logging
import hashlib
import worker

class local_cstworker(worker.worker):
    def __init__(self,id,type,config,log_obj):
        super().__init__(id,type,config,log_obj)
        
        #configs
        #cstType,cstPatternDir,tempPath,taskFileDir,resultDir,cstPath
        self.cstType=config['ProjectType']
        self.cstPatternDir=config['cstPatternDir']
        self.tempPath=config['tempPath']
        self.taskFileDir=config['taskFileDir']
        self.resultDir=config['resultDir']
        self.cstProjPath=config['cstPath']
        
        self.paramList=copy.deepcopy(config['paramList'])
        self.taskIndex=0
        self.cstStatus='off'
        self.maxWaitTime=999999
        self.maxStopWaitTime=60
        self.cstProcess=None
        self.cstlog=None

        
        
        #LOGGING#

        self.log.logger.info("TempDir:"+self.tempPath)
        self.log.logger.info("TaskFileDir:"+self.taskFileDir)

        #FINDCST
        self.currentCSTENVPATH=config['CSTENVPATH']

        ##INIT##

        cstPatternName=self.cstType+".pattern"
        cstPatternPath=os.path.join(self.cstPatternDir,cstPatternName)
        if not os.path.exists(cstPatternPath):
            self.log.logger.error("pattern not found for "+self.cstType)
            os._exit(0)
        headerFilePath=os.path.join(self.cstPatternDir,"worker.vb")
        

        ##COPY cstProj TO TEMP
        if (os.path.exists(os.path.splitext(self.cstProjPath)[0])):
            self.log.logger.info("Found useless cstProject related files." )
            shutil.rmtree(os.path.splitext(self.cstProjPath)[0])
            self.log.logger.info(os.path.splitext(self.cstProjPath)[0])
            self.log.logger.info("files removed.")
        self.tmpCstFilePath=os.path.join(self.tempPath,"cstproj.cst")

            ##IF tmpCstFile EXISTS, CHECK FILE MD5
        if os.path.exists(self.tmpCstFilePath):
            md5srcfile=self.getFileMD5(self.cstProjPath)
            md5dstfile=self.getFileMD5(self.tmpCstFilePath)
            if not (md5srcfile==md5dstfile):
                self.log.logger.warning("CstProjectFile is not consistent with the one in temp.")
                self.clearall3()                
        else:
            shutil.copyfile(src=self.cstProjPath,dst=self.tmpCstFilePath)
        
        mainBatchFilePath=self.createMainCSTbatch(headerFilePath,cstPatternPath)
        self.startCSTenv(mainBatchFilePath)

    def genAvilTempPath(self,currTempDir):
        mid=0
        tmpdir=os.path.normpath(currTempDir)
        parentdir=os.path.dirname(tmpdir)
        name=os.path.basename(currTempDir)
        while (os.path.exists(tmpdir)):
            newname=name+"_"+str(mid).zfill(4)
            mid+=1
            tmpdir=os.path.join(parentdir,newname)
        os.mkdir(tmpdir)        
        return tmpdir
    def genAviltaskFileDir(self,currtaskFileDir):
        mid=0
        tmpdir=os.path.normpath(currtaskFileDir)
        parentdir=os.path.dirname(tmpdir)
        name=os.path.basename(currtaskFileDir)
        while (os.path.exists(tmpdir)):
            newname=name+"_"+str(mid).zfill(4)
            mid+=1
            tmpdir=os.path.join(parentdir,newname)
        os.mkdir(tmpdir)
        
        return tmpdir

    def getFileMD5(self,filePath):
        with open(filePath, "rb") as fp:
            md5obj = hashlib.md5()
            md5obj.update(fp.read())
            file_md5 = md5obj.hexdigest()
            return file_md5

    def clearall3(self):
        if (os.path.exists(self.taskFileDir)):
            shutil.rmtree(self.taskFileDir)
            os.mkdir(self.taskFileDir)
        if (os.path.exists(self.resultDir)):
            shutil.rmtree(self.resultDir)
            os.mkdir(self.resultDir)
        if (os.path.exists(self.tempPath)):
            shutil.rmtree(self.tempPath)
            os.mkdir(self.tempPath)
        self.log.logger.info("Worker_(%s):old runfile removed"% self.ID)



    def startCSTenv(self,mainbatchpath):
        cstlogPath=os.path.join(self.tempPath,"cst.log")
        self.cstlog=open(cstlogPath,"wb",buffering=0)
        command ="start cmd /k "+"\""+self.currentCSTENVPATH + "\"" +" -m "+ mainbatchpath
        command2 = "\""+self.currentCSTENVPATH + "\"" +" -m " + "\"" + mainbatchpath + "\""
        if self.cstStatus=='off':
            #command2="start cmd /k "+"\""+batFilePath+"\""
            self.log.logger.info(command2)
            self.cstProcess=subprocess.Popen(command2, stdout=self.cstlog, stderr=self.cstlog,shell=True)
            self.log.logger.info(self.cstProcess)
            self.cstStatus ='on'

    def stopWork(self):
        oFilePath=os.path.join(self.taskFileDir,"terminate.txt")
        file=open(oFilePath,"w")
        file.close()
        self.cstStatus=='off'
        self.taskIndex=0

    def sendTaskFile(self,paramname_list,value_list,run_name):
        tf=os.path.join(self.taskFileDir,str(self.taskIndex)+'.txt')
        f1=open(tf,"w")
        f1.write(run_name+'\n')
        full_param_list=copy.deepcopy(self.paramList)
        if (len(paramname_list)==0 and len(value_list)>0) :
            for i in range(min(len(full_param_list),len(value_list))):
                full_param_list[i]['value']=value_list[i]
        elif (len(paramname_list)>0 and len(value_list)>0):
            for i in range(len(paramname_list)):
                for idict in full_param_list:
                    if(idict["name"]==paramname_list[i]):
                        idict.update({"value":value_list[i]}) 

        for i in range(len(full_param_list)):
            f1.write(str(full_param_list[i]['name'])+'\n')
            f1.write(str(full_param_list[i]['value'])+'\n')
        f1.close()
    

    def createMainCSTbatch(self,hFilePath,mFilePath):
        oFilePath=os.path.join(self.tempPath,"main.bas")
        
        ###saveVBConfigs
        cFilePath=os.path.join(self.tempPath,"configs.txt")
        file_c=open(cFilePath,'w')
        file_c.write(self.cstType+'\n')
        file_c.write(os.path.abspath(self.resultDir)+'\n')
        file_c.write(os.path.abspath(self.tmpCstFilePath)+'\n')
        file_c.write(os.path.abspath(self.taskFileDir)+'\n')
        file_c.close()

        ###
        file_1 = open(hFilePath,'r')
        file_2 = open(mFilePath,'r')
        list1 = []
        for line in file_1.readlines():
            ssd=line
            ssd=re.sub('\'EXTERN\'','',ssd)
            ssd=re.sub('%CONFIGFILEPATH%',os.path.abspath(cFilePath).replace('\\','\\\\'),ssd)
            list1.append(ssd)
        file_1.close()
        list2 = []
        for line in file_2.readlines():
            ss=line
            list2.append(ss)
        file_2.close()
        file_new = open(oFilePath,'w')
        for i in range(len(list1)):
            file_new.write(list1[i])
        for i in range(len(list2)):
            file_new.write(list2[i])
        file_new.close()
        return os.path.abspath(oFilePath)

    def change_uvalue(self,u_param_list,u_value_list):
        self.u_param_list=u_param_list
        self.u_value_list=u_value_list

    def start(self):
        run(self)
    def run(self):
        if self.cstStatus=='off':
            raise BaseException
        self.sendTaskFile(self.u_param_list,self.u_value_list,self.resultName)
        pathc = os.path.join(self.resultDir, self.resultName)
        #WAIT RESULTS
        flagPath=os.path.join(self.taskFileDir,str(self.taskIndex)+".success")
        waitTime=0
        startTime=time.time()
        self.log.logger.info("WorkerID:%r Run:%r Name:%r started."%(self.ID,self.taskIndex,self.resultName))
        self.log.logger.info("Start Time:%r"% time.ctime())
        while not os.path.exists(flagPath):
            #ADD process check HERE
            rcode =self.cstProcess.poll()
            if rcode is None:
                pass
                
            else:
                self.log.logger.error("CST ENV stopped")
                os._exit(0)
            time.sleep(1)
            currentTime=time.time()
            escapedTime=currentTime-startTime
            if (escapedTime>self.maxWaitTime):
                raise TimeoutError
            
          
        self.log.logger.info("WorkerID:%r Run:%r Name:%r success."%(self.ID,self.taskIndex,self.resultName))
        self.log.logger.info("ElapsedTime:%r"%escapedTime)
        self.log.logger.info("End Time:%r"%time.ctime())
        self.taskIndex+=1
        runResult=result.readModeResult(pathc,1)
        return runResult

    def stop(self):
        self.log.logger.info("Stopping CST, Please Wait")
        secs=0
        if not self.cstlog is None:
            self.cstlog.close()
        rcode =self.cstProcess.poll()
        if rcode is None:
            self.stopWork()
        while secs<self.maxStopWaitTime :
            rcode =self.cstProcess.poll()
            if rcode is None:
                time.sleep(1)
                secs+=1
                self.log.logger.info("%r secs"%(secs))
            else:
                self.log.logger.info("Stop success")
                return True
        self.log.logger.error("failed to stop cst, try killing the process.")
        self.kill()
        time.sleep(10)
        if not self.cstProcess is None:
            self.log.logger.warning("cst killed.")
            return True
        else:
            self.log.logger.error("killing failed")
            return False

    def kill(self):
        if not self.cstlog is None:
            self.cstlog.close()
        if self.cstProcess is None:
            self.cstProcess.kill()
        



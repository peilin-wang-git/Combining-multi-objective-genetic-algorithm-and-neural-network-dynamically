import cstworker
import time,os
import threading
import queue
import copy
import time


class manager(object):
    def __init__(self,gconfm,pconfm,params,log_obj,maxTask=2):
        super().__init__()
        
        self.log=log_obj
        self.gconf=gconfm.conf
        self.pconf=pconfm.conf
        self.curcwd=os.getcwd()
        self.cstPatternDir=self.gconf['BASE']['datadir']
        os.chdir(pconfm.currProjectDir)
        self.tempPath=os.path.abspath(self.pconf['DIRS']['tempdir'])
        self.taskFileDir=os.path.abspath(self.pconf['DIRS']['tempdir'])
        self.resultDir=os.path.abspath(self.pconf['DIRS']['resultdir'])
        self.cstProjPath=os.path.abspath(self.pconf['CST']['CSTFilename'])
        self.cstType=self.pconf['PROJECT']['ProjectType']
        self.paramList=params
        os.chdir(self.curcwd)

        self.sleep_time = 18
        
        #PARALLEL
        self.maxParallelTasks=maxTask
        self.cstWorkerList=[]
        self.mthreadList=[]
        self.taskQueue=queue.Queue()
        self.resultQueue=queue.Queue()
        workerID=0
        for i in range(self.maxParallelTasks):            
            self.cstWorkerList.append(self.createLocalWorker(str(workerID)))
            print("created cstworker. ID=",workerID)
            workerID+=1

    def getResultDir(self):
        return self.resultDir

    def mthread(self,idx):
        while True:            
            if (self.taskQueue.qsize()!=0):
                
                mtask=self.taskQueue.get()
                print(mtask)
                resultvalue=self.cstWorkerList[idx].runWithParam(mtask['pname_list'],mtask['v_list'],mtask['job_name'])
                result={}
                result['value']=resultvalue
                result['name']=mtask['job_name']
                self.resultQueue.put(result)
                
                present_time = time.time()
                while  present_time - self.time<self.sleep_time:
                    print("time:",present_time - self.time)
                    time.sleep(self.sleep_time - (present_time - self.time))
                    present_time = time.time()
                self.time = time.time()

            else:
                break



    def createLocalWorker(self,workerID):
        mconf={}
        mconf['tempPath']=os.path.join(self.tempPath,"worker_"+workerID)
        mconf['CSTENVPATH']=self.gconf['CST']['cstexepath']
        mconf['ProjectType']=self.cstType
        mconf['cstPatternDir']=self.cstPatternDir
        mconf['resultDir']=self.resultDir
        mconf['cstPath']=self.cstProjPath
        mconf['paramList']=self.paramList
        os.makedirs(mconf['tempPath'],exist_ok=True)
        print(mconf['tempPath'])
        mconf['taskFileDir']=os.path.join(self.taskFileDir,"worker_"+workerID)
        mcstworker_local=cstworker.local_cstworker(id=workerID, type="local",config=mconf,log_obj=self.log)
        return mcstworker_local

    def start(self):
        for i in range(self.maxParallelTasks):
            ithread=threading.Thread(target=manager.mthread,args=(self,i,))
            self.mthreadList.append(ithread)
        
        self.time = time.time()
        for thread in self.mthreadList:
            thread.start()
            
            present_time = time.time()
            while  present_time - self.time<self.sleep_time:
                print("time:",present_time - self.time)
                time.sleep(self.sleep_time - (present_time - self.time))
                present_time = time.time()
            self.time = time.time()
            
    def stop(self):
        for iworker in self.cstWorkerList:
            ithread=threading.Thread(target=iworker.stop)
            ithread.start()

    def synchronize(self):
        #WAIT UNTIL ALL TASK FINISHED
        for thread in self.mthreadList:
            thread.join()
        self.mthreadList=[]

    def getFullResults(self):
        mlist=[]
        for i in range(self.resultQueue.qsize()):
            mlist.append(self.resultQueue.get())
        return mlist

    def getFirstResult(self):
        return self.resultQueue.get()
    def addTask(self,param_name_list=[],value_list=[],job_name="default"):
        self.time = time.time()
        mtask={}
        mtask['pname_list']=param_name_list
        mtask['v_list']=value_list
        mtask['job_name']=job_name
        self.taskQueue.put(mtask)

    def runWithx(self,x,job_name):
        self.addTask(value_list=x,job_name=job_name)
        self.start()
        self.synchronize()
        result=self.getFirstResult()
        return result['value']

    
    def runWithParam(self,name_list,value_list,job_name):
        self.addTask(param_name_list=name_list, value_list=value_list,job_name=job_name)
        self.start()
        self.synchronize()
        result=self.getFirstResult()
        return result['value']


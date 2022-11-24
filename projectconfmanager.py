import configparser
import os,shutil,re,json
import logger
import subprocess
import hashlib
import tempfile
import projectutil

###读取或生成ProjConf.ini文件
class ProjectConfmanager(object):
    def __init__(self,GlobalConfigManager=None,Logger=None):
        self.conf= configparser.ConfigParser()
        self.log=Logger
        self.gconf=GlobalConfigManager.conf

        self.maincwd=os.getcwd()
        self.currProjectDir=None #外部 ref->oldcwd



    def openProjectDir(self,newProjectDir,slient=False):
        self.log.logger.info("尝试打开project目录%s。"% newProjectDir)
        if (self.currProjectDir!=None):
            self.savecfg(slient=slient)
        self.currProjectDir=os.path.abspath(newProjectDir)
        cfgfilename="project.ini"
        cfgpath=os.path.join(self.currProjectDir,cfgfilename)
        if(not os.path.exists(cfgpath)):
            self.log.logger.info("未找到配置文件%s，尝试重新创建。"% cfgfilename)
            self.createProjectConf(cfgfilename=cfgfilename)

        self.checkProjectStatus()



    def savecfg(self,cfgfilename="project.ini",slient=False):
        oldcwd=os.getcwd()
        os.chdir(self.currProjectDir)
        cfgfilePath=os.path.join(self.currProjectDir,"project.ini")
        f=open(cfgfilePath,"w")
        self.conf.write(f)
        f.close()
        os.chdir(oldcwd)

    def createProjectConf(self,cfgfilename="project.ini",projectname=""):
        #所有路径都是相对于.\project\myproject这一文件夹   
        self.log.logger.info("创建配置文件%s。"% cfgfilename)
        self.conf.clear()
        cstfile=None
        currprojdir=self.currProjectDir        
        avilprojname=os.path.basename(os.path.realpath(currprojdir))
        os.chdir(currprojdir)

        currprojname=None
        if(projectname!=""):
            currprojname=projectname
        else:
            self.log.logger.info("未指定project名,使用为默认目录名%s" % avilprojname)
            currprojname=avilprojname

        found =False
        for root,dirs,files in os.walk("."):
            if(found ==True) :
                break
            for file in files:
                if file.endswith('.cst'):
                    cstfile=file
                    self.log.logger.info("找到CST项目文件%s"% file)
                    found=True
                    break
                else:
                    self.log.logger.error("未找到CST项目文件于%s目录"%currprojdir)
                    self.log.logger.error("退出程序")
                    os._exit(1)

        self.conf.add_section("PROJECT")
        self.conf.set('PROJECT','ProjectName',currprojname)
        self.conf.set('PROJECT','ProjectType',currprojname)
        self.conf.set('PROJECT','ProjectDescription',"")
        self.conf.add_section("DIRS")   
        #保存为相对路径
        os.chdir(self.maincwd) 
        projresultdirabspath=os.path.abspath(self.gconf['BASE']['resultdir'])
        projresulttempabspath=os.path.abspath(self.gconf['BASE']['tempdir'])
        os.chdir(self.currProjectDir)
        projresultrelpath=os.path.relpath(projresultdirabspath)
        projtemprelpath=os.path.relpath(projresulttempabspath)
        self.conf.set('DIRS','resultdir',projresultrelpath)
        self.conf.set('DIRS','tempdir',projtemprelpath)
        self.conf.add_section("CST")      
        self.conf.set('CST','CSTFilename',os.path.basename(cstfile))  

        file_md5=self.genMD5FromCST(cstfile)
        self.conf.set('CST','CSTFileMD5',file_md5.hexdigest())

        self.conf.set('CST','UseMpi',"False")  
        self.conf.set('CST','MpiNodeList',"")  
        self.conf.set('CST','UseRemoteCalculaton',"False")  
        self.conf.set('CST','DCMainControlAddress',"")  
        self.conf.add_section("PARAMETERS")
        self.conf.set('PARAMETERS','paramfile',"params.json")  
        self.readParametersFromCST()

        #保存配置文件
        self.savecfg()

        os.chdir(self.maincwd)#切回工作目录
        self.log.logger.info("创建配置文件结束。")

    def printconf(self):
        self.log.logger.info("项目信息:")
        self.log.logger.info("ProjectName:%s"% self.conf['PROJECT']['ProjectName'])
        self.log.logger.info("ProjectType:%s"% self.conf['PROJECT']['ProjectType'])
        self.log.logger.info("ProjectDescription:%s"% self.conf['PROJECT']['ProjectDescription'])
        self.log.logger.info("项目result目录:%s"% self.conf['DIRS']['resultdir'])
        self.log.logger.info("项目temp目录:%s"% self.conf['DIRS']['tempdir'])
        self.log.logger.info("CST文件名:%s"% self.conf['CST']['CSTFilename'])
        self.log.logger.info("CSTFileMD5:%s"% self.conf['CST']['CSTFileMD5'])
        self.log.logger.info("使用MPI:%s"% self.conf['CST']['UseMpi'])
        self.log.logger.info("MPI节点文件:%s"% self.conf['CST']['MpiNodeList'])
        self.log.logger.info("UseRemoteCalculaton:%s"% self.conf['CST']['UseRemoteCalculaton'])
        self.log.logger.info("DCMainControlAddress:%s"% self.conf['CST']['DCMainControlAddress'])
        self.log.logger.info("参数列表文件:%s"% self.conf['PARAMETERS']['paramfile'])
        self.printparams()

    def printparams(self):
        oldcwd=os.getcwd()
        os.chdir(self.currProjectDir)
        paramfile=self.conf['PARAMETERS']['paramfile']
        f=open(paramfile,"r")
        pamlist=json.load(f)
        print(pamlist)
        f.close()
        os.chdir(oldcwd)

    def getParamsList(self, jsonpath=""):
        oldcwd=os.getcwd()
        os.chdir(self.currProjectDir)
        if (jsonpath==""):
            paramfile=self.conf['PARAMETERS']['paramfile']
        else:
            paramfile=jsonpath
        f=open(paramfile,"r")
        pamlist=json.load(f)
        f.close()
        os.chdir(oldcwd)
        return pamlist


    def genMD5FromCST(self,cstfilepath):
        #生成MD5
        fp=open(cstfilepath,"rb")
        dat=fp.read()
        file_md5 = hashlib.md5(dat)
        fp.close()
        return file_md5
        

    def readParametersFromCST(self,savejsonname="params.json"):        
        projectname=self.conf['PROJECT']['ProjectName']
        oldcwd=os.getcwd()
        os.chdir(self.currProjectDir)
        self.log.logger.info("正在从CST文件中读取参数列表。")
        tempfile.tempdir=self.conf['DIRS']['tempdir']
        tmp_bas=tempfile.NamedTemporaryFile(mode="w",suffix='.bas',delete =False)
        tmp_txt=tempfile.NamedTemporaryFile(mode="w",suffix='.txt',delete =False)
        tmp_cst=tempfile.NamedTemporaryFile(mode="wb",suffix='.cst',delete =False)
        tmp_bas_name = tmp_bas.name
        tmp_cst_name = tmp_cst.name
        tmp_txt_name = tmp_txt.name
        os.chdir(self.maincwd)
        vbasrcpath=os.path.abspath(os.path.join(self.gconf['BASE']['datadir'],"readParamsT.vb"))
        os.chdir(self.currProjectDir)
        vbadstpath=os.path.join(tempfile.gettempdir(),tmp_bas_name)
        midfilepath=os.path.abspath(os.path.join(tempfile.gettempdir(),tmp_txt_name))
        
        
        cstfilepath=os.path.abspath(os.path.join(self.currProjectDir,self.conf['CST']['CSTFilename']))
        tmpcstpath=os.path.abspath(os.path.join(tempfile.gettempdir(),tmp_cst_name))
        #tmpcstfile
        fcst=open(cstfilepath,"rb")
        content=fcst.read()
        tmp_cst.file.write(content)
        
        tmp_cst.file.close()
        
        
        file_1=open(vbasrcpath,"r")
        file_2=tmp_bas.file
        list1=[]
        for line in file_1.readlines():
            ssd=line
            ssd=re.sub('%PARAMDSTPATH%',midfilepath.replace('\\','\\\\'),ssd)
            ssd=re.sub('%CSTPROJFILE%',tmpcstpath.replace('\\','\\\\'),ssd)
            list1.append(ssd)
        file_1.close()
        for i in range(len(list1)):
            file_2.write(list1[i])
        file_2.close()

        #command=

        command = "\""+self.gconf['CST']['cstexepath'] + "\"" +" -m " + "\"" + vbadstpath + "\""
        with subprocess.Popen(command, stdout=subprocess.PIPE,shell=True) as self.cstProcess:
            for line in self.cstProcess.stdout:
                self.log.logger.info(line)
        os.chdir(self.currProjectDir)
        projectutil.custom_ascii_2_json(midfilepath,savejsonname)
        os.chdir(oldcwd)
        pass


    
    def checkProjectStatus(self,cfgfilename="project.ini",slient=False,force=False):
        oldcwd=os.getcwd()
        os.chdir(self.currProjectDir)
        cfgpath=cfgfilename
        cstfileflag=False
        cfgfileflag=False
        if(os.path.exists(cfgpath)):
            cfgfileflag=True
        else:
            self.log.logger.error("未找到配置文件%s\n"% cfgfilename)
            self.log.logger.error("退出程序")
            os._exit(1)

        self.conf.clear()
        self.conf.read(cfgfilename)
        cstfilepath=self.conf['CST']['CSTFilename']
        if(os.path.exists(cstfilepath)):
            cstfileflag=True
        else:
            self.log.logger.error("未找到cst文件%s\n"% cstfilepath)
            self.log.logger.error("退出程序")
            os._exit(1)
        currCSTMD5=self.genMD5FromCST(cstfilepath).hexdigest()
        savedCSTMD5=self.conf['CST']['CSTFileMD5']
        if(currCSTMD5!=savedCSTMD5):
            self.log.logger.warning("记录的CST文件MD5%s与实际的%s不一致，已被修改"%(savedCSTMD5,currCSTMD5))            
            nex=False
            while(nex!=True):
                self.log.logger.warning("是否重新生成参数列表n/y")
                a=input() 
                if(a=='y' or a=='Y'):
                    self.log.logger.info("正在重新生成参数列表")
                    self.readParametersFromCST()
                    self.savecfg()
                    self.log.logger.warning("生成结束，请重新运行程序")
                    os._exit(1)
                elif(a=='n' or a=='N'):
                    nex=True
            nex=False
            while(nex!=True):
                self.log.logger.warning("是否更新MD5n/y")
                a=input() 
                if(a=='y' or a=='Y'):
                    nex=True
                elif(a=='n' or a=='N'):                    
                    os._exit(1)
            self.conf['CST']['CSTFileMD5']=currCSTMD5
            self.savecfg()
            self.log.logger.warning("已更新保存的MD5")
        paramjsonpath=self.conf['PARAMETERS']['paramfile']
        if(not os.path.exists(paramjsonpath)):
            self.log.logger.warning("未找到参数列表文件%s"%paramjsonpath)            
            nex=False
            while(nex!=True):
                self.log.logger.warning("是否重新生成参数列表n/y")
                a=input() 
                if(a=='y' or a=='Y'):
                    self.log.logger.info("正在重新生成参数列表")
                    self.readParametersFromCST()
                    self.savecfg()
                    self.log.logger.warning("生成结束，请重新运行程序")
                    os._exit(1)
                elif(a=='n' or a=='N'):
                    nex=True
            nex=False
        self.log.logger.info("%s测试结束\n"% cfgfilename)
        os.chdir(oldcwd)







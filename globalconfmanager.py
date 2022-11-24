import configparser
import os,shutil
import logger,json
class GlobalConfmanager(object):
    
    def __init__(self,Logger=None):
        self.projlist=[]
        self.log=Logger
        self.conf = configparser.ConfigParser()
        if (not os.path.exists(r'.\config')):
            os.makedirs(r'.\config')
        self.def_global_cfg_path=r'.\config\default.ini'
        self.curr_global_cfg_path=r'.\config\current.ini'
        ##Generate new default conf file
        if (not os.path.exists(self.def_global_cfg_path)):
            self.conf.add_section("BASE")
            self.conf.set('BASE','datadir','./data')
            self.conf.set('BASE','tempdir','./temp')
            self.conf.set('BASE','logdir','./log')  
            self.conf.set('BASE','resultdir','./result')        
            self.conf.add_section("CST")
            self.conf.set('CST','cstver','')
            self.conf.set('CST','cstexepath','')
            self.conf.add_section("PROJECT")
            self.conf.set('PROJECT','currprojdir','')

        if (not os.path.exists(self.curr_global_cfg_path)):

            self.log.logger.warning("未找到curr_global_cfg.")
            self.log.logger.warning("使用default.")
        else:
            self.log.logger.warning("找到curr_global_cfg:%s" % self.curr_global_cfg_path)
            self.conf.read(self.curr_global_cfg_path)            

    def printconf(self):
        self.log.logger.info("全局信息:")
        self.log.logger.info("data目录:%s"% self.conf['BASE']['datadir'])
        self.log.logger.info("temp目录:%s"% self.conf['BASE']['tempdir'])
        self.log.logger.info("log目录:%s"% self.conf['BASE']['logdir'])
        self.log.logger.info("result目录:%s"% self.conf['BASE']['resultdir'])
        self.log.logger.info("CST版本:%s"% self.conf['CST']['cstver'])
        self.log.logger.info("CSTEXE路径:%s"% self.conf['CST']['cstexepath'])
        self.log.logger.info("project目录:%s"% self.conf['PROJECT']['currprojdir'])

    def saveconf(self):
        f=open(self.curr_global_cfg_path,'w')
        self.conf.write(f)
        self.log.logger.info("已保存全局配置文件于%s。" %self.curr_global_cfg_path)
        f.close()



    def checkConfig(self):
        #测试CST环境位置
        self.log.logger.info("检查全局配置开始.")
        cfg=self.conf
        if (not os.path.exists(cfg['CST']['cstexepath'])):
            self.log.logger.warning("定义的cstexepath:%s不存在。" %cfg['CST']['cstexepath'] )
            self.log.logger.warning("尝试寻找cstexepath。")
            try:
                cfg['CST']['cstexepath'],cfg['CST']['cstver']=self.findCSTenv()
            except FileNotFoundError:
                self.log.logger.error("未找到CST ENV PATH,请从config指定PATH")
                self.log.logger.error("退出程序")
                os._exit(1)

        #测试CST版本
        if (int(cfg['CST']['cstver'])<2015 or int(cfg['CST']['cstver'])>2020):
            self.log.logger.warning("定义的cstver不正确(req:cstver>2015&&<2021)。")
            self.log.logger.warning("尝试从cstexepath寻找cstver。")
            self.log.logger.error("功能未实现，请手动指定。")
            self.log.logger.error("退出程序")
            os._exit(1)
        self.log.logger.debug("CST ENV PATH为%s" % cfg['CST']['cstexepath'])
        self.log.logger.debug("CST 版本为%s" % cfg['CST']['cstver'])
        #测试各个路径是否存在，若否则创建目录
        for names,dirs in cfg['BASE'].items():
            if(not os.path.exists(dirs)):                
                os.makedirs(dirs)
                self.log.logger.info("已建立%s于%s。" %(names,dirs))

        self.log.logger.info("检查全局配置结束.")

    def findCSTenv(self):
        self.log.logger.info("寻找CSTenv开始")
        ed1=":\\Program Files (x86)\\CST Studio Suite "
        ed2="\\CST DESIGN ENVIRONMENT.exe"
        for j in range(2015,2025):        
            for i in range(ord('C'),ord('Z')):            
                if os.path.exists(chr(i)+ed1+str(j)+ed2):
                    success=True
                    CSTexepath=chr(i)+ed1+str(j)+ed2
                    CSTver=str(j)
                    self.log.logger.info("FOUND CST VERSION %s at %s"% (str(j),CSTexepath))
                    self.log.logger.info("寻找CSTenv结束")
                    return CSTexepath,CSTver
        raise FileNotFoundError
        return None,None

    ###检查project
    def checkCurrProject(self):
        self.log.logger.info("检查project开始")
        curprojdirpath=self.conf['PROJECT']['currprojdir']
        if os.path.exists(curprojdirpath):
            self.log.logger.info("找到project目录:%s"% curprojdirpath)
        else:
            self.log.logger.info("未找到project目录:%s"% curprojdirpath)
            os.makedirs(curprojdirpath,exist_ok=True)
            self.log.logger.info("已创建project目录:%s"% curprojdirpath)
        self.log.logger.info("检查project结束")



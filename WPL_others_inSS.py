import os
import sys
import shutil
import re
import result
import math
import matplotlib.pyplot as plt
import cstmanager
import yfunction
import time
import random
from decimal import Decimal
import csv
import seaborn as sns
from keras.models import load_model
import copy
import numpy as np
import WPL_transformer
import pandas as pd
from pathlib import Path

class others_inSS(object):
    def __init__(self, save_location, input_name, input_cst_name, hidden_name, output_name, resonant_frequency, link_parameter):
        self.save_location = save_location
        self.input_name = input_name
        self.input_cst_name = input_cst_name
        self.hidden_name = hidden_name
        self.output_name = output_name
        self.init_frequency = [160, 850]
        self.cell = [20, 30]
        self.resonant_frequency = resonant_frequency
        self.link_parameter = link_parameter

    def write_many(self, title, data, count):
        name = self.save_location + str(count) + "_" + title + ".csv"
#         print("name:",name)
        data.to_csv(name, index=True, sep=',')
        self.write_log("----data "+str(count) + "_" + title + ".csv"+" has produced")
    
    def write_log(self, data):
        txt_name = self.save_location + "log.txt"
        file = open(txt_name,"a") 
        file.write(time.ctime() + data+"\n")
        file.close()
        
    def get_post_processing(self, sample):
        sample = copy.deepcopy(sample)
        
        sample["distance_to_resonant_frequency"] = abs(sample["Mode1_frequency"]-self.resonant_frequency)
        sample["distance_2ed_2_1st"] = abs(sample["Mode2_frequency"] - sample["Mode1_frequency"])
        
        limit_list = [100]+list(np.zeros(len(self.output_name)-1))
        for j in range(len(self.output_name)):
            sample = sample[sample[self.output_name[j]]>limit_list[j]]
        
        sample = sample[sample["Mode1_R_divide_Q"]>10]
        sample = sample[sample["Mode1_shunt_impedance"]>5000]
        sample.reset_index(inplace=True,drop=True)

        return sample

    def transfer(self, link_signal, samples_destributed, count):#str(1000000+self.count)[1:]+"_"+name，000040_CHK
        present_time = "_"+str(time.ctime()).replace(':','').replace(' ','_')
        self.link_parameter[link_signal]["present_filename"] = str(1000000+count)[1:]+"_" + self.link_parameter[link_signal]["computer_name"][:7] + present_time
        path = r"\\%s\%s\%s.csv"%(self.link_parameter[link_signal]["computer_name"],self.link_parameter[link_signal]["disk"],self.link_parameter[link_signal]["present_filename"])
        url=Path(path)
        # url=Path("\\\\"+link_parameter[link_signal]["computer_name"]+"\\"+link_parameter[link_signal]["disk"]+"\\")
        # print(url.exists())
        self.write_many(self.link_parameter[link_signal]["present_filename"]+".csv", samples_destributed,count)
        samples_destributed.to_csv(url, index=True,sep=',')
        self.write_log("----"+path+"has created")
    
    def distribution_cumputer(self, samples_,count):
        samples = copy.deepcopy(samples_)
        samples.reset_index(inplace=True,drop=True)
        samples_num = len(samples)
        print("samples_num:",samples_num)
        samples_distribution_num = int(samples_num/(len(self.link_parameter)+1))

        samples_destributed = []
        for i in range(len(self.link_parameter)+1):
            
            if i!=len(self.link_parameter):
                samples_part = samples.loc[i*samples_distribution_num: (i+1)*samples_distribution_num-1]
                samples_part.reset_index(inplace=True,drop=True)
                samples_destributed.append(samples_part)
                print("#####",i,"\n",samples_part)
            else:
                samples_part = samples.loc[i*samples_distribution_num: samples_num-1]
                samples_part.reset_index(inplace=True,drop=True)
                samples_destributed.append(samples_part)
                print("#####",i,"\n",samples_part)
        print("samples_destributed:",samples_destributed)
        for i in range(len(self.link_parameter)):
            self.transfer(i, samples_destributed[i], count)#以时间命名
        
        return samples_destributed[len(self.link_parameter)]
    
    def get_distribution_result(self, count):
        samples_distributed = pd.DataFrame()
        for i in range(len(self.link_parameter)):
            # path = r"\\\\"+self.link_parameter[i]["computer_name"]+"\\"+self.link_parameter[i]["disk"]+"\\"+self.link_parameter[i]["present_filename"]+"_Withresult.csv"
            path = r"\\%s\%s\%s.csv"%(self.link_parameter[i]["computer_name"],self.link_parameter[i]["disk"],self.link_parameter[i]["present_filename"]+"_Withresult")
            url=Path(path)#(r'\\WIN-UKQU29OQ0SJ\d\5_class_present_predict_2_all.csv')
            while(url.exists()!=True):
                sleep_time = 10
                time.sleep(sleep_time)
                self.write_log("----"+path+" did not exist, had waited %ss"%sleep_time)
                print(time.ctime(), path+" did not exist, had waited %ss"%sleep_time)
            samples_temp = pd.read_csv(url).drop(["Unnamed: 0"],1)
            self.write_many(self.link_parameter[i]["present_filename"]+"_Withresult.csv", samples_temp,count)
            samples_distributed = samples_distributed.append(samples_temp)
        return samples_distributed
        
    def set_round(self, x):
        x_after_set = []
        for x0 in x:
            x_after_set.append(x0)
        return x_after_set
    
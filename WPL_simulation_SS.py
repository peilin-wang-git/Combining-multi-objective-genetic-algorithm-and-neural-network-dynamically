import os
import sys
import shutil
import re
# import result
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
import WPL_others_inSS
import pandas as pd

class simulation(object):
    def __init__(self, save_location, input_name, input_cst_name, hidden_name, output_name, link_parameter, Rt_value, manager, log, text_name, resonant_frequency):
        self.save_location = save_location
        self.input_name = input_name
        self.input_cst_name = input_cst_name
        self.hidden_name = hidden_name
        self.output_name = output_name
        self.init_frequency = [160, 850]
        self.cell = [20, 30]
        self.link_parameter = link_parameter
        self.other_function = WPL_others_inSS.others_inSS(self.save_location, self.input_name, self.input_cst_name, self.hidden_name, self.output_name, resonant_frequency, link_parameter)
        self.Rt_value = Rt_value
        self.text_name = text_name
        
        self.w = manager
        # self.results = result
        self.log = log
        


    def reflect_parameter(self, samples):
        samples.reset_index(inplace=True,drop=True)
        Rt_left = self.Rt_value[0]
        Rt_right = self.Rt_value[1]
        samples["Req"] = samples["random1"]+Rt_left#116~216
        samples["nose"] = samples["random2"] * (samples["Req"] - Rt_left)#100~200
        
        samples["R3_left"] = (1-samples["random3_l"]) * (samples["Req"] - Rt_left - samples["nose"])#0~190
        samples["R3_right"] = (1-samples["random3_r"]) * (samples["Req"] - Rt_right - samples["nose"])#0~190
        for i in range(len(samples)):
            samples.loc[i,"Leq"] = max(samples.loc[i, "R3_left"], samples.loc[i,"R3_right"])*2 + samples.loc[i, "random4"]#1~630

        samples["concave_L"] = samples["random5_l"] * samples["Leq"]/2
        for i in range(len(samples)):
            samples.loc[i,"alpha_left"] = math.pi/2 - math.atan(samples.loc[i,"concave_L"]/(samples.loc[i,"random3_l"]*(samples.loc[i,"Req"] - Rt_left - samples.loc[i,"nose"])))
            samples.loc[i,"r2_left"] = samples.loc[i,"random8_l"] * samples.loc[i,"R3_left"] * (math.sin(samples.loc[i,"alpha_left"]) + math.cos(samples.loc[i,"alpha_left"])) / (1 + math.cos(samples.loc[i,"alpha_left"]))
            samples.loc[i,"r1_left"] = samples.loc[i,"random7_l"] * (samples.loc[i,"nose"]**2) / (math.tan(samples.loc[i,"alpha_left"]) * (-samples.loc[i,"nose"] + samples.loc[i,"nose"]/math.sin(samples.loc[i,"alpha_left"])))   
            tan = math.tan(samples.loc[i,"alpha_left"])
            alpha = 1/math.sin(samples.loc[i,"alpha_left"]) - 1
            samples.loc[i,"r0_left"] = samples.loc[i,"random6_l"] * (samples.loc[i,"nose"] - (tan*samples.loc[i,"r1_left"]*alpha))
        
        samples["cocave"] = samples["random5_r"] * samples["Leq"]/2
        for i in range(len(samples)):
            samples.loc[i,"alpha_right"] = math.pi/2 - math.atan(samples.loc[i,"cocave"]/(samples.loc[i,"random3_r"]*(samples.loc[i,"Req"] - Rt_right - samples.loc[i,"nose"])))
            samples.loc[i,"r2_right"] = samples.loc[i,"random8_r"] * samples.loc[i,"R3_right"] * (math.sin(samples.loc[i,"alpha_right"]) + math.cos(samples.loc[i,"alpha_right"])) / (1 + math.cos(samples.loc[i,"alpha_right"]))
            samples.loc[i,"r1_right"] = samples.loc[i,"random7_r"] * (samples.loc[i,"nose"]**2) / (math.tan(samples.loc[i,"alpha_right"]) * (-samples.loc[i,"nose"] + samples.loc[i,"nose"]/math.sin(samples.loc[i,"alpha_right"])))   
            tan = math.tan(samples.loc[i,"alpha_right"])
            alpha = 1/math.sin(samples.loc[i,"alpha_right"]) - 1
            samples.loc[i,"r0_right"] = samples.loc[i,"random6_r"] * (samples.loc[i,"nose"] - (tan *samples.loc[i,"r1_right"]*alpha))

        return samples
    
    def get_y_trans_r_aprallel(self, xs, name):
        y = []
        x_temp = []
        for x in xs:
            x_temp.append(self.other_function.set_round(x))
        xs = x_temp
        
        for j,x in enumerate(xs):
            self.w.addTask(self.input_cst_name,x,name[j])
        
        self.w.start()
        self.w.synchronize()#同步 很重要
        rg=self.w.getFullResults()
        ###SORT RESULTS
        
        rg=sorted(rg,key=lambda x: int(x['name'][x['name'].rfind("_")+1:]))
        
        name = []
        for irg in rg:
            print(irg['name'])
            # y.append(irg['value'])
            name.append(irg['name'])
            # r.append(self.yFunc.Y(irg['value'], nor))
            # r.append(np.sum(irg['value']))
        
        print("y",y)
        # print("r",r)
        
        if len(y)==1:
           return y[0] 
        else:
           return name

    def compire_str(self,a,b):
        # print("compire:"+a+"###"+b+"###")
        if len(a)!=len(b):
            #print("False because len")
            return False
        else:
            for i in range(len(a)):
                #print(a[i],b[i])
                if a[i]!=b[i]:
                    #print("False because str")
                    return False
            return True
 
    def get_modes(self, title, sample, nmodes, save_location):
        location = save_location + title + '\\'
        sub_files = os.listdir(location)

        print("location:\n",location)
        for i in range(len(self.text_name)):        
            for sub_file in sub_files:
                sub_m = os.path.join(location,sub_file)
                
                if self.compire_str(self.text_name.loc[i,"name"]+".txt", sub_file):
                    sample[self.text_name.loc[i,"reflect_name"]] = self.get_value(sub_m, self.text_name.loc[i,"line"], self.text_name.loc[i,"range_down"], self.text_name.loc[i,"range_up"])

        return sample

    def get_value(self, file, line_num, range_up, range_down):
        file = open(file) 
        i = 0
        while 1:
            i += 1
            line = file.readline()
            # print("present:",line_num,i)
        # f = open(file)
        # text = f.read()
            if i==line_num:
                # print("***the whole text:",line)
                # print("***the text:",line[range_up:])
                print("***the text:",line[range_up:range_down])
                return float(line[range_up:range_down])
            if not line:
                print("not this")
                break
        
    def simulate(self, samples_, which_phase, simulation_flag, count, user):
        if user=="master":
            try:
                samples = copy.deepcopy(samples_[self.hidden_name+self.input_name+self.output_name+["phase"]])
            except:
                samples = copy.deepcopy(samples_[self.hidden_name+self.input_name+["phase"]])
        else:
            try:
                samples = copy.deepcopy(samples_[self.hidden_name+self.input_name+self.output_name+["phase", "simulation_name"]])
            except:
                samples = copy.deepcopy(samples_[self.hidden_name+self.input_name+["phase", "simulation_name"]])
        
        phase_list = list(samples["phase"])
       
        samples.reset_index(inplace=True,drop=True)
        # columns_need = self.hidden_name+self.input_name+["simulation_name","phase"]
        if user=="master":
            number_list = range(0,len(samples),1)
            for i in number_list:
                print(samples.loc[i,"phase"], i)
                samples.loc[i, "simulation_name"] = str(1000000+count)[1:]+"_"+samples.loc[i,"phase"]+"_Mode"+str(101)[1:]+"_"+str(i)
            if len(self.link_parameter)!=0:
                samples = self.other_function.distribution_cumputer(samples,count)
        num_samples = len(samples)
        
        if "New" in phase_list:
            if which_phase[0]==0:
                samples_get_result = pd.DataFrame()
                x = np.append(np.full((num_samples,1),self.init_frequency[0]),np.full((num_samples,1),self.init_frequency[1]),axis=1)
                x = np.append(x,np.full((num_samples,1),self.Rt_value[0]),axis=1)
                x = np.append(x,np.full((num_samples,1),self.Rt_value[1]),axis=1)
                x = np.append(x,np.full((num_samples,1),self.cell[0]),axis=1)
                x = np.append(x, np.array(samples[self.input_name]), axis=1)
                # number_list = range(0,num_samples,1)
                cst_name = self.get_y_trans_r_aprallel(x, list(samples["simulation_name"]))
                for i,one_of_name in enumerate(cst_name):
                    sample = self.get_modes(one_of_name, samples.loc[samples["simulation_name"]==one_of_name], 1, "result\\")
                    samples_get_result = samples_get_result.append(sample)

                # self.other_function.write_many("samples_get_result_"+str(1),samples_get_result, count)
            
            elif which_phase[0]==1:
                mode_num = simulation_flag[0]

                samples_get_result = pd.DataFrame()
                x = np.append(np.full((num_samples,1),self.init_frequency[0]),np.full((num_samples,1),self.init_frequency[1]),axis=1)
                x = np.append(x,np.full((num_samples,1),self.Rt_value[0]),axis=1)
                x = np.append(x,np.full((num_samples,1),self.Rt_value[1]),axis=1)
                x = np.append(x,np.full((num_samples,1),self.cell[0]),axis=1)
                x = np.append(x, np.array(samples[self.input_name]), axis=1)

                number_list = range(0,num_samples,1)
                
                samples_get_result_temp = pd.DataFrame()
                location_ini = np.where(np.array(number_list) == simulation_flag[1])[0][0]
                
                cst_name_exist = list(samples["simulation_name"])[:location_ini]

                cst_name = self.get_y_trans_r_aprallel(x[location_ini+1:], list(samples["simulation_name"])[location_ini+1:])
                for i,one_of_name in enumerate(cst_name_exist):
                    sample = self.get_modes(one_of_name, samples.loc[samples["simulation_name"]==one_of_name], mode_num, simulation_flag[2])
                    samples_get_result = samples_get_result.append(sample)
                
                for i,one_of_name in enumerate(cst_name):
                    sample = self.get_modes(one_of_name, samples.loc[number_list[location_ini+1+i]], mode_num, "result\\")
                    samples_get_result = samples_get_result.append(sample)

                # self.other_function.write_many("samples_get_result_"+str(mode_num),samples_get_result, count)
                which_phase[0] = 0
                
        elif "CHK" in phase_list:
            if which_phase[0]==0:
                samples_get_result = pd.DataFrame()
                x = np.append(np.array(samples["Mode1_frequency"]-10).reshape((len(samples),1)),np.array(samples["Mode2_frequency"]+10).reshape((len(samples),1)),axis=1)
                x = np.append(x,np.full((num_samples,1),self.Rt_value[0]),axis=1)
                x = np.append(x,np.full((num_samples,1),self.Rt_value[1]),axis=1)
                x = np.append(x,np.full((num_samples,1),self.cell[1]),axis=1)
                x = np.append(x, np.array(samples[self.input_name]), axis=1)
                # number_list = range(0,num_samples,1)
                cst_name = self.get_y_trans_r_aprallel(x, list(samples["simulation_name"]))
                for i,one_of_name in enumerate(cst_name):
                    sample = self.get_modes(one_of_name, samples.loc[samples["simulation_name"]==one_of_name], 1, "result\\")
                    samples_get_result = samples_get_result.append(sample)

                # self.other_function.write_many("high_accuracy_result_"+str(1),samples_get_result, count)
            
            elif which_phase[0]==1:
                mode_num = simulation_flag[0]

                samples_get_result = pd.DataFrame()
                x = np.append(np.array(samples["Mode1_frequency"]-10).reshape((len(samples),1)),np.array(samples["Mode2_frequency"]+10).reshape((len(samples),1)),axis=1)
                x = np.append(x,np.full((num_samples,1),self.Rt_value[0]),axis=1)
                x = np.append(x,np.full((num_samples,1),self.Rt_value[1]),axis=1)
                x = np.append(x,np.full((num_samples,1),self.cell[1]),axis=1)
                x = np.append(x, np.array(samples[self.input_name]), axis=1)

                number_list = range(0,num_samples,1)
                
                samples_get_result_temp = pd.DataFrame()
                location_ini = np.where(np.array(number_list) == simulation_flag[1])[0][0]
                
                cst_name_exist = list(samples["simulation_name"])[:location_ini]

                cst_name = self.get_y_trans_r_aprallel(x[location_ini+1:], list(samples["simulation_name"])[location_ini+1:])
                for i,one_of_name in enumerate(cst_name_exist):
                    sample = self.get_modes(one_of_name, samples.loc[samples["simulation_name"]==one_of_name], mode_num, simulation_flag[2])
                    samples_get_result = samples_get_result.append(sample)
                
                for i,one_of_name in enumerate(cst_name):
                    sample = self.get_modes(one_of_name, samples.loc[samples["simulation_name"]==one_of_name], mode_num, "result\\")
                    samples_get_result = samples_get_result.append(sample)

                # self.other_function.write_many("high_accuracy_result_"+str(mode_num),samples_get_result, count)
                self.which_phase[0] = 0
        else:
            mode_num = 1
            Mode1_frequency = np.array(samples["Mode1_frequency"]).reshape(1,-1)
            Mode2_frequency = np.array(samples["Mode2_frequency"]).reshape(1,-1)
            temp_frequency = np.append(Mode1_frequency,Mode2_frequency,axis=0)
            
            samples_get_result = pd.DataFrame()
            x = np.append(np.min(temp_frequency,axis=0).reshape((num_samples,1))-100,np.max(temp_frequency,axis=0).reshape((num_samples,1))+100,axis=1)
            x = np.append(x,np.full((num_samples,1),self.Rt_value[0]),axis=1)
            x = np.append(x,np.full((num_samples,1),self.Rt_value[1]),axis=1)
            x = np.append(x,np.full((num_samples,1),self.cell[0]),axis=1)
            x = np.append(x, np.array(samples[self.input_name]), axis=1)
            
            number_list = range(0,num_samples,1)
            cst_name = self.get_y_trans_r_aprallel(x, list(samples["simulation_name"]))
            for i,one_of_name in enumerate(cst_name):
                sample = self.get_modes(one_of_name, samples.loc[samples["simulation_name"]==one_of_name], 1, "result\\")
                samples_get_result = samples_get_result.append(sample)
            # self.other_function.write_many("samples_get_result_"+str(mode_num),samples_get_result, count)
        if user=="master":
            samples_get_result = samples_get_result.append(self.other_function.get_distribution_result(count))
        samples_get_result = self.other_function.get_post_processing(samples_get_result)
        return samples_get_result
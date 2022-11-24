#不是正确的，这个版本没有调频
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
import WPL_others_inSS
import WPL_simulation_SS
import pandas as pd

class MLBGA(object):
    def __init__(self, manager, params):
        self.which_phase = [0,0,"train_NN"]
        self.location_named = "result22060301_3000initsamples"#"result22021801_20init_k2_NoPreference_ChooseNum100_50_NoAdjustFrequency"
        
        self.simulation_flag = [1,745,"my_subject_kbann\\"+self.location_named+"\\part1\\result\\"]#[Mode(一般就是1), number, location]
        self.count = 0
        
        self.user = "slave"
        
        self.save_location = "my_subject_kbann\\"+self.location_named+"\\picture\\"
        
        self.input_name = ['Req', 'Leq', 'R3_left', 'R3_right', 'cocave', 'concave_L', 'nose', 'r0_left', 'r0_right', 'r1_left', 'r1_right', 'r2_left', 'r2_right']
        self.input_cst_name = ["fmin","fmax","Rt_left","Rt_right","cell"] + self.input_name
        self.hidden_name = ["random1","random2","random3_r","random3_l","random4","random5_r","random5_l","random6_r","random6_l","random7_r","random7_l","random8_r","random8_l"]
        self.output_name = ['Mode1_frequency','Mode1_R_divide_Q','Mode1_shunt_impedance','Mode1_Q-factor','Mode2_frequency','Mode2_R_divide_Q']
        
        self.text_name =  pd.DataFrame([["Mode1Frequency",3,30,-1,"Mode1_frequency"],
                                        ["Mode1R_Q",3,30,-1,"Mode1_R_divide_Q"],
                                        ["Mode1ShuntImpedance",3,30,-1,"Mode1_shunt_impedance"],
                                        ["Mode1Q-Factor",3,30,-1,"Mode1_Q-factor"],
                                        ["Mode2Frequency",3,30,-1,"Mode2_frequency"],
                                        ["Mode2R_Q",3,30,-1,"Mode2_R_divide_Q"]],columns=["name","line", "range_down","range_up","reflect_name"])
                                        
        self.link_parameter = [#{"ip":"172.1.10.232", "computer_name":"WIN-232", "disk":"e\\cst_new_220531_generalconduct\\my_subject_kbann\\"+self.location_named+"\\picture\\"},
                               {"ip":"172.1.10.230", "computer_name":"WIN-230", "disk":"e\\wangpeilin\\cst_new_220531_generalconduct\\my_subject_kbann\\"+self.location_named+"\\picture\\"},
                               {"ip":"172.1.10.231", "computer_name":"WIN-231", "disk":"e\\wangpeilin\\cst_new_220531_generalconduct\\my_subject_kbann\\"+self.location_named+"\\picture\\"},
                               {"ip":"172.1.10.130", "computer_name":"WIN-UKQU29OQ0SJ", "disk":"d\\wangpeilin\\cst_new_220531_generalconduct\\my_subject_kbann\\"+self.location_named+"\\picture\\"}]


        self.Rt_value = [16,16]
        self.bound = [[100,200],[0.01,0.95],[0.05,0.95],[0.05,0.95],[1,250],[0.15,0.95],[0.15,0.95],[0.05,0.95],[0.05,0.95],[0.05,0.95],[0.05,0.95],[0.05,0.70],[0.05,0.70]]
        
        self.k = 0.1
        self.count = 0
        self.predict_time = 5
        self.end_num = 0*self.predict_time
        self.opt_object = ["R_divide_Q_divide_delta","shunt_impedance_divide_delta","distance_2ed_2_1st_divide_delta","Mode2_R_divide_Q_divide_delta"]
        self.opt_object_from_output = ["Mode1_R_divide_Q","Mode1_shunt_impedance","distance_2ed_2_1st","Mode2_R_divide_Q"]
        self.trend = [1, 1, 1, -1]#1代表取极大值，-1代表极小值
        
        self.congestion = [0,0,0,0]
        
        self.LHS_predict_num = 5000
        self.crossover_predict_num = 5000
        self.mutation_predict_num = 5000
        
        self.simulation_num = 50
        self.crossover_num = 5#inheritance_real
        self.mutation_num = 5#variate_real
        self.predict_num = self.simulation_num-(self.crossover_num+self.mutation_num)#choose_parents_predict less than 1/2*self.allow_predict_num

        self.resonant_frequency = 499.65
        self.constraint_threshold = 0.05

        self.other_function = WPL_others_inSS.others_inSS(self.save_location, self.input_name, self.input_cst_name, self.hidden_name, self.output_name, self.resonant_frequency, self.link_parameter)
        self.simulation = WPL_simulation_SS.simulation(self.save_location, self.input_name, self.input_cst_name, self.hidden_name, self.output_name, self.link_parameter, self.Rt_value, manager, open(os.path.join(manager.getResultDir(), "result.log"), "w"), self.text_name, self.resonant_frequency)
                
        # self.choose_num = 150#choose_predict_num+choose_real_num+parents
        self.parents_feasiable_num = 100
        self.parents_infeasiable_num = 50
                
        #LHSample
        self.ini_allow_predict_number = 3000
        
        #NN
        self.nn_object = 0
        self.nn_model = 0
        
        #transformer
        self.maxlen = 7 #句子的最大长度，输入参数的数量，13
        self.model_dim = 2     # 词嵌入的维度
        self.batch_size = 32
        self.epochs = 500
        
        #build
        self.dense_num_pre = 8
        self.dense_num = 8
        self.head_num = 2
        self.activation = "elu"        
        #eda
        self.parzen_list = pd.DataFrame()
        
        #C++
        self.C_file = "C++thread052301.exe"
        
        self.threshold1 = 1.4
        self.threshold2 = 1
        self.delta_frequent1 = 0.1
        self.delta_frequent2 = 500

    def init(self):
        self.count = self.which_phase[1]
        self.nn_object = WPL_transformer.WPL_Transformer(
                                self.save_location, self.input_name, self.output_name, 
                                self.maxlen, self.model_dim, self.batch_size, self.epochs,
                                self.dense_num_pre,self.dense_num,self.head_num,self.activation)
        self.nn_model = self.nn_object.build_kbann()
        self.other_function.write_log(" NN has built")
        
        if self.which_phase[0]==0:
            dec_val = self.LHS_decision_value(self.ini_allow_predict_number)
            dec_val = self.simulation.reflect_parameter(dec_val)
            
            self.other_function.write_many("init_sample_input", dec_val, self.count)
            self.other_function.write_log(" sample has produced")
        elif self.which_phase[0]==1 and self.which_phase[2]=="init_samples":
            dec_val = pd.read_csv(self.save_location + "0_init_sample_input.csv").drop(["Unnamed: 0"],1)

        if self.which_phase[0]==0 or (self.which_phase[0]==1 and self.which_phase[2]=="init_samples"):
            dec_val["phase"] = "New"
            all_ind = self.simulation.simulate(dec_val, self.which_phase, self.simulation_flag, self.count, "master")
            all_ind["flag_mutated"] = "No"
            all_ind = self.other_function.get_post_processing(all_ind)
            all_ind = self.get_fitness_function(all_ind)
            self.other_function.write_many("all_ind", all_ind, self.count)
        elif self.which_phase[0]==1 and self.which_phase[2]=="train_NN":
            all_ind = pd.read_csv(self.save_location + str(int(self.which_phase[1]/5)*5) + "_all_ind" + ".csv").drop(["Unnamed: 0"],1)
            self.which_phase[0]=0
        else:
            all_ind = pd.read_csv(self.save_location + str(int(self.which_phase[1]/5)*5) + "_all_ind" + ".csv").drop(["Unnamed: 0"],1)
        
        self.nn_model = self.nn_object.train(self.nn_model, all_ind, self.count)
        self.nn_object.test(self.nn_model, all_ind, self.count)
        
        all_ind["flag_simulated_before"] = "No"
        if len(all_ind.loc[all_ind["CV"]==0])!=0:
            ind_sort_feasiable = self.non_domination_sort(all_ind.loc[all_ind["CV"]==0], self.parents_feasiable_num, "Feasiable")
            all_ind_feasiable = self.select_parents(ind_sort_feasiable, self.parents_feasiable_num, "Feasiable")
        else:
            all_ind_feasiable = pd.DataFrame()
        ind_sort_infeasiable = self.non_domination_sort(all_ind.loc[all_ind["CV"]!=0], self.parents_infeasiable_num+self.parents_feasiable_num-len(all_ind_feasiable), "InFeasiable")
        all_ind_infeasiable = self.select_parents(ind_sort_infeasiable, self.parents_infeasiable_num+self.parents_feasiable_num-len(all_ind_feasiable), "InFeasiable")
                    
        parents = copy.deepcopy(all_ind_feasiable)
        parents = parents.append(all_ind_infeasiable)
        self.other_function.write_many("parents", parents, self.count)

        self.other_function.write_log(" "+str(self.count)+" generation parents has produced")
        
        if self.which_phase[0]==0 or self.which_phase[1]%5==0:
            parents_pre = parents
        else:
            parents_pre = pd.read_csv(self.save_location + str(self.which_phase[1]) + "_parents_for_pre" + ".csv").drop(["Unnamed: 0"],1)

        return all_ind, parents, parents_pre
    
    def non_domination_sort(self, all_ind, quantity, phase):
        samples = copy.deepcopy(all_ind)
        samples.reset_index(inplace=True,drop=True)
        print("len(samples):",len(samples))
        
        with open("temp.txt","w") as f:
            f.write(str(self.count)+"\n")
            f.write(str(len(self.opt_object))+"\n")
            f.write(phase+"\n")
            f.write(str(quantity)+"\n")
            f.write(".//"+self.save_location.replace("\\", "//"))
            f.close()
        self.other_function.write_many("data_to_c", samples[self.opt_object], self.count)

        os.system(self.C_file+" /wait")
        f = open('temp1.txt')
        int_f = int(f.read())
        f.close()
        samples_fresh = pd.read_csv(self.save_location + str(self.count)+"_class_present_"+phase+"_"+str(int_f) + ".csv").drop(["Unnamed: 0"],1)

        samples[["dominated_num","class"]] = 0
        samples[["dominated_num","class"]] = samples_fresh[["dominated_num","class"]]
        self.other_function.write_many("class_present_"+phase+"_"+str(int_f)+"_all",samples, self.count)
        samples = samples[samples["class"]!=0]

        return samples.drop(["dominated_num"],1)

    def LHS_decision_value(self, number):
        bounds = []
        delta = []
        
        bounds = self.bound
        dimension = len(bounds)
        print("LHSample,x0",bounds)
        point_num = number
        
        for i in bounds:
            # bounds.append([i*98/100,i*102/100])
            delta.append((i[0]-i[1])/point_num)
            
        result = np.empty([point_num, dimension])
        temp = np.empty([point_num])
        d = 1.0 / point_num

        for i in range(dimension):
            for j in range(point_num):
                temp[j] = np.random.uniform(
                    low=j * d, high=(j + 1) * d, size = 1)[0]

            np.random.shuffle(temp)

            for j in range(point_num):
                result[j, i] = temp[j]
        
        print("LHSample result:",result)
        #对样本数据进行拉伸
        b = np.array(bounds)
        lower_bounds = b[:,0]
        upper_bounds = b[:,1]
        # print(lower_bounds)
        # print(upper_bounds)
        if np.any(lower_bounds > upper_bounds):
            print('范围出错')
            return None

        #   sample * (upper_bound - lower_bound) + lower_bound
        np.add(np.multiply(result,
                        (upper_bounds - lower_bounds),
                        out=result),
            lower_bounds,
            out=result)
        
        LHS = pd.DataFrame(result)
        LHS.columns = self.hidden_name
        return LHS    

    def judge_end(self, samples_fresh):
        self.count += 1
        if self.count>self.end_num :
            print("the best sample is:\n",samples_fresh)
            self.write_many("the_best", samples_fresh, self.count)
            return False
        else:
            return True
    
    def get_congestion(self, samples_choosing, sample_undetermined):
        samples = np.array(copy.deepcopy(samples_choosing))
        full_samples = np.full((len(sample_undetermined),len(samples[:,0]),len(samples[0,:])), samples)
    
        min_column = np.min(full_samples, axis=1)
        max_column = np.max(full_samples, axis=1)
    
        full_samples = full_samples.swapaxes(0,1)
        full_samples = full_samples - np.array(sample_undetermined[self.opt_object])
        full_samples = full_samples / (max_column-min_column)
        full_samples = np.abs(full_samples)
        full_samples = full_samples.swapaxes(0,1)
    
        a = np.array(self.congestion)
        B = np.matmul(full_samples,a)+np.full((len(sample_undetermined),len(samples[:,0])), 1)
    
        full_samples_ = full_samples**2
        relative_distence = np.sum(full_samples_, axis=2)
        relative_distence = relative_distence**0.5

        return np.sum(B/relative_distence, axis=1)    
    
    def accessibility(self, samples_choosing, quantity):
        choosed_num = 0
        samples_parents = pd.DataFrame()
        if quantity<=len(self.opt_object):
            samples_choosing = samples_choosing.loc[samples_choosing["flag_simulated_before"]=="No"]
            samples_parents = samples_choosing.sort_values(by = self.opt_object[0], ascending = True).head(quantity)
        else:
            for i in range(len(self.opt_object)):
                samples_choosing.reset_index(inplace=True,drop=True)
                min_index = np.argmin(np.array(samples_choosing[self.opt_object[i]]))
                samples_parents = samples_parents.append(samples_choosing.loc[min_index])
                samples_choosing = samples_choosing.drop([min_index])

            samples_choosing.reset_index(inplace=True,drop=True)
            
            samples_parents = samples_parents.append(samples_choosing.loc[samples_choosing["flag_simulated_before"]=="Yes"])
            samples_choosing = samples_choosing.drop(list(np.where(np.array(samples_choosing["flag_simulated_before"]=="Yes")==True)[0]))#.drop(samples_choosing["flag_simulated_before"]==True)
            choosed_num = len(samples_parents.loc[samples_parents["flag_simulated_before"]=="No"])
            self.choose_count = 0
            print(time.ctime(), "get_parents_predict quantity=",quantity)
            while choosed_num<quantity:
                print(time.ctime(), "get_parents_predict choosed_num=",choosed_num)
                self.choose_count += 1
                samples_choosing["score"] = 0
                samples_choosing.reset_index(inplace=True,drop=True)
                samples_choosing["score"] = self.get_congestion(samples_parents[self.opt_object], samples_choosing)
                
                min_index = np.argmin(np.array(samples_choosing["score"]))
                samples_parents = samples_parents.append(samples_choosing.loc[min_index])
                samples_choosing = samples_choosing.drop([min_index])

                choosed_num += 1
                samples_parents = samples_parents.drop(["score"],1)
            samples_parents = samples_parents.loc[samples_parents["flag_simulated_before"]=="No"]
        return samples_parents

    def get_fitness_function(self, samples):
        samples = copy.deepcopy(samples)
        samples.reset_index(inplace=True,drop=True)

        samples["distance_2ed_2_1st"] = abs(samples["Mode1_frequency"]-samples["Mode2_frequency"])
        
        max_distance = max(samples["distance_to_resonant_frequency"])
        min_distance = min(samples["distance_to_resonant_frequency"])
        
        samples["CV"] = samples["distance_to_resonant_frequency"]/max_distance
        samples.loc[samples["distance_to_resonant_frequency"]<self.constraint_threshold, "CV"] = 0

        temp = (samples["distance_to_resonant_frequency"]>self.constraint_threshold)
        temp_ = (samples["distance_to_resonant_frequency"]<=self.constraint_threshold)
        temp1 = temp*samples["distance_to_resonant_frequency"]
        max_dis = max(temp1)
        
        temp3 = (math.exp(4)/(math.exp(4)+self.count/self.predict_time*self.k))**4
        
        for i in range(len(self.opt_object)):
            if self.trend[i] == 1:
                temp2 = samples[self.opt_object_from_output[i]]*(temp*temp3+temp_*1)
            else:
                temp2 = (max(samples[self.opt_object_from_output[i]])-samples[self.opt_object_from_output[i]])*(temp*temp3+temp_*1)
            max_temp = max(temp2)
            min_temp = min(temp2)
            samples[self.opt_object[i]] = 1-((temp2-min_temp)/(max_temp-min_temp)-temp1/max_dis*(1-temp3))
        
        return samples

    def select_parents(self, ind_sort, quantity, phase):
        self.other_function.write_log("     select_parents "+str(quantity)+" samples selected in "+phase)
        print("ind_sort：",ind_sort)
        ind_sort = copy.deepcopy(ind_sort)
        
        samples_parents = pd.DataFrame()
        class_num = 1
        
        if phase=="Feasiable" and len(ind_sort)<=quantity:
            samples_parents = ind_sort
        else:
            while len(samples_parents)<quantity:
                samples_choosing = ind_sort[ind_sort["class"]==class_num]
            
                if class_num<=10:
                    print("quantity",quantity,"len(samples_parents)=",len(samples_parents), "len(samples_choosing)=", len(samples_choosing), 'len(samples_choosing.loc[samples_choosing["flag_simulated_before"]==False])=', len(samples_choosing.loc[samples_choosing["flag_simulated_before"]=="No"]))
                    self.other_function.write_many("select_parents_class_num_"+str(class_num), samples_choosing, self.count)
                if (len(samples_choosing.loc[samples_choosing["flag_simulated_before"]=="No"])+len(samples_parents))<=quantity:
                    samples_parents = samples_parents.append(samples_choosing.loc[samples_choosing["flag_simulated_before"]=="No"])
                else:
                    samples_choosed = self.accessibility(samples_choosing, quantity-len(samples_parents))
                    samples_parents = samples_parents.append(samples_choosed)
                class_num += 1
                samples_parents.reset_index(inplace=True,drop=True)
        return samples_parents.drop(["class"],1)

    def crossover(self, samples_fresh, quantity, phase):
        samples_fresh = copy.deepcopy(samples_fresh)
        samples_fresh.reset_index(inplace=True,drop=True)
        self.other_function.write_log("     crossover number is: "+str(quantity)+" "+phase)

        random_vector = np.random.rand(quantity, len(self.hidden_name))*0.8+0.1
        father_ = []
        mather_ = []
        for i in range(quantity):
            father = 0
            mother = 0
            while father==mother:
                father = random.randint(0, len(samples_fresh)-1)
                mother = random.randint(0, len(samples_fresh)-1)
            father_.append(father)
            mather_.append(mother)
        son = pd.DataFrame(random_vector*np.array(samples_fresh.loc[father_,self.hidden_name]) + (1-random_vector)*np.array(samples_fresh.loc[mather_,self.hidden_name]) ,columns = self.hidden_name) 
        
        random_list = []
        for i in range(len(self.hidden_name)):
            random_list = random_list+["inh_ran_"+str(i)]
        son[random_list] = random_vector
        son["number_papa"] = father_
        son["number_mama"] = mather_
        
        if phase=="ToPredict":
            son["phase"] = "PeC"
        elif phase=="ToSimulate":
            son["phase"] = "Cos"
        
        son = self.simulation.reflect_parameter(son)
        son = self.get_prediction(son)
        son = self.other_function.get_post_processing(son)
        self.other_function.write_many("crossover_"+phase, son, self.count)
        
        return son.drop(random_list+["number_papa","number_mama"],1)

    def LHS(self, LHS_predict_num, phase):
        LHS_samples = self.LHS_decision_value(LHS_predict_num)

        LHS_samples = self.simulation.reflect_parameter(LHS_samples)

        LHS_samples["phase"] = "PeL"
        
        LHS_samples = self.simulation.reflect_parameter(LHS_samples)
        LHS_samples = self.get_prediction(LHS_samples)
        LHS_samples = self.other_function.get_post_processing(LHS_samples)
        self.other_function.write_many("LHS_"+phase, LHS_samples, self.count)

        return LHS_samples
    
    def mutation(self, samples_fresh, quantity, phase):
        samples_fresh = copy.deepcopy(samples_fresh)

        Rt_left = self.Rt_value[0]
        samples = pd.DataFrame(columns = self.hidden_name)

        if phase=="ToPredict":
            samples_fresh.reset_index(inplace=True,drop=True)
            for i in range(len(samples_fresh)):
                if  samples_fresh.loc[i,"distance_to_resonant_frequency"]>self.constraint_threshold  and (samples_fresh.loc[i,"Mode1_frequency"]-self.resonant_frequency)/2.5525+samples_fresh.loc[i,"random1"]>0:
                    samples_fresh.loc[i,"random1"] = (samples_fresh.loc[i,"Mode1_frequency"]-self.resonant_frequency)/2.5525+samples_fresh.loc[i,"random1"]
                    samples.loc[len(samples),self.hidden_name] = samples_fresh.loc[i,self.hidden_name]
            samples["phase"] = "PeM"
        elif phase=="ToSimulate":
            samples_fresh = samples_fresh.sort_values(by = "distance_to_resonant_frequency", ascending = False)
            samples_fresh.reset_index(inplace=True,drop=True)
            mutation_num = 0
            sample_num = 0
            while mutation_num<quantity and sample_num<len(samples_fresh):
                print(sample_num, mutation_num<quantity, samples_fresh.loc[sample_num,"flag_mutated"]=="No", samples_fresh.loc[sample_num,"distance_to_resonant_frequency"]>self.constraint_threshold, (samples_fresh.loc[sample_num,"Mode1_frequency"]-self.resonant_frequency)/2.5525+samples_fresh.loc[sample_num,"random1"]>0)
                if samples_fresh.loc[sample_num,"flag_mutated"]=="No" and samples_fresh.loc[sample_num,"distance_to_resonant_frequency"]>self.constraint_threshold  and (samples_fresh.loc[sample_num,"Mode1_frequency"]-self.resonant_frequency)/2.5525+samples_fresh.loc[sample_num,"random1"]>0:
                    samples_fresh.loc[sample_num,"random1"] = (samples_fresh.loc[sample_num,"Mode1_frequency"]-self.resonant_frequency)/2.5525+samples_fresh.loc[sample_num,"random1"]
                    samples.loc[len(samples),self.hidden_name] = samples_fresh.loc[sample_num,self.hidden_name]
                    mutation_num += 1
                sample_num += 1
            samples["phase"] = "Mut"
        
        if len(samples)>0:
            samples = self.simulation.reflect_parameter(samples)
            samples = self.get_prediction(samples)
            samples = self.other_function.get_post_processing(samples)
            self.other_function.write_many("mutation_"+phase, samples, self.count)

        return samples

    def judge_end(self, samples_fresh):
        self.count += 1
        if self.count>self.end_num :#samples_fresh[0].all < 0.001:
            print("the best sample is:\n",samples_fresh)
            self.other_function.write_many("the_best", samples_fresh, self.count)
            return False
        else:
            return True

    def get_prediction(self, dec_val_for_pre):
        dec_val_for_pre = copy.deepcopy(dec_val_for_pre)
        output = self.nn_object.predict(self.nn_model, np.array(dec_val_for_pre[self.input_name]))
        dec_val_for_pre[self.output_name] = output
        return dec_val_for_pre

    def redistribute_counts_of_phases(self, samples_parents):
        samples_parents = copy.deepcopy(samples_parents)
        samples_parents.reset_index(inplace=True,drop=True)
        prediction_num = 0
        mutation_num = 0
        crossover_num = 0
        for i in range(len(samples_parents)):
            if str(100000+self.count)[1:]+"_Pe" in samples_parents.loc[i,"simulation_name"]:
                prediction_num += 1
            elif str(100000+self.count)[1:]+"_Mu" in samples_parents.loc[i,"simulation_name"]:
                crossover_num += 1
            elif str(100000+self.count)[1:]+"_Co" in samples_parents.loc[i,"simulation_name"]:
                mutation_num += 1
        
        if prediction_num+mutation_num+crossover_num ==0:
            self.crossover_num = int(self.simulation_num*0.33)
            self.mutation_num = int(self.simulation_num*0.33)
            self.predict_num = self.simulation_num - self.mutation_num - self.crossover_num
            self.other_function.write_log("----inheritance_real_num is: "+str(self.crossover_num))
            self.other_function.write_log("----variate_real_num is: "+str(self.mutation_num))
            self.other_function.write_log("----choose_predict_num is: "+str(self.predict_num))
            
        else: 
            pre_expected = prediction_num*self.simulation_num/(prediction_num+mutation_num+crossover_num)#8,0,0
            mut_expected = mutation_num*self.simulation_num/(prediction_num+mutation_num+crossover_num)
            cro_expected = crossover_num*self.simulation_num/(prediction_num+mutation_num+crossover_num)
        
            temp = len(np.array(samples_parents["flag_mutated"])[(np.array(samples_parents["flag_mutated"])=="No") * (np.array(samples_parents["CV"])!=0)])
            if mut_expected>temp:#,2,
                pre_expected += (mut_expected - temp)/2#24,2,24
                cro_expected += (mut_expected - temp)/2
                mut_expected = temp
            
            cro_expected = round(cro_expected)
            mut_expected = round(mut_expected)
            pre_expected = self.simulation_num - mut_expected - cro_expected
            
            if mut_expected<5:
                if cro_expected<pre_expected:#3,5,0
                    pre_expected -= (5-mut_expected)
                else:
                    cro_expected -= (5-mut_expected)
                mut_expected = 5
            if pre_expected<5:
                if cro_expected<mut_expected:#5,3,0
                    mut_expected -= (5-pre_expected)
                else:
                    cro_expected -= (5-pre_expected)
                pre_expected = 5
            if cro_expected<5:
                if mut_expected<pre_expected:
                    pre_expected -= (5-cro_expected)
                else:
                    mut_expected -= (5-cro_expected)
                cro_expected = 5
            
            self.crossover_num = round(cro_expected)
            self.mutation_num = round(mut_expected)
            self.predict_num = round(self.simulation_num - mut_expected - cro_expected)
            self.other_function.write_log("----inheritance_real_num is: "+str(self.crossover_num))
            self.other_function.write_log("----variate_real_num is: "+str(self.mutation_num))
            self.other_function.write_log("----choose_predict_num is: "+str(self.predict_num))
    
    def judge_if_mutated(self, all_ind, mutation):
        all_ind = copy.deepcopy(all_ind)
        if len(mutation)!=0:
            all_ind.reset_index(inplace=True,drop=True)
            mutation.reset_index(inplace=True,drop=True)
            for i in range(len(all_ind)):
                for j in range(len(mutation)):
                    if  all_ind.loc[i,"random4"] == mutation.loc[j,"random4"]:
                        all_ind.loc[i,"flag_mutated"] = "Yes"
        return all_ind

    def start(self):
        start_time = time.time()
        
        if self.user == "monitor":
            all_ind, parents, parents_for_pre = self.init()#samples_new_gen_predict每代最后赋予
        
            # self.other_function.write_many("temp1", all_ind, self.count)
            while self.judge_end(parents):
                if self.which_phase[0]==0 or self.which_phase[2]=="predict":
                    if self.which_phase[0]==1 and self.which_phase[2]=="predict":
                        predict_time = 5 - self.which_phase[1]%5
                        self.which_phase[0] = 0
                    else:
                        predict_time = self.predict_time
                    
                    for i in range(predict_time-1):
                        # if self.which_phase[0]==0 or (self.which_phase[0]!=0 and self.which_phase[1]%5==i):
                        # self.which_phase[0] = 0
                        dec_val_for_pre = pd.DataFrame()
                        
                        predict_crossovered = self.crossover(parents_for_pre, self.crossover_predict_num, "ToPredict")#inheritance_predict
                        predict_LHS = self.LHS(self.LHS_predict_num, "ToPredict")#eda
                        predict_mutated = self.mutation(predict_LHS.append(predict_crossovered), self.mutation_predict_num, "ToPredict")#variate_predict
                        #predict_mutated = self.LHS(self.mutation_predict_num, "ToPredict")#
            
                        dec_val_for_pre = dec_val_for_pre.append(predict_crossovered)
                        dec_val_for_pre = dec_val_for_pre.append(predict_LHS)
                        dec_val_for_pre = dec_val_for_pre.append(predict_mutated)
                    
                        ind_pre = self.get_fitness_function(dec_val_for_pre)
                        ind_pre["flag_simulated_before"] = "No"
                        self.other_function.write_many("get_fitness_function_Predicted", ind_pre, self.count)
                        
                        ind_sort = self.non_domination_sort(ind_pre, 50, "Predicted")
                        parents_for_pre = ind_sort.loc[ind_sort["class"]==1]
                        self.other_function.write_many("parents_for_pre", parents_for_pre, self.count)
                        self.count += 1

                    dec_val_for_pre = pd.DataFrame()
                    predict_crossovered = self.crossover(parents_for_pre, self.crossover_predict_num, "ToPredict")#inheritance_predict
                    predict_LHS = self.LHS(self.LHS_predict_num, "ToPredict")#eda
                    predict_mutated = self.mutation(predict_LHS.append(predict_crossovered), self.mutation_predict_num, "ToPredict")#variate_predict
                    #predict_mutated =  self.LHS(self.mutation_predict_num, "ToPredict")#
                    
                    dec_val_for_pre = dec_val_for_pre.append(predict_crossovered)
                    dec_val_for_pre = dec_val_for_pre.append(predict_LHS)
                    dec_val_for_pre = dec_val_for_pre.append(predict_mutated)
            
                    ind_pre = dec_val_for_pre
                    ind_pre["flag_simulated_before"] = "No"
                    all_ind_prediction = self.get_prediction(all_ind)
                    # self.other_function.write_many("temp2", all_ind, self.count)
                    all_ind_prediction = self.other_function.get_post_processing(all_ind_prediction)
                    all_ind_prediction["flag_simulated_before"] = "Yes"
            
                    dec_val_for_pre = ind_pre.append(all_ind_prediction)
                    ind_pre = self.get_fitness_function(dec_val_for_pre)
                    self.other_function.write_many("get_fitness_function_Predicted", ind_pre, self.count)
            
                    ind_sort = self.non_domination_sort(ind_pre, self.predict_num, "Predicted")
                    ind_1 = self.select_parents(ind_sort, self.predict_num, "Predicted")
                    self.other_function.write_many("ind_1", ind_1, self.count)
                
                    mutation = self.mutation(parents, self.mutation_num, "ToSimulate")
                    crossover = self.crossover(parents, self.crossover_num+self.mutation_num-len(mutation), "ToSimulate")
                else:
                    # self.count += self.predict_time-1
                    ind_1 = pd.read_csv(self.save_location + str(self.count) + "_ind_1.csv").drop(["Unnamed: 0"],1)
                    mutation = pd.read_csv(self.save_location + str(self.count) + "_mutation_ToSimulate.csv").drop(["Unnamed: 0"],1)
                    crossover = pd.read_csv(self.save_location + str(self.count) + "_crossover_ToSimulate.csv").drop(["Unnamed: 0"],1)
            
                ind_2 = mutation
                ind_2 = ind_2.append(crossover)
            
                ind_2 = self.get_prediction(ind_2)
                ind_to_simulate = ind_1.append(ind_2)
                self.other_function.write_many("prepare_to_simulate", ind_to_simulate, self.count)
            
                if self.which_phase[0]==0 or (self.which_phase[0]!=0 and self.which_phase[2]=="simulate"):
                    ind_to_simulate = self.other_function.get_post_processing(ind_to_simulate)
                    ind_simulated = self.simulation.simulate(ind_to_simulate, self.which_phase, self.simulation_flag, self.count, "master")
                    self.other_function.write_many("simulated", ind_simulated, self.count)
                    self.which_phase[0]=0
                else:
                    ind_simulated = pd.read_csv(self.save_location + str(self.count) + "_simulated.csv").drop(["Unnamed: 0"],1)
                ind_simulated["flag_mutated"] = "No"
            
            
                all_ind = self.judge_if_mutated(all_ind, mutation)
                all_ind = all_ind.append(ind_simulated)
                all_ind.reset_index(inplace=True,drop=True)            
            
                all_ind = self.other_function.get_post_processing(all_ind)
                all_ind = self.get_fitness_function(all_ind)
                self.other_function.write_many("all_ind", all_ind, self.count)

                all_ind["flag_simulated_before"] = "No"
                if len(all_ind.loc[all_ind["CV"]==0])!=0:
                    ind_sort_feasiable = self.non_domination_sort(all_ind.loc[all_ind["CV"]==0], self.parents_feasiable_num, "Feasiable")
                    all_ind_feasiable = self.select_parents(ind_sort_feasiable, max(self.parents_feasiable_num, int(len(ind_sort_feasiable[ind_sort_feasiable["class"]==1])*0.9)), "Feasiable")
                else:
                    all_ind_feasiable = pd.DataFrame()
                ind_sort_infeasiable = self.non_domination_sort(all_ind.loc[all_ind["CV"]!=0], self.parents_infeasiable_num+self.parents_feasiable_num-len(all_ind_feasiable), "InFeasiable")
                all_ind_infeasiable = self.select_parents(ind_sort_infeasiable, max(self.parents_infeasiable_num+self.parents_feasiable_num-len(all_ind_feasiable), int(len(ind_sort_infeasiable[ind_sort_infeasiable["class"]==1])*0.9)), "InFeasiable")
            
                parents = copy.deepcopy(all_ind_feasiable)
                parents = parents.append(all_ind_infeasiable)
                self.other_function.write_many("parents", parents, self.count)
                parents_for_pre = parents
            
                self.nn_object.test(self.nn_model, parents, self.count)
                self.nn_model = self.nn_object.train(self.nn_model, all_ind, self.count)
                self.other_function.write_log(str(self.count)+" nn has trained")
            
                self.redistribute_counts_of_phases(parents)
                self.which_phase[0] = 0
        
        elif self.user == "slave":
            start_time = time.time()
            while(1):
                files = os.listdir(self.save_location)
                file_result = [val for val in files if "Withresult" in val and ".csv" in val]
                file_noresult = [val for val in files if not "Withresult" in val and ".csv" in val]
                if len(file_noresult)==len(file_result):
                    time.sleep(2)
                else:
                
                    for one in file_result:
                        title = one.replace('_Withresult','')
                        file_noresult.remove(title)
                    self.other_function.write_log("----"+file_noresult[0]+" had been found")
                    samples_ = pd.read_csv(self.save_location+file_noresult[0])
                    self.count = int(float(file_noresult[0][0:6]))
                    ind_simulated = self.simulation.simulate(samples_ , self.which_phase, self.simulation_flag, self.count, "slave")
                    ind_simulated.to_csv(self.save_location+file_noresult[0].replace('.csv','_Withresult.csv'))
                    self.other_function.write_log("----"+file_noresult[0].replace('.csv','_Withresult')+" had been created")

        end_time = time.time()
        print(start_time-end_time)
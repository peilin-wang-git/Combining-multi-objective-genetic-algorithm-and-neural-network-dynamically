"""produce LHS sample"""
"""修改内容
calculate_output
start中得到基准坐标的值删了
__init__中调用删了
get_sorte判断条件变化
"""
import os
import sys
import shutil
import re
# import result
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
# import cstmanager
# import yfunction
import time
import random
from decimal import Decimal
import csv
import seaborn as sns
import pandas as pd
# import find_variated

class myAlg01(object):
    def __init__(self, location, relative_location, input_name, bound, variated_num):

        #读写
        self.location = location#"D:\\wangpeilin\\cst_new_210115\\my_subject_kbann\\"#'D:\\wangpeilin\\cst_new\\my subject\\'
        self.relative_location = relative_location

        self.input_name = input_name
        self.dimension_input = len(self.input_name)
        self.bound = bound

        self.count = 0
        self.variated_num = variated_num#less than 1/2*self.generation_num
        
        #parzen
        self.parzen_list = []
        self.hn = 1/self.variated_num#初始窗大小为：取值范围/初始样本数

    def LHSample(self):
        bounds = []
        delta = []
        
        bounds = self.bound
        dimension = len(bounds)
        print("LHSample,x0",bounds)
        point_num = (self.variated_num)*self.count
        
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
        return result

    def normal_parzen_list(self, parzen_list):
        # print(np.array(parzen_list).shape, np.array(self.bound)[:,0].shape, np.array(self.bound)[:,1].shape)
        normaled_list = (np.array(parzen_list)-np.array(self.bound)[:,0])/(np.array(self.bound)[:,1]-np.array(self.bound)[:,0])
        normaled_list = pd.DataFrame(normaled_list)
        normaled_list.columns = self.input_name
        # print(normaled_list)

        return normaled_list
        
    def get_k(self,x):
        x_square = np.square(x)
        x_sum = np.sum(x_square)
        x_sqrt = np.sqrt(x_sum)
        return (math.exp(-(x_sqrt)/2))/(np.sqrt(2*math.pi))
    
    def get_parzen_p(self,x):
        normaled_list = self.normal_parzen_list(self.parzen_list)
        x = (x-np.array(self.bound)[:,0])/(np.array(self.bound)[:,1]-np.array(self.bound)[:,0])
        self.hn = self.hn/2
        
        sum_k = 0
        line_num = normaled_list.shape[0]
        column_num = normaled_list.shape[1]
        normaled_list = np.array(normaled_list).reshape(line_num, column_num)
        for i in range(normaled_list.shape[0]):
            k = self.get_k((x-normaled_list[i,:])/self.hn)
            # print("i=",i,",k=",k)
            sum_k += k
        
        p = sum_k/((line_num)*(self.hn**self.dimension_input))
        # print(x, self.count, line_num, sum_k, (self.hn**self.dimension_input), line_num,p)
        
        return p
        
    def write_many(self, title, data):
        name = self.location + self.relative_location + "picture\\" + str(self.count) + "_" + title + ".csv"
#         print("name:",name)
        data.to_csv(name,index=True,sep=',')
  
    def start(self, parzen_list, count):
        self.parzen_list = parzen_list
        self.count = count
        
        x = self.LHSample()
        samples = pd.DataFrame(x)
        samples.columns = self.input_name
        samples["p"] = None
        
        for i in range(len(x[:,0])):
            samples.loc[i,"p"] = self.get_parzen_p(x[i])
        
        samples = samples.sort_values(by = "p", ascending = True).head(self.variated_num*2*self.count).sample(frac=1).head(self.variated_num)
        
        return samples.reset_index(drop=True)
 
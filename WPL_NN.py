import numpy as np
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
import keras.initializers as initializers
from keras import optimizers
from keras import regularizers
import copy
import random
from keras.models import load_model
from keras.models import Model

# import tensorflow as tf
import keras.layers as layers
from keras.layers import Input, Dense
import tensorflow.compat.v1 as tf
from keras.utils import plot_model
from keras import backend as K
import pandas as pd
tf.disable_v2_behavior()

class WPL_NN(object):
    def __init__(self,location,input_name,output_name):
        self.location = location#'H:\\test20200831\\cst20210114\\cst_new\\my_subject_kbann\\test_NN\\'#C:\\Users\\asus\\Desktop\\test20200831\\cst_new\\my subject\\'
        # self.relative_location = 'test_NN\\'
        self.accuracy = 100#1e-13
        self.save_number_one_time = 200
        self.alpha = 0.001

        self.input_name = input_name#['Req', 'Leq', 'R3_left', 'R3_right', 'cocave', 'concave_L', 'nose', 'r0_left', 'r0_right', 'r1_left', 'r1_right', 'r2_left', 'r2_right']
        self.output_name = output_name#['frequent','R_divide_Q','shunt_impedance','Q-factor','voltage','total_loss']
        
        #kbann
        self.input_num = len(self.input_name)
        self.output_num = 1
        self.random_pram = 0.5
        self.activity = 'elu'
        
        #process
        self.Y_std = 0
        self.Y_mean = 0
        self.X_mean = 0
        self.X_std = 0
        
        self.count = 0
        
    def write_end(self, data, filename):

        name = self.location + str(self.count) + "_" + filename + ".csv"
        #print("name:",name)
        data.to_csv(name,index=True,sep=',')

    def get_sample(self, samples):
# 生成数据
        X = np.array(samples[self.input_name])
        Y = np.array(samples[self.output_name])

        return Y,X

    def pre_process_train(self,Y_train,X_train):
        
        self.X_mean = X_train.mean(axis=0)
        X_train -= self.X_mean
        self.X_std = X_train.std(axis=0)
        X_train /= self.X_std
        # print(X_test_x.shape, X_mean_x.shape)
        
        self.Y_mean = Y_train.mean(axis=0)
        Y_train -= self.Y_mean
        self.Y_std = Y_train.std(axis=0)
        Y_train /= self.Y_std
        
        Y = []
        for i in range(len(Y_train[0,:])):
            Y.append(Y_train[:,i])

        data = pd.DataFrame([[self.X_mean, self.X_std, self.Y_mean, self.Y_std]], columns=["X_mean","X_std","Y_mean","Y_std"]) 
        self.write_end(data,"log_pre")
        
        return Y,X_train
        
    def pre_process_test(self, X_train):
        
        # self.X_mean = X_train.mean(axis=0)
        X_train -= self.X_mean
        # self.X_std = X_train.std(axis=0)
        X_train /= self.X_std
        # print(X_test_x.shape, X_mean_x.shape)
        

        # data = pd.DataFrame([[self.X_mean, self.X_std, self.Y_mean, self.Y_std]], columns=["X_mean","X_std","Y_mean","Y_std"]) 
        # self.write_end(data,str(self.count)+"_log_pre")
        
        return X_train
    
    def build_kbann(self):
        inputs_x = layers.Input(shape=(self.input_num,))
        # x = layers.Dense(5, activation='sigmoid', kernel_initializer=initializers.RandomUniform(minval=-0.25, maxval=0.25, seed=None))(inputs_x)
        
        # print(initializers.RandomUniform(minval=-0.25, maxval=0.25, seed=None))
        # inputs_y1 = layers.Input(shape=(1,))
        out = []
        for i in range(len(self.output_name)):
            x1 = layers.Dense(5, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(inputs_x)
            #x1_1 = layers.Dense(7, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(x1)
            out1 = Dense(1, activation='linear')(x1)
            out.append(out1)
        
        # # inputs_y2 = layers.Input(shape=(1,))
        # x2 = layers.Dense(7, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(inputs_x)
        # x2_1 = layers.Dense(7, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(x2)
        # # x2_1 = layers.concatenate([inputs_y2, x2])
        # out2 = Dense(1, activation='linear')(x2_1)
        
        # # inputs_y3 = layers.Input(shape=(1,))
        # x3 = layers.Dense(7, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(inputs_x)
        # x3_1 = layers.Dense(7, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(x3)
        # # x3_1 = layers.concatenate([inputs_y3, x3])
        # out3 = Dense(1, activation='linear')(x3_1)
        
        # # inputs_y4 = layers.Input(shape=(1,))
        # x4 = layers.Dense(7, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(inputs_x)
        # x4_1 = layers.Dense(7, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(x4)
        # # x4_1 = layers.concatenate([inputs_y4, x4])
        # out4 = Dense(1, activation='linear')(x4_1)
        
        # # inputs_y5 = layers.Input(shape=(1,))
        # x5 = layers.Dense(7, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(inputs_x)
        # x5_1 = layers.Dense(7, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(x5)

        # # x5_1 = layers.concatenate([inputs_y5, x5])
        # out5 = Dense(1, activation='linear')(x5_1)
        
        # # inputs_y6 = layers.Input(shape=(1,))
        # x6 = layers.Dense(7, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(inputs_x)
        # x6_1 = layers.Dense(7, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(x6)
        # # x6_1 = layers.concatenate([inputs_y6, x6])
        # out6 = Dense(1, activation='linear')(x6_1)

        # x7 = layers.Dense(7, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(inputs_x)
        # x7_1 = layers.Dense(7, activation=self.activity, kernel_initializer=initializers.RandomUniform(minval=-self.random_pram, maxval=self.random_pram, seed=1))(x6)
        # # x6_1 = layers.concatenate([inputs_y6, x6])
        # out7 = Dense(1, activation='linear')(x7_1)       
        
        model = Model(inputs=[inputs_x], outputs=out)
        model.summary()
        # plot_model(model, to_file='model.png')

# 选定loss函数和优化器
        adam = optimizers.Adam(lr=0.001*self.alpha, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # K.manual_variable_initialization(True)
        model.compile(loss='mse', optimizer='adam')
        
        return model
        
    def test_behind_process(self,Y_pred,Y_test,X_test):
        Y_pred = Y_pred*self.Y_std + self.Y_mean
        # Y_test = Y_test*self.Y_std + self.Y_mean
        X_test = X_test*self.X_std + self.X_mean

        # print(Y_pred.shape, Y_test.shape, Y_std, Y_mean, X_test_y.shape)
        print(Y_pred,"\n",Y_test,"\n",X_test.shape)
        self.write_end(pd.DataFrame([(abs(Y_pred - Y_test)/Y_test).sum(axis=0)/(len(Y_pred[:,0]))], columns=self.output_name),"log_behind")
        # self.write_end(['rough-standard compariable :'+str(np.sum((X_test - Y_test)**2)/(len(Y_pred[:,0])*len(Y_pred[0,:])))],"log")
        # self.plot(self.history, Y_pred,Y_test,X_test)
        return Y_pred,Y_test,X_test

    def predict_behind_process(self,Y_test,X_test):
        # Y_pred = Y_pred*self.Y_std + self.Y_mean
        Y_test = Y_test*self.Y_std + self.Y_mean
        X_test = X_test*self.X_std + self.X_mean

        # print(Y_pred.shape, Y_test.shape, Y_std, Y_mean, X_test_y.shape)
        # print(Y_pred,"\n",Y_test,"\n",X_test.shape)
        # self.write_end(pd.DataFrame([(((Y_pred - Y_test)/Y_pred)**2).sum(axis=0)/(len(Y_pred[:,0]))], columns=self.output_name),"log_behind")
        # self.write_end(['rough-standard compariable :'+str(np.sum((X_test - Y_test)**2)/(len(Y_pred[:,0])*len(Y_pred[0,:])))],"log")
        # self.plot(history, Y_pred,Y_test,X_test)
        return Y_test,X_test


    def plot(self, history, Y_pred,Y_test,X_test):
    
        history_dict = history.history
        print(history_dict.keys())
    
# 绘制训练 & 验证的准确率值
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()
        plt.savefig(self.location+str(self.count) + "_" + "relationship of "+"loss"+"_"+"val_loss"+".jpg")

        # for i in range(6):
# 绘制训练 & 验证的损失值
            # plt.figure(i)
            # fig, ax = plt.subplots()

            # print(str(i*2+1))
            # plt.plot(history.history['dense_'+str(i*3+2)+'_loss'])
            # plt.plot(history.history['val_dense_'+str(i*3+2)+'_loss'])
            # plt.title('Model loss')
            # plt.ylabel('Loss')
            # plt.xlabel('Epoch')
            # plt.legend(['Train', 'Test'], loc='upper left')
            # # plt.show()
            # plt.savefig(self.location+str(self.count) + "_" + "relationship of "+"num"+str(i+1)+"loss"+"_"+"val_loss"+".jpg")
            # plt.close('all')  #避免内存泄漏
        x_title = self.input_name
        y_title = self.output_name
        # y_title = ['frequent']
        label = ["label", "predict"]
        area = np.pi * 1**2  # 点面积 
        print(X_test.shape, Y_test.shape, Y_pred.shape)
        for i in range(len(x_title)):
            for j in range(len(y_title)):
                
                plt.figure(i*len(y_title)+j+1)
                fig, ax = plt.subplots()
                fig.subplots_adjust(right=0.8)
                
                if i<(len(x_title)-1):
                    print("i=",i)
                    plt.title("relationship of "+y_title[j]+"_"+x_title[i])
                    plt.xlabel(x_title[i])
                    plt.ylabel(y_title[j])
                    plt.scatter(X_test[:,i], Y_test[:,j], s=area, alpha=0.4, c='#00CED1',label=label[0])
                    plt.scatter(X_test[:,i], Y_pred[:,j], s=area, alpha=0.4, c='#DC143C',label=label[1])
                    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                    plt.savefig(self.location+str(self.count) + "_" + "relationship of "+y_title[j]+"_"+x_title[i]+".jpg")
                    plt.close('all')  #避免内存泄漏
                else:
                    Y_test_index = np.argsort(Y_test[:,j].T)
                    x1 = np.arange(0, len(Y_test), 1)
                    plt.title("relationship of "+y_title[j]+"_"+x_title[i])
                    plt.xlabel(x_title[i])
                    plt.ylabel(y_title[j])
                    for k,Y_index in enumerate(Y_test_index):
                        # print(x1[k], Y_test[Y_index,j])
                        plt.scatter(x1[k], Y_test[Y_index,j], s=area, alpha=0.4, c='#00CED1',label=label[0])
                        plt.scatter(x1[k], Y_pred[Y_index,j], s=area, alpha=0.4, c='#DC143C',label=label[1])
                        for l in range(len(label)):
                            label[l] = "_nolegend_"
                    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
                    plt.savefig(self.location+str(self.count) + "_" + "relationship of "+y_title[j]+"_"+x_title[i]+".jpg")
                    plt.close('all')  #避免内存泄漏
                print("relationship of "+y_title[j]+"_"+x_title[i]+".jpg has saved")
    
    def write_many(self, title, data):
        name = self.location + "picture\\" + str(self.count) + "_" + title + ".csv"
#         print("name:",name)
        data.to_csv(name,index=True,sep=',')
    
    def train(self, model, samples, count):
        self.count = count
        Y_train,X_train = self.get_sample(samples)
        Y_train,X_train = self.pre_process_train(Y_train,X_train)
        # i =1
        # for layer in model.layers:
            # print(layer.get_weights())
        #print("##############################################",Y_train.shape)
        self.history = model.fit([X_train], Y_train,
                        validation_split=0.25, epochs=5000, batch_size=1000, verbose=2)
        model.save(self.location+str(self.count)+'_my_model.h5')
        return model
        
    def predict(self, model, X_test):
        X_test = self.pre_process_test(X_test)
        Y_test = np.array(model.predict([X_test])).reshape(len(self.output_name), -1).T
        Y_test,X_test = self.predict_behind_process(Y_test,X_test)
        return Y_test
    
    def test(self, model, samples, count):
        self.count = count
        samples = copy.deepcopy(samples)
        X_pred = self.pre_process_test(np.array(samples[self.input_name]))
        Y_pred = np.array(model.predict([X_pred])).reshape(len(self.output_name), -1).T
        Y_test = np.array(samples[self.output_name])
        Y_pred,Y_test,X_pred = self.test_behind_process(Y_pred,Y_test,X_pred)
        Y_pred_name = []
        for name in self.output_name:
            Y_pred_name.append("pre_"+name)
        for i in range(len(self.output_name)):
            samples[Y_pred_name[i]] = Y_pred[:,i]
            
        self.write_end(samples, "nn_test")
    
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer,LayerNormalization
import tensorflow as tf
import numpy as np
import copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, GlobalAveragePooling1D
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,RMSprop
import matplotlib.pyplot as plt
# import _thread
# import threading
import time
# from Transformer import Transformer,MultiHeadAttention,PositionalEncoding,Add#,Embedding,PositionalEncoding,Add

class ScaledDotProductAttention(Layer):
    def __init__(self,mode,**kwargs):
        assert mode == "encoder" or mode == "decoder", "The parameter 'mode' can only receive two values, 'encoder' and 'decoder'."
        self.masking_num = -2**32
        self.mode = mode
        super(ScaledDotProductAttention, self).__init__(**kwargs)
    # padding mask
    # 将0值位置置为一个极小的负数，使得softmax时该值接近0
    def padding_mask(self, QK):
        padding = tf.cast(tf.equal(QK,0),tf.float32)
        padding *= self.masking_num
        return QK+padding
    # sequence mask(传说中的下三角)
    def sequence_mask(self,QK):
        # 初始化下三角矩阵
        seq_mask = 1-tf.linalg.band_part(tf.ones_like(QK), -1, 0)
        seq_mask *= self.masking_num
        return QK+seq_mask
    # 输入为qkv三个矩阵和一个mask矩阵
    def call(self, inputs):
        assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
        queries, keys, values = inputs
        # 转换为32位
        if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')
        # Qk计算
        matmul = tf.matmul(queries,keys,transpose_b=True)
        dk = tf.cast(tf.shape(keys)[-1],tf.float32)
        matmul = matmul / tf.sqrt(dk) # QxK后缩放dk**(0.5)
        # mask层,区别encoder和decoder部分
        if self.mode == "encoder":
            matmul = self.padding_mask(matmul)
        else:
            matmul = self.sequence_mask(matmul)
        softmax_out = K.softmax(matmul)  # SoftMax层
        return K.batch_dot(softmax_out, values) # 最后乘V
    def get_config(self):
        config = super().get_config()
        config.update({
            'masking_num': self.masking_num,
            "mode" : self.mode
        })
        return config
    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention(Layer):
    def __init__(self, heads=8,model_dim=512,mode="encoder",trainable=True,**kwargs):
        self.heads = heads
        self.head_dim = 3
        self.mode = mode
        self.model_dim = model_dim
        self.trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)
    # 随机初始化Q K V矩阵权重，在这里所有头都进行训练
    def build(self,input_shape):
        self.weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self.heads * self.head_dim),#列数意味着头数*每个头的列数
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='weights_queries')
        self.weights_keys = self.add_weight(
            shape=(input_shape[0][-1], self.heads * self.head_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='weights_keys')
        self.weights_values = self.add_weight(
            shape=(input_shape[0][-1], self.heads * self.head_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='weights_values')
        self.weights_last_matrix = self.add_weight(
            shape=(self.heads * self.head_dim, self.model_dim),
            initializer='glorot_uniform',
            trainable=self.trainable,
            name='weights_last_matrix')
            
        self.shape= input_shape
        super(MultiHeadAttention, self).build(input_shape)
    def call(self, inputs):
        assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
        # 注意，这里传入的qkv并不是真正的qkv，而是上一层的embedding(3个),之后乘权重才是真正的qkv
        queries, keys, values = inputs
        print("==-==-==-==-==-==-==:",queries.shape)
        print("==-==-==-==-==-==-==:",self.weights_queries.shape)
        # 初始化
        queries_linear = K.dot(queries, self.weights_queries)
        keys_linear = K.dot(keys, self.weights_keys)
        values_linear = K.dot(values, self.weights_values)
        # print("==-==-==-==-==-==-==:",queries_linear.shape)
        # 多头切割
        queries_multi_heads = tf.concat(tf.split(queries_linear, self.heads, axis=2), axis=1)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self.heads, axis=2), axis=1)
        values_multi_heads = tf.concat(tf.split(values_linear, self.heads, axis=2), axis=1)

        att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]
        attention = ScaledDotProductAttention(mode=self.mode)
        att_out = attention(att_inputs)

        outputs = tf.concat(tf.split(att_out, self.heads, axis=1), axis=2)
        outputs = K.dot(outputs,self.weights_last_matrix)
        return outputs
    def get_config(self):
        config = super().get_config()
        config.update({
            'head_dim': self.head_dim,
            'heads': self.heads,
            "mode" : self.mode,
            "trainable" : self.trainable
        })
        return config
    def compute_output_shape(self, input_shape):
        return input_shape

# encoder和decoder都要用到的前向传播
def FeedForwardNetwork(units_dim,model_dim):
    return Sequential([Dense(units_dim, activation='relu'),Dense(model_dim)])

# 接embedding层，位置编码
class PositionalEncoding(Layer):
    def __init__(self,model_dim,**kwargs):
        self.model_dim = model_dim
        super(PositionalEncoding, self).__init__(**kwargs)
    def get_angles(self,pos,i,d_model):
        return pos/(np.power(10000, (2 * (i//2)) / np.float32(d_model)))
    def call(self,embedding):
        # 输入的是embedding，所以embedding的行就是当前这句话
        # embedding.shape[0]=数据量
        # embedding.shape[1]=句子长度
        # embedding.shape[2]=词嵌入维度
        sentence_length=embedding.shape[1]
        positional_encoding = np.zeros(shape=(sentence_length,self.model_dim))
        # 计算sin/cos位置编码(论文里有公式，懒得备注了)
        for pos in range(sentence_length):
            for i in range(self.model_dim):
                positional_encoding[pos, i] = self.get_angles(pos,i,self.model_dim)
        positional_encoding[:, 0::2] = np.sin(positional_encoding[:, 0::2])  # 用于偶数索引2i
        positional_encoding[:, 1::2] = np.cos(positional_encoding[:, 1::2])  # 用于奇数索引2i+1
        return K.cast(positional_encoding, 'float32')
    def get_config(self):
        config = super().get_config()
        config.update({
            'mmodel_dim': self.model_dim,
        })
        return config
    def compute_output_shape(self,input_shape):
        return input_shape


class Add(Layer):
    def __init__(self,**kwargs):
        super(Add, self).__init__(**kwargs)
    # 这里的inputs指embedding+positional encoding
    def call(self, inputs):
        input_a, input_b = inputs
        res = input_a+input_b
        return res
    def compute_output_shape(self, input_shape):
        return input_shape[0]        
        
class WPL_Transformer(object):
    def __init__(self,location, input_name, output_name, 
                maxlen, model_dim, batch_size, epochs,
                dense_num_pre, dense_num, head_num, activation):
        self.input_name = input_name#['Req', 'Leq', 'R3_left', 'R3_right', 'cocave', 'concave_L', 'nose', 'r0_left', 'r0_right', 'r1_left', 'r1_right', 'r2_left', 'r2_right']
        self.output_name = output_name#['frequent','R_divide_Q','shunt_impedance','Q-factor','voltage','total_loss']
        self.location = location
        
        #process
        self.Y_std = 0
        self.Y_mean = 0
        self.X_mean = 0
        self.X_std = 0
        
        self.count = 0
        
        self.maxlen = maxlen #句子的最大长度，输入参数的数量，13
        self.model_dim = model_dim     # 词嵌入的维度
        self.batch_size = batch_size
        self.epochs = epochs#5000
        
        #build
        self.dense_num_pre = dense_num_pre
        self.dense_num = dense_num
        self.head_num = head_num
        self.activation = activation

    def write_end(self, data, filename):
        name = self.location + str(self.count) + "_" + filename + ".csv"
        #print("name:",name)
        data.to_csv(name,index=True,sep=',')

    def get_sample(self, samples):
        # 生成数据
        X = np.array(samples[self.input_name]).astype('float32')
        Y = np.array(samples[self.output_name]).astype('float32')

        return Y,X

    def pre_process_train(self,Y_train,X_train):
        self.X_mean = X_train.mean(axis=0)
        X_train = X_train - self.X_mean
        self.X_std = X_train.std(axis=0)
        X_train = X_train / self.X_std
        # print(X_test_x.shape, X_mean_x.shape)
        
        self.Y_mean = Y_train.mean(axis=0)
        Y_train = Y_train - self.Y_mean
        self.Y_std = Y_train.std(axis=0)
        Y_train = Y_train / self.Y_std
        
        Y_train = Y_train.astype("float32")
        
        Y = []
        for i in range(len(Y_train[0,:])):
            Y.append(Y_train[:,i])

        data = pd.DataFrame([[self.X_mean, self.X_std, self.Y_mean, self.Y_std]], columns=["X_mean","X_std","Y_mean","Y_std"]) 
        self.write_end(data,"log_pre")
        
        X_train_ = np.append(X_train[:,:6], np.zeros((len(X_train[:,0]),1)),axis=1)
        X_train__ = np.append(X_train_, X_train[:,6:],axis=1).reshape(((-1, self.maxlen, self.model_dim)))
        X_train__ = X_train__.astype("float32")
        print("the shape is:",X_train__.shape)
        
        return Y,X_train__

    def pre_process_test(self, X_train):
        X_train = X_train - self.X_mean
        X_train = X_train / self.X_std

        X_train_ = np.append(X_train[:,:6], np.zeros((len(X_train[:,0]),1)),axis=1)
        X_train__ = np.append(X_train_, X_train[:,6:],axis=1).reshape(((-1, self.maxlen, self.model_dim)))
        print("the shape is:",X_train__.shape)
        
        return X_train__
    
    
    def build_kbann(self):
        inputs = Input(shape=(self.maxlen, self.model_dim), name="inputs")
        
        outputs = []
        for i in range(len(self.output_name)):
            encodings = PositionalEncoding(self.model_dim)(inputs)
            encodings = Add()([inputs, encodings])
            x = MultiHeadAttention(heads=self.head_num,model_dim=self.model_dim)([encodings, encodings, encodings])
            x = tf.reduce_mean(x, 1)
            x_1 = Dense(self.dense_num_pre, activation=self.activation)(tf.reshape(inputs,(-1,self.maxlen*self.model_dim),name="Dense_embedding"))
            x = concatenate([x_1, x])
            x = Dense(self.dense_num, activation=self.activation)(x)
            output = Dense(1, activation='linear')(x)
            outputs.append(output)
       
        model = Model(inputs=[inputs], outputs=outputs)
        # model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                        # loss='categorical_crossentropy')
        model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                      loss='mse')
        
        return model
    
    def test_behind_process(self,Y_pred,Y_test,X_test):
        Y_pred = Y_pred*self.Y_std + self.Y_mean
        # Y_test = Y_test*self.Y_std + self.Y_mean
        X_test = copy.deepcopy(X_test)
        X_test = X_test.reshape([-1,len(self.input_name)+1])
        print("X_test.shape:",X_test.shape, X_test[:,:6].shape, X_test[:,7:-1].shape)
        # X_test_ = X_test[:,:6]
        X_test_ = np.append(X_test[:,:6], X_test[:,7:],axis=1)
        print("X_test_.shape:",X_test_.shape)
        X_test_ = X_test_*self.X_std + self.X_mean

        # print(Y_pred.shape, Y_test.shape, Y_std, Y_mean, X_test_y.shape)
        print(Y_pred,"\n",Y_test,"\n",X_test_.shape)
        self.write_end(pd.DataFrame([(abs(Y_pred - Y_test)/Y_test).sum(axis=0)/(len(Y_pred[:,0]))], columns=self.output_name),"log_behind")
        # self.write_end(['rough-standard compariable :'+str(np.sum((X_test - Y_test)**2)/(len(Y_pred[:,0])*len(Y_pred[0,:])))],"log")
        self.plot(self.history, Y_pred,Y_test,X_test_)

        # new_threading_1 = threading.Thread(target=self.plot, args=(self.history, Y_pred,Y_test,X_test_,))  #定义一个新线程，线程的名称，要做什么工作
        # new_threading_1.start()
        # try:
            # print("i*3+2 = ",i*3+2)
            # _thread.start_new_thread(self.plot, (self.history, Y_pred,Y_test,X_test_,) )
        # except:
            # print ("Error: 无法启动线程")
            # return
        
        return Y_pred,Y_test,X_test

    def predict_behind_process(self,Y_test,X_test):
        Y_test = Y_test*self.Y_std + self.Y_mean
        X_test = X_test.reshape([-1,len(self.input_name)+1])
        X_test_ = np.append(X_test[:,:6], X_test[:,7:],axis=1)
        
        X_test_ = X_test_*self.X_std + self.X_mean
        

        return Y_test,X_test_
        
    def plot_loss(self,history,i):
        plt.figure()
        plt.plot(history.history['dense_'+str(i)+'_loss'])
        plt.plot(history.history['val_dense_'+str(i)+'_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()
        plt.savefig(self.location+str(self.count) + "_" + "relationship of "+"num"+str(i)+"loss"+"_"+"val_loss"+".jpg")
        plt.close('all')  #避免内存泄漏
        
    
    def plot_scatter(self,x_title,y_title,X_test,Y_test,Y_pred):
        area = np.pi * 1**2  # 点面积 
        label = ["label", "predict"]
        
        plt.figure()
        plt.title("relationship of "+y_title+"_"+x_title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.scatter(X_test, Y_test, s=area, alpha=0.4, c='#00CED1',label=label[0])
        plt.scatter(X_test, Y_pred, s=area, alpha=0.4, c='#DC143C',label=label[1])
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        plt.savefig(self.location+str(self.count) + "_" + "relationship of "+y_title+"_"+x_title+".jpg")
        plt.close('all')  #避免内存泄漏
        print("relationship of "+y_title+"_"+x_title+".jpg has saved")

    def plot_scatter_sorted(self,x_title,y_title,X_test,Y_test,Y_pred):
        area = np.pi * 1**2  # 点面积 
        label = ["label", "predict"]
        
        plt.figure()
        plt.title("relationship of "+y_title+"_"+x_title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        # for k,Y_index in enumerate(Y_test_index):
        # print(x1[k], Y_test[Y_index,j])
        plt.scatter(X_test, Y_test, s=area, alpha=0.4, c='#00CED1',label=label[0])
        plt.scatter(X_test, Y_pred, s=area, alpha=0.4, c='#DC143C',label=label[1])
        for l in range(len(label)):
            label[l] = "_nolegend_"
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
        plt.savefig(self.location+str(self.count) + "_" + "relationship of "+y_title+"_"+x_title+".jpg")
        plt.close('all')  #避免内存泄漏
        print("relationship of "+y_title+"_"+x_title+".jpg has saved")
    
    def plot(self, history, Y_pred,Y_test,X_test):
        history_dict = history.history
        print(history_dict.keys())
    
# 绘制训练 & 验证的准确率值
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()
        plt.savefig(self.location+str(self.count) + "_" + "relationship of "+"loss"+"_"+"val_loss"+".jpg")
        plt.close('all')  #避免内存泄漏

        for i in range(len(self.output_name)):
            self.plot_loss(history, i*3+2)
# 绘制训练 & 验证的损失值
            # plt.figure(i)
            # try:
                # print("i*3+2 = ",i*3+2)
                # _thread.start_new_thread( self.plot_loss, (history, i*3+2,) )
            # except:
                # print ("Error: 无法启动线程")
                # return
            # self.plot_loss(i*3+2)
        x_title = self.input_name
        y_title = self.output_name
        # y_title = ['frequent']
        print(X_test.shape, Y_test.shape, Y_pred.shape)
        for i in range(len(x_title)):
            for j in range(len(y_title)):
                
                # plt.figure(i*len(y_title)+j+1)
                # fig, ax = plt.subplots()
                # fig.subplots_adjust(right=0.8)
                
                if i<(len(x_title)-1):
                    print("i=",i)
                    self.plot_scatter(x_title[i],y_title[j],X_test[:,i],Y_test[:,j],Y_pred[:,j])
                    # try:
                        # _thread.start_new_thread( self.plot_scatter, (x_title[i],y_title[j],X_test[:,i],Y_test[Y_index,j],) )
                    # except:
                        # print ("Error: 无法启动线程")
                        # # return

                else:
                    Y_test_index = np.argsort(Y_test[:,j].T)
                    x1 = np.arange(0, len(Y_test), 1)
                    Y_test_ = []
                    Y_pred_ = []
                    # X = []
                    for k,Y_index in enumerate(Y_test_index):
                        # X.append(X_test[k])
                        Y_test_.append(Y_test[Y_index,j])
                        Y_pred_.append(Y_pred[Y_index,j])
                    
                    self.plot_scatter_sorted(x_title[i],y_title[j],x1,Y_test_,Y_pred_)
                    # try:
                        # _thread.start_new_thread( self.plot_scatter_sorted, (x_title[i],y_title[j],x1,Y_test,Y_pred,Y_test_index,j,) )
                    # except:
                        # print ("Error: 无法启动线程")
                        # return

    def train(self, model, samples, count):
        self.count = count
        Y_train,X_train = self.get_sample(samples)
        Y_train,X_train = self.pre_process_train(Y_train,X_train)
        # i =1
        # for layer in model.layers:
            # print(layer.get_weights())
        # print(Y_train.T.shape)
        self.history = model.fit([X_train], Y_train,
                        validation_split=0.25, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        model.save(self.location+str(self.count)+'_my_model.h5')
        return model
          
    def predict(self, model, X_test):
        X_test = X_test.astype('float32')
        X_test = self.pre_process_test(X_test)
        X_test = X_test.astype('float32')
        Y_test = np.array(model.predict([X_test])).reshape(len(self.output_name), -1).T
        Y_test,X_test = self.predict_behind_process(Y_test,X_test)
        return Y_test


    def test(self, model, samples, count):
        self.count = count
        samples = copy.deepcopy(samples)
        X_pred = self.pre_process_test(np.array(samples[self.input_name]).astype('float32'))
        X_pred = X_pred.astype('float32')
        Y_pred = np.array(model.predict([X_pred])).reshape(len(self.output_name), -1).T
        Y_test = np.array(samples[self.output_name])
        Y_pred,Y_test,X_pred = self.test_behind_process(Y_pred,Y_test,X_pred)
        Y_pred_name = []
        for name in self.output_name:
            Y_pred_name.append("pre_"+name)
        for i in range(len(self.output_name)):
            samples[Y_pred_name[i]] = Y_pred[:,i]
            
        self.write_end(samples, "nn_test")

# self.nn_object = WPL_NN.WPL_NN(self.location+self.relative_location+"picture\\", self.input_name, self.output_name)
# self.nn_model = self.nn_object.build_kbann()
# input_name = ['Req', 'Leq', 'R3_left', 'R3_right', 'cocave', 'concave_L', 'nose', 'r0_left', 'r0_right', 'r1_left', 'r1_right', 'r2_left', 'r2_right']
# output_name = ['frequent','R_divide_Q','shunt_impedance','Q-factor','voltage','total_loss']

# transformer = WPL_Transformer("C:\\Users\\Administrator\\Desktop\\test_NN\\picture\\", input_name, output_name)
# # my.nn_object = WPL_NN.WPL_NN(my.location+my.relative_location+"picture\\", my.input_name, my.output_name)
# nn_model = transformer.build_kbann()
# print(nn_model.summary())
# # predict_dominatd = pd.read_csv("C:\\Users\\Administrator\\Desktop\\0_init_sample_output" + ".csv").drop(["Unnamed: 0"],1)
# predict_dominatd = pd.read_csv("C:\\Users\\Administrator\\Desktop\\test_NN\\0_init_sample_output" + ".csv").drop(["Unnamed: 0"],1)
# nn_model = transformer.train(nn_model, predict_dominatd, 200)
# predict_dominatd = pd.read_csv("C:\\Users\\Administrator\\Desktop\\test_NN\\0_init_sample_output_old" + ".csv").drop(["Unnamed: 0"],1)
# transformer.test(nn_model, predict_dominatd, 200)

# #- 1.使用MultiHeadAttention训练imdb数据
# vocab_size = 13
# maxlen = 7 #句子的最大长度，输入参数的数量，13
# model_dim = 2     # 词嵌入的维度
# batch_size = 32
# epochs = 1000
# num_layers = 2


# # 读取imdb数据
# # (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=maxlen, num_words=vocab_size)
# y_train = np.array(predict_dominatd["frequent"]/1000).reshape((-1,1))
# x_train = pd.DataFrame(np.ones(len(y_train)), columns = ["add"])
# # print(predict_dominatd[['Req', 'Leq']])
# x_train = x_train.join(predict_dominatd[['Req', 'Leq', 'R3_left', 'R3_right', 'cocave', 'concave_L', 'nose', 'r0_left', 'r0_right', 'r1_left', 'r1_right', 'r2_left', 'r2_right']])
# # x_train = np.array(pd.merge(x_train, ))
# x_train = np.array(x_train).reshape(((len(y_train), 7, 2)))
# print(x_train.shape)
# print(x_train.shape, y_train.shape)

                
# print(model.summary())

# history = model.fit(x_train, y_train,
          # batch_size=batch_size, epochs=epochs, validation_split=0.2)

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
# plt.close('all')  #避免内存泄漏

# # plt.title("relationship of "+y_title[j]+"_"+x_title[i])
# # plt.xlabel(x_title[i])
# # plt.ylabel(y_title[j])
# area = np.pi * 1**2
# y = model.predict(x_train)*1000
# plt.scatter(predict_dominatd["r2_right"], predict_dominatd["frequent"], s=area, alpha=0.4, c='#00CED1',label="test")
# plt.scatter(predict_dominatd["r2_right"], y, s=area, alpha=0.4, c='#DC143C',label="predict")
# plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
# plt.show()
# plt.close('all')  #避免内存泄漏
        
# # print(sum((abs(y - predict_dominatd["frequent"])/y))/(len(y[:,0])))
# # print("y_predict:\n",y)

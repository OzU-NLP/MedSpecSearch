
import numpy as np
import tensorflow as tf
import pandas
import pickle
import time
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix
import itertools
import gensim


import EmbedHelper
import DataLoader



def KL(alpha,outputSize):
    beta=tf.constant(np.ones((1,outputSize)),dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha,axis=1,keep_dims=True)
    S_beta = tf.reduce_sum(beta,axis=1,keep_dims=True)
    lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha),axis=1,keep_dims=True)
    lnB_uni = tf.reduce_sum(tf.lgamma(beta),axis=1,keep_dims=True) - tf.lgamma(S_beta)
    
    dg0 = tf.digamma(S_alpha)
    dg1 = tf.digamma(alpha)
    
    kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keep_dims=True) + lnB + lnB_uni
    return kl

def mse_loss(p, alpha, global_step, annealing_step,outputSize): 
    S = tf.reduce_sum(alpha, axis=1, keep_dims=True) 
    E = alpha - 1
    m = alpha / S
    
    A = tf.reduce_sum((p-m)**2, axis=1, keep_dims=True) 
    B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keep_dims=True) 
    
    annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))
    
    alp = E*(1-p) + 1 
    C =  annealing_coef * KL(alp,outputSize)
    return (A + B) + C

def loss_EDL(p, alpha, global_step, annealing_step,outputSize):
    S = tf.reduce_sum(alpha, axis=1, keep_dims=True) 
    E = alpha - 1

    A = tf.reduce_mean(tf.reduce_sum(p * (tf.digamma(S) - tf.digamma(alpha)),1, keepdims=True))

    annealing_coef = tf.minimum(1.00,tf.cast(global_step/annealing_step,tf.float32))

    alp = E*(1-p) + 1 
    B =  annealing_coef * KL(alp,outputSize)

    return (A + B)


# In[4]:


class PyramidCNNVShort:
    
    hiddenSize = 250
    
    def __init__(self,inputSize,vectorSize,outputSize):
        hiddenSize = self.hiddenSize
        
        self.paperGraph = tf.Graph()
        with self.paperGraph.as_default():

            self.initializer = tf.contrib.layers.variance_scaling_initializer()

            self.nn_inputs = tf.placeholder(tf.string,[None,inputSize])
            self.nn_vector_inputs = tf.placeholder(tf.float32,[None,inputSize,vectorSize])

            self.token_lengths = tf.placeholder(tf.int32,[None])

            self.nn_outputs = tf.placeholder(tf.int32,[None])
            
            self.annealing_step = tf.placeholder(dtype=tf.int32) 
            
            self.uncertaintyRatio = tf.placeholder(dtype=tf.float32)
            
            outputsOht = tf.one_hot(self.nn_outputs,outputSize)
            # print("Output OHT 1 :",outputsOht.shape)

            self.isTraining = tf.placeholder(tf.bool)
            
            # print("Inputs :",self.nn_vector_inputs.shape)

            fullInputs = self.nn_vector_inputs
            fullVectorSize = fullInputs.shape[2]

            # print("Concat :",fullInputs.shape)


            with tf.name_scope("ConvLayers"):

                num_filters = hiddenSize
                cnnInput = tf.expand_dims(fullInputs, -1)

                # print("filter input",cnnInput.shape)


                convouts = []
                conv1 = tf.layers.conv2d(cnnInput,hiddenSize,(3,fullVectorSize),(1,1),padding="valid",activation=None,use_bias=True,name="PreBlock")       
                blockPool = tf.nn.max_pool(conv1,ksize=[1,126,1,1],strides=[1,1,1,1],padding="VALID",name=("Pool-"))
                
                conv2 = tf.layers.conv2d(cnnInput,hiddenSize,(4,fullVectorSize),(1,1),padding="valid",activation=None,use_bias=True,name="PreBlock2")
                blockPool2 = tf.nn.max_pool(conv2,ksize=[1,125,1,1],strides=[1,1,1,1],padding="VALID",name=("Pool-"))
                
                conv3 = tf.layers.conv2d(cnnInput,hiddenSize,(5,fullVectorSize),(1,1),padding="valid",activation=None,use_bias=True,name="PreBlock3")
                blockPool3 = tf.nn.max_pool(conv3,ksize=[1,124,1,1],strides=[1,1,1,1],padding="VALID",name=("Pool-"))
                
                convouts.append(blockPool)
                convouts.append(blockPool2)
                convouts.append(blockPool3)
                
                # print("-")
                # print("conv1 shape :",conv1.shape)
                # print("blockPool shape : ",blockPool.shape)
                #
                # print("-")
                # print("conv2 shape :",conv2.shape)
                # print("blockPool2 shape : ",blockPool2.shape)
                #
                # print("-")
                # print("conv2 shape :",conv3.shape)
                # print("blockPool2 shape : ",blockPool3.shape)
                num_filters_total = 3*hiddenSize

                # print("convouts : ",convouts)
                h_pool = tf.concat(convouts,3)
                # print("hpool : ",h_pool.shape)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

                # print("h_pool_flat : ",h_pool_flat.shape)

                h_drop = tf.layers.dropout(h_pool_flat,0.5,training=self.isTraining)




            with tf.name_scope("fully-connected"):
                fc1_neurons = 150
                fc2_neurons = 100
                fc3_neurons = 25

                fc1 = tf.layers.dense(h_drop,activation=tf.nn.leaky_relu,name="fc1",use_bias=True,kernel_initializer=self.initializer,units=fc1_neurons)
                fcD1 = tf.layers.dropout(fc1,0.5,training=self.isTraining)


            with tf.name_scope("output"):
                scores = tf.layers.dense(fcD1,activation=None,name="logits",use_bias=True,kernel_initializer=self.initializer,units=outputSize)
                self.evidence = tf.exp(scores/1000)
            
            with tf.name_scope("evaluation"):
                global_step = tf.Variable(0,trainable=False)
                init_learn_rate = 0.001
                decay_learn_rate = tf.train.exponential_decay(init_learn_rate,global_step,100,0.90,staircase=True)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                optimizer = tf.train.AdamOptimizer()
                
            with tf.name_scope("accuracy"):
                # print(outputsOht.shape)
                # print(tf.argmax(outputsOht, 1).shape)
                self.predictions = tf.argmax(scores, 1, name="predictions")
                self.truths = tf.argmax(outputsOht, 1)
                self.correct_predictions = tf.equal(self.predictions, self.truths)
                # print(self.correct_predictions.shape)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
                match = tf.reshape(tf.cast(tf.equal(self.predictions, self.truths), tf.float32),(-1,1))
                
                
                
            with tf.name_scope("uncertainty"):
                self.alpha = self.evidence +1

                self.uncertainty = outputSize / tf.reduce_sum(self.alpha, axis=1, keep_dims=True) #uncertainty

                self.prob = self.alpha/tf.reduce_sum(self.alpha, 1, keepdims=True) 

                total_evidence = tf.reduce_sum(self.evidence ,1, keepdims=True) 
                mean_ev = tf.reduce_mean(total_evidence)
                self.mean_ev_succ = tf.reduce_sum(tf.reduce_sum(self.evidence ,1, keepdims=True)*match) / tf.reduce_sum(match+1e-20)
                self.mean_ev_fail = tf.reduce_sum(tf.reduce_sum(self.evidence ,1, keepdims=True)*(1-match)) / (tf.reduce_sum(tf.abs(1-match))+1e-20) 
            
            
                flatUncertainty = tf.reshape(self.uncertainty,shape=[-1,1])
                flatCP = tf.reshape(self.correct_predictions,shape=[-1,1])
                
                # print("Flat Uncertainty : ",flatUncertainty.shape)
                # print("Flat CP :",flatCP.shape)
                zeros = tf.cast(tf.zeros_like(flatUncertainty),dtype=tf.bool)
                ones = tf.cast(tf.ones_like(flatUncertainty),dtype=tf.bool)
                ucAccuraciesBool = tf.where(tf.less_equal(flatUncertainty,self.uncertaintyRatio),ones,zeros)
                
                # print("Mask : ",ucAccuraciesBool.shape)
                self.ucAccuracies = tf.boolean_mask(flatCP,ucAccuraciesBool)
                
                self.ucAccuracy = tf.reduce_mean(tf.cast(self.ucAccuracies,"float"))
                
                self.dataRatio = tf.shape(self.ucAccuracies)[0] / tf.shape(flatCP)[0]
            
            with tf.name_scope("loss"):
                self.loss = tf.reduce_mean(loss_EDL(outputsOht, self.alpha, global_step, self.annealing_step,outputSize))
                regLoss = tf.add_n([tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()])       
                regularazationCoef = 0.0000005   
                self.loss += regLoss*regularazationCoef

            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss,global_step=global_step,colocate_gradients_with_ops=True)


def __init__(self,classDict,embedHandler,dataHandler,inputLength,nnModel):
        self.classDict = classDict
        self.dataHandler = dataHandler
        self.embedHandler = embedHandler
        self.inputLength = inputLength
        self.nnModel = nnModel
        
        self.sess = tf.Session(graph=nnModel.paperGraph)
        with nnModel.paperGraph.as_default():
            with self.sess.as_default():
                saver = tf.train.Saver()
                saver.restore(self.sess, "/tmp/model.ckpt")
# In[5]:


class Predicter:
    EmbeddingsFolderPath = "Embeddings"
    ModelsFolderPath = "NNModels"
    SystemDataFolderPath = "SystemData"
    
    def __init__(self):
        with open(Predicter.SystemDataFolderPath+'/fold0classDict.pkl', 'rb') as f:
            self.classDict =  pickle.load(f)
        self.dataHandler = DataLoader.DataHandler

        self.inputLength = 128

        self.loadModels(Predicter.ModelsFolderPath)

        self.nnModel = None
        self.glove = None
        self.htWord2Vec = None
        self.pubmed = None
        self.googleNews = None

        self.initializeEmbeddings()
    
    def loadModels(self,folderPath):
        self.googleNewsNN,self.googleNewsSession = self.getModel(folderPath+"/GoogleNews/model.ckpt",300)
        self.htWord2VecNN,self.htWord2VecSession = self.getModel(folderPath+"/HTW2V/model.ckpt",300)
        self.gloveNN,self.gloveSession = self.getModel(folderPath+"/Glove-Confidence/model.ckpt",300)
        self.pubmedNN,self.pubmedSession = self.getModel(folderPath+"/Pubmed/model.ckpt",200)

        self.nnModel = self.gloveNN
        self.sess = self.gloveSession

    def getModel(self,path,vectorSize):
        with tf.device("/cpu:0"):
            nnModel = PyramidCNNVShort(128, vectorSize, 12)
            sess = tf.Session(graph=nnModel.paperGraph)
            with nnModel.paperGraph.as_default():
                with sess.as_default():
                    saver = tf.train.Saver()
                    saver.restore(sess, path)
        return nnModel,sess


    def initializeEmbeddings(self):
        self.htWord2Vec = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[3], False, 300,
                                                       Predicter.EmbeddingsFolderPath)
        self.googleNews = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[2], False, 300,
                                                       Predicter.EmbeddingsFolderPath)
        self.glove = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[5], False, 300, Predicter.EmbeddingsFolderPath)
        self.pubmed = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[4], False, 200, Predicter.EmbeddingsFolderPath)
        self.embedHandler = self.glove
                
    
        
    def predict(self,ip):
        cleanData,length = self.dataHandler.inputPreprocessor([ip],self.inputLength)
        vecData = self.embedHandler.vectorizeBatch(cleanData)
        
        feedDict = {self.nnModel.nn_vector_inputs:vecData,self.nnModel.isTraining:False}
        uncertainty,prob = self.sess.run([self.nnModel.uncertainty,self.nnModel.prob],feed_dict=feedDict)
        return uncertainty,prob,length
    
    def predictCOP(self,ip):
        uncertainty,prob,length = self.predict(ip)
        uncertainty = uncertainty[0]
        prob = prob[0]
        
        reverseClassDict = {value:key for key,value in self.classDict.items()}
        outputClassCount = len(reverseClassDict)
        
        probDict = {reverseClassDict[i]:prob[i] for i in np.arange(outputClassCount)}
        probMatrix = []
        for i in range(len(prob)):
            probMatrix.append([reverseClassDict[i], prob[i]])

        probMatrix = sorted(probMatrix, key=lambda x: (x[1]), reverse=True)

        maxIdx = np.argmax(prob)
        resultDict = {
            "Uncertainty":uncertainty[0],
            "Confidence":1-uncertainty[0],
            "Prediction":reverseClassDict[maxIdx],
            "PredictionProb":prob[maxIdx],
            "Probabilities":probMatrix,
            "Length":length
        }
        
        return resultDict
    
    def predictModel(self,ip,embeddingType):
        reverseClassDict = {value:key for key,value in self.classDict.items()}

        self.checkEmbeddings(embeddingType)

        if(embeddingType == "Glove"):
            self.embedHandler = self.glove
            self.nnModel = self.gloveNN
            self.sess = self.gloveSession

        elif(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[3]):
            self.embedHandler = self.htWord2Vec
            self.nnModel = self.htWord2VecNN
            self.sess = self.htWord2VecSession

        elif(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[2]):
            self.embedHandler = self.googleNews
            self.nnModel = self.googleNewsNN
            self.sess = self.googleNewsSession

        elif(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[4]):
            self.embedHandler = self.pubmed
            self.nnModel = self.pubmedNN
            self.sess = self.pubmedSession
        else:
            raise Exception("Embedding Given DOESNT exist")


        predDict = self.predictCOP(ip)
        predProb = predDict["Probabilities"]
        predDict["Top3"] = predProb[0:3]

        return predDict

    def checkEmbeddings(self,embeddingType):
        if(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[5]):
            if (self.glove is None):
                self.glove = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[5], False, 300, Predicter.EmbeddingsFolderPath)

        elif(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[3]):
            if(self.htWord2Vec is None):
                self.htWord2Vec = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[3], False, 300,
                                                               Predicter.EmbeddingsFolderPath)
        elif(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[2]):
            if(self.googleNews is None):
                self.googleNews = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[2], False, 300,
                                                               Predicter.EmbeddingsFolderPath)
        elif(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[4]):
            if(self.pubmed is None):
                self.pubmed = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[4], False, 200,
                                                               Predicter.EmbeddingsFolderPath)


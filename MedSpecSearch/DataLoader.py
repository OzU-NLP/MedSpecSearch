
# coding: utf-8

# In[1]:


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


# In[2]:


class DataHandler:
    
    def __init__():
        pass
    
    @staticmethod
    def getUniqueClassMapDict(classList):
        uniques = np.unique(classList)
        count = np.arange(uniques.size)
        listDict = np.hstack((uniques.reshape(-1,1),count.reshape(-1,1)))
        uniqueDict = {elem[0]:int(elem[1]) for elem in listDict}
        return uniqueDict

    
    @staticmethod
    def cleanTextData(textList):
        cleanTextList = []
        for text in textList:
            text = text.lower()

            cleanText = ""

            for word in text.split(" "):
                cleanWord = ""
                for char in word:
                    if(ord(char)>96 and ord(char)<123):
                        cleanWord += char

                cleanText += cleanWord + " "

            cleanTextList += [cleanText.strip()]

        return cleanTextList
    
    @staticmethod
    def idxListToidxDict(idxList):
        idxDict = {}

        for i in range(len(idxList)):
            idxDict[idxList[i]] = i

        return idxDict
    
    @staticmethod
    def calculateLongestSentence(sentenceList):
        longestSentence = 25

        for elem in sentenceList:
            stcLength = len(elem.split(" "))
            if(stcLength>longestSentence):
                longestSentence = stcLength

        return longestSentence
    
    @staticmethod
    def fillSentenceArray(sentence,fillSize,maxLength=1500):
        fillCount = fillSize - len(sentence)

        for i in range(fillCount):
            sentence += [np.zeros(vectorSize)]

        return sentence[0:maxLength]
    
    @staticmethod
    def fillWordListArray(sentence,maxLength):
        fillCount = maxLength - len(sentence)

        for i in range(fillCount):
            sentence += ["[None]"]


        return sentence[0:maxLength]
    
    @staticmethod
    def textIntoWordList(textList,maxLength,embedModel=None):
        embedList = []
        lengthList = []

        for sentence in textList:
            embeddedSentence = []

            for word in sentence.split(" "):
                if (embedModel is not None):
                    if word in embedModel:
                        embedding = word
                        embeddedSentence += [embedding]
                else:
                    embedding = word
                    embeddedSentence += [embedding]
                    
            sentenceLength = len(embeddedSentence)
            embeddedSentence = DataHandler.fillWordListArray(embeddedSentence,maxLength)   
            embedList += [embeddedSentence]
            lengthList += [sentenceLength]

        return embedList,sentenceLength
    
    @staticmethod
    def masterPreprocessor(data,maxLength,shuffle=False,classDict=None):
        if(classDict is None):
            classDict = DataHandler.getUniqueClassMapDict(data[:,1])
        if(shuffle == True):
            np.random.shuffle(data)

        convertedClasses = np.array([classDict[elem] for elem in data[:,1]])
        print("Outputs converted to numerical forms")
        cleanedTextData = DataHandler.cleanTextData(data[:,0])
        print("Input text claned")
        wordList,lengthList = DataHandler.textIntoWordList(cleanedTextData,maxLength)
        print("Input text split into tokens and all inputs padded to maximum length")

        return np.array(wordList),np.array(convertedClasses),classDict
    
    @staticmethod
    def inputPreprocessor(data,maxLength):
        cleanedTextData = DataHandler.cleanTextData(data)
        wordList,lengthList = DataHandler.textIntoWordList(cleanedTextData,maxLength)
        return wordList,lengthList
        
    @staticmethod
    def batchIterator(data,target,batchSize):
        dataSize = data.shape[0]

        while(True):
            randomIdx = np.random.randint(dataSize,size=batchSize)

            yield np.take(data,randomIdx,axis=0),np.take(target,randomIdx)


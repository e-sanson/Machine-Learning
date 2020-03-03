import string
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def stem_tokenizer(text):
    porter_stemmer=PorterStemmer()
    return [porter_stemmer.stem(token) for token in word_tokenize(text.lower())]

def stop_words():
    punctuation=list(string.punctuation)
    stop=stopwords.words("english")+punctuation
    return set(stem_tokenizer(" ".join(stop)))

def document_labels(directory):
    doc_labels=[]
    for author in os.listdir(directory):
        if not(os.path.isdir(directory+"/"+author)):
            print(f"Skiping '{author}' because is not a directory")
        for filename in os.listdir(directory+"/"+author):
            # we save absolute path to file
            filename=os.path.abspath(directory+"/"+author+"/"+filename)
            doc_labels.append([filename,author])
    data=pd.DataFrame(doc_labels,columns=["filename","label"])
    return data

def load_glove_model(glove_filename):
    print("Loading Glove Model")
    f = open(glove_filename, encoding='utf-8')
    model = {}
    D=None
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
        if not D:
            D=len(embedding)
    print("Done.",len(model)," words loaded!")
    model=defaultdict(lambda : np.zeros(D),model)
    return model

def build_word_embedding(index2word,word_vectors):
    V=len(index2word)+1
    D=len(next(iter(word_vectors.values()))) # get the dimension of the fest value (vector) on the word_vector dictionary
    embedding=np.empty((V,D),dtype=np.float)
    embedding[0]=np.zeros(D,dtype=np.float) # first word (index=-1) will be for unknown tokens
    for idx,word in enumerate(index2word):
        if word not in word_vectors:
            v=embedding[0]
        else:
            v=word_vectors[word]
        embedding[idx+1]=v
    return embedding

# map documents to integer, on integer for word.
# unseen words on transform (valuation corpus) will be map to "unseen" index -1
class DocumentEncoder:
    def __init__(self,vectorizer):
        self.vectorizer=vectorizer
    def encode_documents(self,documents):
        texts=[]
        for document in documents:
            words=self.analyzer(document)        
            encoded_text=[self.vocabulary[word] for word in words]
            texts.append(np.array(encoded_text))
        return np.array(texts)
    def fit_transform(self,documents):
        doc_counts=self.vectorizer.fit_transform(documents)
        self.counts=doc_counts.sum(axis=0) # how many times each work appears in corpus
        self.counts=np.asarray(self.counts).ravel() # sparse matrix sums return as matrix but we want a np.array object
        self.analyzer=self.vectorizer.build_analyzer()
        self.vocabulary = defaultdict(lambda: -1,self.vectorizer.vocabulary_)
        self.feature_names=self.vectorizer.get_feature_names()
        return self.encode_documents(documents)
    def transform(self,documents):
        return self.encode_documents(documents)
    def inverse_transform(self,doc):
        # this function is just for demo pourposes right now
        # we use slow but clear method, we would vectorize if we cared more about performance 
        words=[]
        for index in doc:
            if index<0:
                words.append("<UNKNOWN>")
            else:
                words.append(self.feature_names[index])
        return words
   

# and encoder that crops and pads (at the begining) documents to a fix length <L>,
# and split them into <W>   fixed  size windows so that
#
#                N variable sized documents --> N x W  x (L/W) features
#
# <W> must divide <L> exactly or an exception is raised
#
class PaddingEncoder:
    def __init__(self,encoder,L,W):
        self.encoder=encoder
        self.L=L
        self.W=W
        if (self.L % self.W !=0):
            raise Exception(f"Number of windows <W> must divide document length <L> exactly, but {W} does not divide {L}")
        self.K= self.L//self.W
    def pad_documents(self,documents):
        X=np.empty((len(documents),self.W,self.K),dtype=int)
        for idx,document in enumerate(documents):
            cropped=document[:self.L] # Crop at maximum document size
            pad_size=self.L-len(cropped)
            padded=np.pad(cropped,((pad_size,0)),mode="constant",constant_values=-1)
            #print(padded.shape)
            reshaped=padded.reshape(self.W,self.K)
            X[idx]=reshaped
        return X
    def fit_transform(self,documents):
        encoded=self.encoder.fit_transform(documents)
        return self.pad_documents(encoded)
    def transform(self,documents):
        encoded=self.encoder.transform(documents)
        return self.pad_documents(encoded)

# Embed words using a word embedding and average  word vectors over document length windows using idf weights 
class AveragingEmbedder:
    def __init__(self,encoder,word_vectors,vectorizer):
        self.encoder=encoder
        self.word_vectors=word_vectors
        self.vectorizer=vectorizer
    def embed(self,indexes):
        embedding=self.word_embedding[indexes+1]
        weights=self.idf[indexes+1][:,:,:,np.newaxis]
        #print(embedding.shape,weights.shape,end=" ")
        we=np.mean(embedding*weights,axis=-2)
        #print(we.shape)
        return we
    def build_idf(self,idf): # must handle unknown works at index 0
        return np.insert(idf,0,0.0)     
    def fit_transform(self,documents):
        indexes=self.encoder.fit_transform(documents)
        index2word=self.vectorizer.get_feature_names()
        self.idf=self.build_idf(self.vectorizer.idf_)
        self.word_embedding=build_word_embedding(index2word,self.word_vectors)
        #return self.embed(indexes)
        return self.transform(documents)
    def transform(self,documents):
        indexes=self.encoder.transform(documents)
        return self.embed(indexes)
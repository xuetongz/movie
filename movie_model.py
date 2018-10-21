import numpy as np 
import pandas as pd 

import random

seed =0

random.seed(seed)
np.random.seed(seed)

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input,Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,GlobalAveragePooling1D,concatenate,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional,BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold


train=pd.read_csv('......train.tsv',sep='\t')
test=pd.read_csv('......test.tsv',sep='\t')


#data preprocessing
vocabulary_size=12000
maxlen = 50
train = train.sample(frac=1).reset_index(drop=True)
#train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())
#test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())
X = train['Phrase']
test_X = test['Phrase']
Y = to_categorical(train['Sentiment'].values)

tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(list(X))
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=maxlen)
test_X = tokenizer.texts_to_sequences(test_X)
test_X = pad_sequences(test_X, maxlen=maxlen)

################GloVe embedding 
file_name='glove.6B.100d.txt'####GloVe 100 dimension 

def getEmbedding(file_name,volcabulary_size,dim_embedding,tok):
    embedding_index= dict()
    f=open(file_name)
    for line in f:
        values=line.split()
        word=values[0]
        coefs=np.asarray(values[1:],dtype='float32')
        embedding_index[word]=coefs
    f.close()
    embedding_matrix=np.zeros((volcabulary_size, dim_embedding))
    for word,index in tok.word_index.items():
        if index > volcabulary_size-1:
            break
        else:
            embedding_vector=embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index]=embedding_vector
    return embedding_matrix

embedding_matrix=getEmbedding(file_name,vocabulary_size,100,tokenizer)




################
    
def modelBuilder1(kernel1,kernel2,dr):
    inp=Input(shape=(maxlen,))
    embedding=Embedding(vocabulary_size,100,input_length=maxlen,weights=[embedding_matrix],trainable=False)(inp)
    x=SpatialDropout1D(dr,input_shape=(100,))(embedding)
    
    x_lstm= Bidirectional(LSTM(128,return_sequences=True))(x)
    x_conv1=Conv1D(32,kernel_size=kernel1,padding='same',activation='relu')(x_lstm)
    max_x1=GlobalMaxPooling1D()(x_conv1)
    avg_x1=GlobalAveragePooling1D()(x_conv1)
    x_conv2=Conv1D(32,kernel_size=kernel2,padding='same',activation='relu')(x_lstm)
    max_x2=GlobalMaxPooling1D()(x_conv2)
    avg_x2=GlobalAveragePooling1D()(x_conv2)
    
    x_gru= Bidirectional(GRU(128,return_sequences=True))(x)
    x_conv3=Conv1D(32,kernel_size=kernel1,padding='same',activation='relu')(x_gru)
    max_x3=GlobalMaxPooling1D()(x_conv1)
    avg_x3=GlobalAveragePooling1D()(x_conv1)
    x_conv4=Conv1D(32,kernel_size=kernel2,padding='same',activation='relu')(x_gru)
    max_x4=GlobalMaxPooling1D()(x_conv4)
    avg_x4=GlobalAveragePooling1D()(x_conv4)
    
    x_all=concatenate([max_x1,max_x2,max_x3,max_x4,avg_x1,avg_x2,avg_x3,avg_x4])
    x_all=BatchNormalization()(x_all)
    x_all=Dropout(dr)(x_all)
    x_all=Dense(units = 64 , activation='relu')(x_all)
    x_all=BatchNormalization()(x_all)
    x_all=Dropout(dr)(x_all)
    x_all=Dense(units = 32 , activation='relu')(x_all)
    x_all=Dropout(dr)(x_all)
    x_all=Dense(5,activation='softmax')(x_all)
    model=Model(inputs =inp,outputs=x_all)
    return model
    
    
def modelBuilder2(k1,k2,dr):
    inp=Input(shape=(maxlen,))
    embedding=Embedding(vocabulary_size,100,input_length=maxlen,weights=[embedding_matrix],trainable=False)(inp)
    x=Dropout(dr,input_shape=(100,))(embedding)
    x_lstm= Bidirectional(LSTM(128,return_sequences=True))(x)
    
    x_conv1=Conv1D(32,kernel_size=k1,padding='same',activation='relu')(x_lstm)
    max_x1=GlobalMaxPooling1D()(x_conv1)
    avg_x1=GlobalAveragePooling1D()(x_conv1)
    
    x_conv2=Conv1D(32,kernel_size=k2,padding='same',activation='relu')(x_lstm)
    max_x2=GlobalMaxPooling1D()(x_conv2)
    avg_x2=GlobalAveragePooling1D()(x_conv2)
    
    x_all=concatenate([max_x1,max_x2,avg_x1,avg_x2])
    x_all=BatchNormalization()(x_all)
    x_all=Dropout(dr)(x_all)
    x_all=Dense(units = 64 , activation='relu')(x_all)
    x_all=Dropout(dr)(x_all)
    x_all=Dense(5,activation='softmax')(x_all)
    
    model=Model(inputs =inp,outputs=x_all)
    return model
    
    
    
def kfoldprediction(folds,weight_output,model_builder,k1,k2,dr,X,Y,epoch,batch,pretrained,weight_input):
    skf=StratifiedKFold(n_splits=folds,shuffle=True)
    pred=np.zeros((test_X.shape[0],5))
    file_path = weight_output
    for i,(train_index, test_index ) in enumerate (skf.split(X,train['Sentiment'])):
        print('fold'+str(i+1))
        X_train,X_valid = X[train_index],X[test_index]
        y_train,y_valid = Y[train_index],Y[test_index]
        new_model=model_builder(k1,k2)
        if pretrained==True:
            new_model.load_weights(weight_input)
        model_callback=EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0, mode='auto')
        model_check_point = ModelCheckpoint(weight_output, monitor = "val_loss", verbose = 1,save_best_only = True, mode = "min")
        new_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        new_model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=epoch, batch_size=batch,verbose=1,callbacks = [model_check_point, model_callback])
        pred+=new_model.predict(test_X,batch_size=1024,verbose=0)
    return pred
    
f=5
epoch =15
batch = 128
weight_file_path = "best_modellstm.hdf5"

pred1=kfoldprediction(folds=f,weight_output=weight_file_path,model_builder=modelBuilder1,k1=2,k2=3,dr=0.2,X=X,Y=Y,epoch=epoch,batch=batch,pretrained=False,weight_input=input_file)
pred2=kfoldprediction(folds=f,weight_output=weight_file_path,model_builder=modelBuilder1,k1=3,k2=4,dr=0.3,X=X,Y=Y,epoch=epoch,batch=batch,pretrained=False,weight_input=input_file)
pred3=kfoldprediction(folds=f,weight_output=weight_file_path,model_builder=modelBuilder1,k1=1,k2=2,dr=0.4,X=X,Y=Y,epoch=epoch,batch=batch,pretrained=False,weight_input=input_file)
pred4=kfoldprediction(folds=f,weight_output=weight_file_path,model_builder=modelBuilder2,k1=2,k2=3,dr=0.3,X=X,Y=Y,epoch=epoch,batch=batch,pretrained=False,weight_input=input_file)
pred5=kfoldprediction(folds=f,weight_output=weight_file_path,model_builder=modelBuilder2,k1=2,k2=4,dr=0.4,X=X,Y=Y,epoch=epoch,batch=batch,pretrained=False,weight_input=input_file)

final_pred=pred1+pred2+pred3+pred4+pred5
test['Sentiment']= final_pred.argmax(axis=1)
test[["PhraseId", "Sentiment"]].to_csv("submission.csv", index=False)


##########################################





















import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
from keras.preprocessing import sequence
import nltk

import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split 



DATA_PATH = './data/'
EMBEDDING_DIR = './data/'

max_features=20000
embed_size=300
max_seq_len=150

label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def read_input_data():
    train_df = pd.read_csv('./data' + '/train.csv', sep=',', header=0)
    test_df = pd.read_csv('./data' + '/test.csv', sep=',', header=0)
    
    train_df['comment_text']=train_df.comment_text.astype(str)
    train_df['comment_text'].fillna(' ', inplace=True)
    test_df['comment_text'].fillna(' ', inplace=True)

    print("num train: ", train_df.shape[0])
    print("num test: ", test_df.shape[0])
    
    return (train_df,test_df)

            
def tokenize_data(train_df,test_df):
    raw_docs_train = train_df['comment_text'].tolist()
    raw_docs_test = test_df['comment_text'].tolist() 
    num_classes = len(label_names)
    
    regex_tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english')) 
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

    print("pre-processing train data...")
    processed_docs_train = []
    for doc in tqdm(raw_docs_train):
        try: 
            tokens = regex_tokenizer.tokenize(doc)
            filtered = [word for word in tokens if word not in stop_words]
            processed_docs_train.append(" ".join(filtered))
        except Exception as e:
            print (e)
            #print (doc)
    #end for

    processed_docs_test = []
    for doc in tqdm(raw_docs_test):
        try:
            tokens = regex_tokenizer.tokenize(doc)
            filtered = [word for word in tokens if word not in stop_words]
            processed_docs_test.append(" ".join(filtered))
        except:
            print (doc)
    #end for

    print("tokenizing input data...")
    tokenizer = Tokenizer(num_words=max_features, lower=True, char_level=False)
    tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  #leaky
    word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
    word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
    word_index = tokenizer.word_index
    print("dictionary size: ", len(word_index))

    #pad sequences
    word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
    word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)
    
    return (word_seq_train,word_seq_test,word_index)

def prepare_embeddings(file,word_index):
    #load embeddings
    EMBEDDING_FILE = file

    embeddings_index = {}
    with open(EMBEDDING_FILE, encoding='utf8') as f:
        for line in tqdm(f):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    #embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix

def init_model(embedding_matrix):
    
    #model parameters
    num_filters = 64 
    embed_dim = 200 
    weight_decay = 1e-4

    num_classes = len(label_names)
    #CNN architecture
    print("training CNN ...")
    model = Sequential()
    model.add(Embedding(max_features, embed_size,
              weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.8))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(num_classes, activation='sigmoid'))  #multi-label (k-hot encoding)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

def train_model(model,epochs,batch_size,seq_train,y_train):
    #define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=4, verbose=1)
    callbacks_list = [early_stopping]
    
    X_train, X_val, Y_train, Y_val = train_test_split(seq_train, y_train, test_size=0.2)
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), verbose=1)
    print (hist)
    return hist

def create_submission_file(test_df,y_test,filename):
    #create a submission
    submission_df = pd.DataFrame(columns=['id'] + label_names)
    submission_df['id'] = test_df['id'].values 
    submission_df.shape
    submission_df[label_names] = y_test 
    submission_df.to_csv(filename, index=False)
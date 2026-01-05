from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pickle
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from keras.models import Sequential
from keras.layers import Dense, Flatten, Bidirectional, LSTM, RepeatVector, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model, Sequential, load_model
from keras.callbacks import ModelCheckpoint
import os
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score
import tensorflow as tf
import time
from DCGAN import DCGAN
import sys
sys.path.append('./tools/')
from utils import save_images, save_source
from data_generator import ImageDataGenerator


global unamee, dcgan_model, sess
global X_train, X_test, y_train, y_test, tfidf_vectorizer, sc
global model
global filename
global X, Y, dataset

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
generator = ImageDataGenerator(batch_size = 32, height = 128, width = 128, z_dim = 256, scale_size=(128, 128), shuffle=False, mode='train')
val_generator = ImageDataGenerator(batch_size = 32, height = 128, width = 128, z_dim = 256, scale_size=(128, 128), shuffle=False, mode='test')

#function to calculate SSIM between original and fake image
def getSSIM(original, fake):
    orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    fakes = cv2.cvtColor(fake, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(orig, fakes, data_range = fakes.max() - fakes.min())
    return ssim_value

with tf.Graph().as_default():
    sess = tf.Session(config=config)
    dcgan_model = DCGAN(sess=sess, lr = 0.001, keep_prob = 1., model_num = None, batch_size = 32, age_loss_weight = None, gan_loss_weight = None,
                        fea_loss_weight = None, tv_loss_weight = None)
    dcgan_model.imgs = tf.placeholder(tf.float32, [32, 128, 128, 3])
    dcgan_model.true_label_features_128 = tf.placeholder(tf.float32, [32, 128, 128, 5])
    dcgan_model.ge_samples = dcgan_model.generate_images(dcgan_model.imgs, dcgan_model.true_label_features_128, stable_bn=False, mode='train')
    dcgan_model.get_vars()
    dcgan_model.saver = tf.train.Saver(dcgan_model.save_g_vars)
    # Start running operations on the Graph.
    sess.run(tf.global_variables_initializer())
    if dcgan_model.load(dcgan_model.saver, 'dcgan', 399999):
        print("DCGAN model successfully loaded")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

X = np.load("model/X.npy")
Y = np.load("model/Y.npy")
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
X = tfidf_vectorizer.fit_transform(X).toarray()
data = X
sc = StandardScaler()
X = sc.fit_transform(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
Y = np.reshape(Y, (Y.shape[0], (Y.shape[1] * Y.shape[2] * Y.shape[3])))
Y = Y.astype('float32')
Y = Y/255
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

model = Sequential()
#creating gan model
model.add(Conv2D(32, (1, 1), activation='relu', input_shape=(X.shape[1], X.shape[2], X.shape[3])))
#max layer to collect relevant features from gan layer
model.add(MaxPooling2D((1, 1)))
#adding another layer
model.add(Conv2D(64, (1, 1), activation='relu'))
model.add(MaxPooling2D((1, 1)))
model.add(Conv2D(128, (1, 1), activation='relu'))
model.add(MaxPooling2D((1, 1)))
model.add(Flatten())
model.add(RepeatVector(2))
#adding spatial attention model
model.add(Bidirectional(LSTM(128, activation = 'relu')))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(Y.shape[1], activation='sigmoid'))
# Compile and train the model.
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = model.fit(X_train, y_train, batch_size = 16, epochs = 15, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    model.load_weights("model/cnn_weights.hdf5")
f = open('model/cnn_history.pckl', 'rb')
data = pickle.load(f)
f.close()
print(data['accuracy'])
accuracy_value = data['accuracy'][9]

def generateDetectFake(filename):
    global sess, dcgan_model
    arr = ['original','fake1', 'fake2', 'fake3', 'fake4']
    img_list = []
    source = val_generator.load_imgs(filename, 128)
    train_imgs = generator.load_train_imgs("CelebDataset/train", 128)
    temp = np.reshape(source, (1, 128, 128, 3))
    save_source(temp, [1, 1], "output/"+arr[0]+".jpg")
    images = np.concatenate((temp, train_imgs), axis=0)
    for j in range(1, generator.n_classes):
        true_label_fea = generator.label_features_128[j]
        dict = {
            dcgan_model.imgs: images,
            dcgan_model.true_label_features_128: true_label_fea,
            }
        samples = sess.run(dcgan_model.ge_samples, feed_dict=dict)
        image = np.reshape(samples[0, :, :, :], (1, 128, 128, 3))
        save_images(image, [1, 1], "output/"+arr[j]+".jpg")
    orig = cv2.imread("output/"+arr[0]+".jpg")
    orig = cv2.resize(orig, (250, 250))
    for i in range(len(arr)):
        img = cv2.imread("output/"+arr[i]+".jpg")
        img = cv2.resize(img, (250,250))
        cv2.putText(img, arr[i], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        img_list.append(img)            
    images = cv2.hconcat(img_list)
    return images

def HumanFacesAction(request):
    if request.method == 'POST':
        global sess, dcgan_model
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        if os.path.exists('ImageApp/static/'+fname):
            os.remove('ImageApp/static/'+fname)
        with open('ImageApp/static/'+fname, "wb") as file:
            file.write(myfile)
        file.close()
        img = generateDetectFake('ImageApp/static/'+fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':"DCGAN Generated Fake Images", 'img': img_b64}
        return render(request, 'ViewResult.html', context)    

def HumanFaces(request):
    if request.method == 'GET':
        return render(request, 'HumanFaces.html', {})

def PredictPerformance(request):
    if request.method == 'GET':
       return render(request, 'PredictPerformance.html', {})

def TexttoImageAction(request):
    if request.method == 'POST':
        global tfidf_vectorizer, sc, model
        text_data = request.POST.get('t1', False)
        answer = text_data.lower().strip()
        model = load_model("model/cnn_weights.hdf5")
        data = answer
        data = cleanText(data)
        test = tfidf_vectorizer.transform([data]).toarray()
        test = sc.transform(test)
        test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
        predict = model.predict(test)
        predict = predict[0]
        predict = np.reshape(predict, (128, 128, 3))
        predict = cv2.resize(predict, (300, 300))
        img = cv2.cvtColor(predict, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':"Text = "+text_data, 'img': img_b64}
        return render(request, 'ViewResult.html', context)    

def TrainModel(request):
    if request.method == 'GET':
        global accuracy_value
        f = open("model/cnn_history.pckl", 'rb')
        train_values = pickle.load(f)
        f.close()
        acc_value = train_values['accuracy']
        loss_value = train_values['loss']
        plt.figure(figsize=(6,4))
        plt.grid(True)
        plt.xlabel('EPOCH')
        plt.ylabel('Accuracy')
        plt.plot(acc_value, 'ro-', color = 'green')
        plt.plot(loss_value, 'ro-', color = 'blue')
        plt.legend(['Accuracy', 'Loss'], loc='upper left')
        plt.title('DCGAN Training Accuracy & Loss Graph')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':'<font size="3" color="blue">DCGAN Accuracy : '+str(accuracy_value)+"</font>", 'img': img_b64}
        return render(request, 'ViewResult.html', context)        

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def AdminLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        if username == "admin" and password == "admin":
            context= {'data':'welcome '+username}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'Invalid login details'}
            return render(request, 'AdminLogin.html', context)          


def TexttoImage(request):
    if request.method == 'GET':
       return render(request, 'TexttoImage.html', {})
        

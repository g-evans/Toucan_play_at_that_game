#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural network CNN walkthrough

This code will recreate the ImageNet classification problem and several of the best solutions. It is focused on the CNN structure
so will recreate the performance on a single classification, rather than 1,000.

Architectures:
    AlexNet
    VGG
    GoogleNet
    residual networks - Bottleneck blocks
                        ResNeXt - parallel
                        
Flow follows this direction:
    1. initialise funcitons and environment areas
    2. Create df of all images and save to file
    3. Use this to build the training/validation and test sets and save to file
    4. Read data sets from file, build model and then train it - Export analysis
    5. Step 4 is repeated for each of the architectures listed above

"""

#%% Environment initiation

import pathlib
import tarfile
import time
import numpy as np 
import pandas as pd
import matplotlib.image as mpimg
from bs4 import BeautifulSoup
import re
import os
import glob 
import PIL
import PIL.Image
import os
import time
import glob
import math
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, \
                                    BatchNormalization, Dropout, concatenate, \
                                    GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorboard
import tensorflow_datasets as tfds
from matplotlib import patches
from tensorflow import keras
import matplotlib.pyplot as plt
import random
import sys
import seaborn as sns
import cv2
from scikitplot.metrics import plot_roc
from skimage.util.shape import view_as_windows
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score,accuracy_score,classification_report,precision_recall_curve
from sklearn.model_selection import train_test_split


def clean_classification_name(name):
    if name.find(',')==-1:
        return name 
    else:
        return name[:name.find(',')]
    
def get_classes_dict():
    classes = pd.read_csv('/Volumes/GE_2022/Image_analysis/Data/imagenet-object-localization-challenge/LOC_synset_mapping.txt',sep='\t',names=['text'])
    classes['id'] = classes.apply(lambda row: row['text'][:9],axis=1)
    classes['class'] = classes.apply(lambda row: clean_classification_name(row['text'][10:]),axis=1)
    classes = classes.set_index('id')
    classes = classes.drop(columns=['text'])
    classes = classes.append(pd.DataFrame(['Not found'],columns=classes.columns,index=['Not found']))

    classes_dict = classes.T.to_dict(orient='records')
    
    return classes_dict[0]
    
TRAINING_FOLDERER = '/Volumes/GE_2022/Image_analysis/Data/ILSVRC/Data/CLS-LOC/'
TOUCAN_ID = 'n01843383'
TAR_FILE = '/Volumes/GE_2022/Image_analysis/Data/imagenet-object-localization-challenge/imagenet_object_localization_patched2019.tar.gz'
OUTPUT_FOLDER = TAR_FILE[:37]
MODEL_OUTPUT_FOLDER = OUTPUT_FOLDER+'output/'
ROOT_LOGDIR = '/Volumes/GE_2022/Image_analysis/training_logs/'
ANNOTATION_FOLDER = '/Volumes/GE_2022/Image_analysis/Data/ILSVRC/Annotations'
LIST_OF_10_ANIMAL_IDS = ['n01843383','n01632777','n01669191','n01774384','n01774750','n01775062','n01806143','n01833805','n01882714','n01910747','Not found']
ALL_CLASS_NAMES_DICT = get_classes_dict()
LIST_OF_10_ANIMALS = [ALL_CLASS_NAMES_DICT.get(x) for x in LIST_OF_10_ANIMAL_IDS]
GET_CATEGORY_INT_FROM_ID = dict(zip(LIST_OF_10_ANIMAL_IDS,range(11)))
INPUT_SHAPE = (227, 227, 3)
AlexNet_input_shape = (227, 227, 3)
VGG_input_shape = (224,224,3)
GoogleNet_input_shape = VGG_input_shape
#googlenet variables
kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

DEBUG_LEVEL = 0

def change_cdr():
    os.chdir(os.path.dirname(sys.argv[0]))

    
def check_external_drive():
    if not os.path.exists(OUTPUT_FOLDER): raise Exception('Can\'t find the external drive')
    else: print('Drive found')

def initialising_the_tar_files():
    # Pointing to zip folder
    start_time = time.time()
    tar = tarfile.open(TAR_FILE, "r:gz")
    
    # Extracting members - 26mins RU
    print('tarfile.open\t\t'+str(round(time.time()-start_time,2))+'s')
    members = tar.getmembers() # Takes 26 mins
    print('tar.getmembers'+str(round(time.time()-start_time,2))+'s')
    
    return tar,members

def extract_images_for_ten_animals():
    tar,members = initialising_the_tar_files()
        
def view_on_folder_structure(members):
    locations = []
    for i,member in enumerate(members): 
        name = re.sub('\d{1,9}','XXXXXX',member.name)
        if name not in locations: locations.append(name)
    return locations

def get_run_logdir():
    global ROOT_LOGDIR
    run_id = time.strftime('run_%Y%m%d-%H_%M_%S')
    return os.path.join(ROOT_LOGDIR, run_id)

def print_image_from_loc(iloc):
    img=mpimg.imread(iloc)
    plt.imshow(img)
    plt.show()
    
def print_image_from_array(array):
    plt.imshow(array.astype(int))
    plt.show()

def extract_annotation_values_from_filename(annotation_loc):
    
    with open(annotation_loc,'r') as f: atext = f.read()
    soup = BeautifulSoup(atext, 'html.parser')
    annotation_vals = extract_annotation_values_from_soup(soup)
    return annotation_vals

def extract_annotation_values_from_soup(soup):
    attr_texts = []
    list_of_attrs = ['xmin','xmax','ymin','ymax','name','truncated','difficult']
    for i in list_of_attrs:     attr_texts.append(soup.find(i).text)
    return attr_texts
 
def print_image_with_annotation(iloc,aloc):
    if DEBUG_LEVEL>0: print('print_image_with_annotation')
    annotation_vals = extract_annotation_values_from_filename(aloc)
    avals = [int(x) for x in annotation_vals[:4]]
    fig = plt.figure()
    ax = fig.subplots(1)
    img=mpimg.imread(iloc)
    
    [x for x in glob.iglob(ANNOTATION_FOLDER+'/**/'+annotation_vals[4]+'.xml',recursive=True)]
    
    rect = patches.Rectangle((avals[0],avals[2]),avals[1]-avals[0],avals[3]-avals[2], edgecolor='r', facecolor="none")
    ax.imshow(img)
    ax.add_patch(rect)
    plt.show()

def print_image_and_annotation_from_pd_row(row):
    print_image_with_annotation(row.image_loc,row.annotation_loc)

def print_numpy_array(array):
    fig = plt.figure()
    ax = fig.subplots(1)
    ax.imshow(array)
    plt.show()

def numpy_array_from_row_using_zero_padding(row,input_shape):
    img=mpimg.imread(row.image_loc)
    # greyscale to rgb
    if len(img.shape)==2:
        img = np.repeat(img,3).reshape(img.shape+(3,))
    # Crop if there is an annotation file
    if not pd.isnull(row.annotation_loc):
        crop_dims = [int(x) for x in extract_annotation_values_from_filename(row.annotation_loc)[:4]]
        cropped_img = img[crop_dims[2]:crop_dims[3],crop_dims[0]:crop_dims[1]]
    else:   cropped_img = img
    # Resize to fixed dimensions
    # if cropped_img.shape
    newx, newy, channels = input_shape
    resized_img = cv2.resize(cropped_img,(newx,newy))
    return resized_img

def find_cropped_and_resized_numpy_array_from_row(row,input_shape):
    img=mpimg.imread(row.image_loc)
    # greyscale to rgb
    if len(img.shape)==2:
        img = np.repeat(img,3).reshape(img.shape+(3,))
    # Crop if there is an annotation file
    if not pd.isnull(row.annotation_loc):
        crop_dims = [int(x) for x in extract_annotation_values_from_filename(row.annotation_loc)[:4]]
        cropped_img = img[crop_dims[2]:crop_dims[3],crop_dims[0]:crop_dims[1]]
    else:   cropped_img = img
    # Resize to fixed dimensions
    
    newx, newy, channels = input_shape
    resized_img = cv2.resize(cropped_img,(newx,newy))
    return resized_img

def find_numpy_array_from_row(row):
    return mpimg.imread(row.image_loc)
    
def get_image_shape_from_loc(iloc):
    return find_numpy_array_from_row(iloc).shape
    

def test_shape_of_image(iloc):
    if os.path.isdir(iloc): return False
    img=mpimg.imread(iloc)
    if len(img.shape)!=3: return False
    if img.shape[2]!=3: return False
    else: return True

def aspect_ratio_from_annotation_values(vals):
    vals = [int(x) for x in vals[:4]]
    # Height divided by width 
    return (vals[2]-vals[3]) / (vals[0]-vals[1])

def get_array_from_image_loc(iloc):
    img=mpimg.imread(iloc)
    return img

def match_in_list(string,list_of_ids):
    for i in list_of_ids:
        if re.search(i,string): return True
    return False

def gather_df_info():
    # Finding existing files
    files = [x for x in glob.iglob('/Volumes/GE_2022/Image_analysis/Data/ILSVRC/Data/CLS-LOC/**',recursive=True)]
    annotations = [x for x in glob.iglob(ANNOTATION_FOLDER+'/**',recursive=True)]
    
    # Putting them into dataframe
    df = pd.DataFrame({'image_loc':files})
    df['basename'] = df.apply(lambda row: os.path.basename(row.image_loc),axis=1)
    df['filename'] = df.apply(lambda row: os.path.splitext(row.basename)[0],axis=1)
    df['dirname'] = df.apply(lambda row: os.path.dirname(row.image_loc),axis=1)
    df['isdir'] = df.apply(lambda row: os.path.isdir(row.image_loc),axis=1)
    
    df_annotations = pd.DataFrame({'annotation_loc':annotations})
    df_annotations['anotation_basename'] = df_annotations.apply(lambda row: os.path.basename(row.annotation_loc),axis=1)
    df_annotations['filename'] = df_annotations.apply(lambda row: os.path.splitext(row.anotation_basename)[0],axis=1)
        
    df = df.merge(df_annotations,how='left',on='filename')
    df['toucan'] = [x.startswith('n01843383') for x in df['filename']]
    df['id'] = df.apply(lambda row: os.path.basename(row.dirname),axis=1)
    df['in_list_of_10'] = df.apply(lambda row: match_in_list(row.filename,LIST_OF_10_ANIMALS),axis=1)
    
    df['category_int'] =  df.apply(lambda row: GET_CATEGORY_INT_FROM_ID.get(row.id),axis=1).fillna(10)
    df['category_text'] = df.apply(lambda row: ALL_CLASS_NAMES_DICT.get(row.basename[:9]),axis=1)
    df['clean'] = (df.isdir==False) & (~df.image_loc.isnull() & (~df.annotation_loc.isnull()))
    df.to_pickle('/Volumes/GE_2022/Image_analysis/dataframe.pkl')

def read_df_from_file():
    #Now read df from file as is quicker than rebuilding
    df = pd.read_pickle('/Volumes/GE_2022/Image_analysis/dataframe.pkl')
    return df

'''
def create_training_and_valuation_sets_only_annotated_images(df,input_shape):
    # split data
    toucan_clean =  df[df.toucan & df.clean]
    toucan_unclean = df[df.toucan & ~df.clean & ~df.isdir]
    not_toucan = df[~df.toucan & df.clean]
    half_length = len(toucan_clean)


    # create arrays
    X_train_true = np.zeros(shape=(half_length,)+input_shape)
    X_train_false = np.zeros(shape=(half_length,)+input_shape)
    ctr=0
    
    # Add images to arrays
    # True images
    for i,row in toucan_clean.sample(half_length).iterrows(): 
        img = find_cropped_and_resized_numpy_array_from_row(row,input_shape)
        # Reshaping grescale to consistent matrix shape
        X_train_true[ctr,:,:,:] = img
        ctr+=1
    
    ctr=0
    # False images
    for i,row in not_toucan.sample(half_length).iterrows(): 
        img = find_cropped_and_resized_numpy_array_from_row(row,input_shape)
        # Reshaping grescale to consistent matrix shape
        if not test_shape_of_image(row.image_loc): img = np.stack((img,)*3, axis=-1)
        X_train_false[ctr,:,:,:] = img
        ctr+=1
    
    # Splitting training and validation
    validation_percentage = 20/100
    true_len = X_train_true.shape[0]
    number_to_use_for_validation_set = int(true_len * validation_percentage)
    X_train = np.vstack((X_train_true[:-number_to_use_for_validation_set,:,:,:],X_train_false[:-number_to_use_for_validation_set,:,:,:]))/255
    X_val = np.vstack((X_train_true[-number_to_use_for_validation_set:,:,:,:],X_train_false[-number_to_use_for_validation_set:,:,:,:]))/255
    # y values
    y_train = np.array([1] * (true_len-number_to_use_for_validation_set) + [0] * (true_len-number_to_use_for_validation_set))
    y_val = np.array([1] * number_to_use_for_validation_set + [0] * number_to_use_for_validation_set)
    
    np.save(os.path.join(OUTPUT_FOLDER,'X_train.npy'),X_train)
    np.save(os.path.join(OUTPUT_FOLDER,'X_val.npy'),X_val)
    np.save(os.path.join(OUTPUT_FOLDER,'y_train.npy'),y_train)
    np.save(os.path.join(OUTPUT_FOLDER,'y_val.npy'),y_val)
'''
    
def create_training_and_valuation_sets_all_images():
    df = read_df_from_file()
    # split data
    positive_examples =  df[df.in_list_of_10 & ~df.isdir]
    average_group_size = int(np.mean(positive_examples.groupby('category_text')['image_loc'].count()))
    negative_examples = df[~df.in_list_of_10 & ~df.isdir]
    full_df = pd.concat([positive_examples,negative_examples.sample(average_group_size)]).reset_index(drop=True)

    # Splitting training and validation data
    validation_percentage = 10/100
    df_train,df_test = train_test_split(full_df,test_size=validation_percentage)
    df_train, df_val = train_test_split(df_train,test_size=validation_percentage)
    
    # create arrays
    frames, labels = [df_test,df_val,df_train], ['test','val','train']
    for i,df_loop in enumerate(frames): 
        for tmp_shape in [AlexNet_input_shape,VGG_input_shape]: 
            tmp_label = labels[i]+'_'+str(tmp_shape[0])
            create_memmap_array_on_file(df_loop,tmp_label,tmp_shape)
        
        y_array = np.nan_to_num(np.array(df_loop.category_int),nan=10)
        y_label = 'y_'+labels[i].split('_')[0]
        np.save(os.path.join(OUTPUT_FOLDER,y_label+'.npy'),y_array,allow_pickle=True)


def create_memmap_array_on_file(df_func,label,input_shape):
    array_size = df_func.image_loc.count()
    file = os.path.join(OUTPUT_FOLDER,'X_'+label+'.npy')
    if os.path.exists(file):    os.remove(file)
    np.save(file,np.zeros((array_size,)+input_shape))

    # read as memmaps
    memmap = np.load(os.path.join(OUTPUT_FOLDER,'X_'+label+'.npy'),mmap_mode='r+')
    
    ctr=0
    for i,row in df_func.iterrows():
        img = find_cropped_and_resized_numpy_array_from_row(row,input_shape)
        memmap[ctr,:,:,:] = img
        ctr+= 1
    
   

def create_array_from_df(df):
    ctr=0
    array_size = df.image_loc.count()
    tmp_array = np.zeros(shape=(array_size,)+INPUT_SHAPE)
    for i,row in df.iterrows():
        img = find_cropped_and_resized_numpy_array_from_row(row,INPUT_SHAPE)
        # Reshaping grescale to consistent matrix shape
        tmp_array[ctr,:,:,:] = img
        ctr+=1
    y_array = np.nan_to_num(np.array(df.category),nan=10)
    return tmp_array, y_array

def saving_arrays(data_array,y_array,label):
    np.save(os.path.join(OUTPUT_FOLDER,'X_'+label+'.npy'), data_array)
    np.save(os.path.join(OUTPUT_FOLDER,'y_'+label+'.npy'), y_array)
    print('saving:\tX_'+label)

def read_training_and_validation_sets_from_file(input_shape):
    X_train = np.load(os.path.join(OUTPUT_FOLDER,'X_train_'+str(input_shape[0])+'.npy'),mmap_mode='r+')
    X_val = np.load(os.path.join(OUTPUT_FOLDER,'X_val_'+str(input_shape[0])+'.npy'),mmap_mode='r+')
    y_train = np.load(os.path.join(OUTPUT_FOLDER,'y_train.npy'),mmap_mode='r+')
    y_val = np.load(os.path.join(OUTPUT_FOLDER,'y_val.npy'),mmap_mode='r+')
    return X_train,X_val,y_train,y_val

def build_AlexNet_model():

    model = keras.models.Sequential([
        Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=AlexNet_input_shape),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),
        Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),
        Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        BatchNormalization(),
        Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        BatchNormalization(),
        Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        BatchNormalization(),
        MaxPool2D(pool_size=(3,3), strides=(2,2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(11, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])

    return model

def print_model_history_and_ROC_curve(name,model,history,X_val,y_val):
    # Plotting the model improvements during training
    hst_df = pd.DataFrame(history.history)
    fig = hst_df.plot(figsize=(8,5)).get_figure()
    
    plt.grid(True)
    fig.savefig(MODEL_OUTPUT_FOLDER+name+'_training_accuracy.png')
    plt.show()
    # Plotting the ROC curve    
    val_predictions = model.predict(X_val)
    if len(val_predictions)==3: val_predictions = val_predictions[2]
    tst = plot_roc(y_val,val_predictions)
    lines, handles = tst.get_legend_handles_labels()
    classes_dict = get_classes_dict()
    for i in range(10):        
        handles[i] = re.sub('(?<=class )[\d\.]{2,4}',classes_dict.get(LIST_OF_10_ANIMAL_IDS[i]),handles[i]) 
    plt.show()

    

    tst.get_figure().savefig(MODEL_OUTPUT_FOLDER+name+'_ROC.png')
    confusion_matrix = tf.math.confusion_matrix(np.argmax(val_predictions, axis=1), y_val)
    np.save(MODEL_OUTPUT_FOLDER+name+'_confusion_matrix.npy',confusion_matrix.numpy())

    
def run_AlexNet_model():
    # read items from file
    X_train,X_val,y_train,y_val = read_training_and_validation_sets_from_file(AlexNet_input_shape)
    # create model
    AlexNet_model = build_AlexNet_model()
    run_logdir =  get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train[:1000,:,:,:], y_train[:1000])).shuffle(buffer_size=5000).batch(32,drop_remainder=True)
    validation_ds = tf.data.Dataset.from_tensor_slices((X_val[:1000,:,:,:], y_val[:1000])).shuffle(buffer_size=5000).batch(32,drop_remainder=True)
    
    # train model
    history = AlexNet_model.fit(train_ds,
                      epochs=30,
                      validation_data=validation_ds,
                      validation_freq=1,
                      callbacks=[tensorboard_cb])
    
    AlexNet_model.save(MODEL_OUTPUT_FOLDER+'Alexnet_10_model.h5')
    
    print_model_history_and_ROC_curve('AlexNet_10',AlexNet_model,history,X_val,y_val)



def build_VGG_model():
    model = keras.models.Sequential([
        # 1st Conv Block
        Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu', input_shape=VGG_input_shape),
        Conv2D (filters =64, kernel_size =3, padding ='same', activation='relu'),
        MaxPool2D(pool_size =2, strides =2, padding ='same'),
        # 2nd Conv Block
        Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu'),
        Conv2D (filters =128, kernel_size =3, padding ='same', activation='relu'),
        MaxPool2D(pool_size =2, strides =2, padding ='same'),
        # 3rd Conv block  
        Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu'), 
        Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu'), 
        Conv2D (filters =256, kernel_size =3, padding ='same', activation='relu'), 
        MaxPool2D(pool_size =2, strides =2, padding ='same'),
        # 4th Conv block
        Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu'),
        Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu'),
        Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu'),
        MaxPool2D(pool_size =2, strides =2, padding ='same'),
        # 5th Conv block
        Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu'),
        Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu'),
        Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu'),
        MaxPool2D(pool_size =2, strides =2, padding ='same'),
        # Fully connected layers  
        Flatten(), 
        Dense(units = 4096, activation ='relu'), 
        Dense(units = 4096, activation ='relu'), 
        Dense(units = 11, activation ='softmax')
        ])
        
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
    return model

def run_VGG_model():
    # read items from file
    X_train,X_val,y_train,y_val = read_training_and_validation_sets_from_file(VGG_input_shape)
    # create model
    VGG_model = build_VGG_model()
    run_logdir =  get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(X_train.shape[0]).batch(32,drop_remainder=True)
    validation_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(X_val.shape[0]).batch(32,drop_remainder=True)
    
    # train model
    history = VGG_model.fit(train_ds,
                      epochs=30,
                      validation_data=validation_ds,
                      validation_freq=1,
                      callbacks=[tensorboard_cb])

    VGG_model.save(MODEL_OUTPUT_FOLDER+'VGG_model.h5')

    print_model_history_and_ROC_curve('VGG',VGG_model,history,X_val,y_val)
    
#%% GoogleNet implementation 

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def build_GoogleNet_model():
    initial_lrate = 0.01
    
    input_layer = Input(shape=(224, 224, 3))

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)
    
    x = inception_module(x,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a')
    
    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64,
                         name='inception_3b')
    
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)
    
    x = inception_module(x,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         name='inception_4a')
    
    
    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(11, activation='softmax', name='auxilliary_output_1')(x1)
    
    x = inception_module(x,
                         filters_1x1=160,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4b')
    
    x = inception_module(x,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4c')
    
    x = inception_module(x,
                         filters_1x1=112,
                         filters_3x3_reduce=144,
                         filters_3x3=288,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4d')
    
    
    x2 = AveragePooling2D((5, 5), strides=3)(x)
    x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(11, activation='softmax', name='auxilliary_output_2')(x2)
    
    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_4e')
    
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)
    
    x = inception_module(x,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5a')
    
    x = inception_module(x,
                         filters_1x1=384,
                         filters_3x3_reduce=192,
                         filters_3x3=384,
                         filters_5x5_reduce=48,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5b')
    
    x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
    
    x = Dropout(0.4)(x)
    
    x = Dense(11, activation='softmax', name='output')(x)
    
    model = Model(input_layer, [x, x1, x2], name='inception_v1')
    
    sgd = SGD(learning_rate=initial_lrate, momentum=0.9, nesterov=False)
    
    model.compile(loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'], loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
    return model



def run_GoogleNet_model():
    # read items from file
    X_train,X_val,y_train,y_val = read_training_and_validation_sets_from_file(GoogleNet_input_shape)
    # create model
    GoogleNet_model = build_GoogleNet_model()
    run_logdir =  get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    lr_sc = LearningRateScheduler(decay, verbose=1)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(X_train.shape[0]).batch(32,drop_remainder=True)
    validation_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(X_val.shape[0]).batch(32,drop_remainder=True)
    
    # train model
    history = GoogleNet_model.fit(train_ds,
                      epochs=30,
                      validation_data=validation_ds,
                      validation_freq=1,
                      callbacks=[lr_sc])

    GoogleNet_model.save(MODEL_OUTPUT_FOLDER+'GoogleNet_model.h5')

    print_model_history_and_ROC_curve('GoogleNet',GoogleNet_model,history,X_val,y_val)
   
#%%
# GoogleNet_model.summary()
    
# if __name__=='__main__':
#     run_AlexNet_model()
#     run_VGG_model()

    

#%%  Environment initiation
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepping data for the dashboard

"""


import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
import glob
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import os
import sys
from plotly.io import write_json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from CNN_model_training import OUTPUT_FOLDER,MODEL_OUTPUT_FOLDER,LIST_OF_10_ANIMALS
from sklearn.metrics import roc_auc_score,accuracy_score,classification_report,precision_recall_curve,precision_recall_fscore_support
from scikitplot.metrics import plot_roc
import seaborn as sns

def read_test_sets_from_file():
    test_frames = [x for x in glob.iglob(OUTPUT_FOLDER+'**test**')]     
    return [np.load(x,mmap_mode='r') for x in test_frames]
            
    
def predict_model_switch(model,X_test_227,X_test_224):
    if model.input.shape[1]==227:
        y_predict = model.predict(X_test_227)
        return y_predict
    elif model.input.shape[1]==224:
        y_predict = model.predict(X_test_224)
        if len(y_predict)==3: y_predict = y_predict[2]
        return y_predict
    else: 
        raise Exception('Model input size mismatch')
        
def save_roc_curve(label):
    y_test = np.load(OUTPUT_FOLDER+'y_test.npy')
    y_predict = np.load(MODEL_OUTPUT_FOLDER+'y_test_predict_'+str(label)+'.npy')
    y_onehot = pd.DataFrame(data=LabelBinarizer().fit_transform(y_test),columns=LIST_OF_10_ANIMALS)

    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    fig.update_layout(margin={'l': 0, 'b': 0, 't': 40, 'r': 0}, hovermode='closest')

    for i in range(y_predict.shape[1]): 
        y_scores = y_predict[:,i]

        y_true = y_onehot.iloc[:, i]

        fpr, tpr, _ = roc_curve(y_true,y_scores)
        auc_score = roc_auc_score(y_true, y_scores)
    
        name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines',hovertemplate=name))
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(
            xaxis_title='False Positive Rate',
            title='ROC curve for '+str(label)+' model',
            yaxis_title='True Positive Rate',
            xaxis=dict(constrain='domain'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

    fig.update_layout(legend=dict(
                    yanchor="top",
                    y=0.6,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(0,0,0,0)'
                ))
    write_json(fig,MODEL_OUTPUT_FOLDER+label+'_ROC.json')


#%% Making predictions

X_test_227, X_test_224, y_test = read_test_sets_from_file()

model_list_locs = [x for x in glob.iglob(MODEL_OUTPUT_FOLDER+'**.h5',recursive=True)]
model_list_names = [os.path.basename(x).split('.')[0].replace('_model','') for x in model_list_locs]
number_of_models = len(model_list_names)

for i in model_list_locs:
    label = os.path.basename(i).split('.')[0].replace('_model','')
    model = tf.keras.models.load_model(i)
    y_test_predict = predict_model_switch(model,X_test_227,X_test_224)
    np.save(MODEL_OUTPUT_FOLDER+'y_test_predict_'+label,y_test_predict)

#%% Build df
classification_dict = []
accuracy = []
for i in model_list_locs:
    label = os.path.basename(i).split('_model.')[0]
    predict_loc = MODEL_OUTPUT_FOLDER+'y_test_predict_'+label+'.npy'
    y_pred = np.load(predict_loc)
    cr = classification_report(y_test,np.argmax(y_pred,axis=1),target_names=LIST_OF_10_ANIMALS,output_dict=True)
    del cr['accuracy']
    classification_dict.append(cr)
    accuracy.append(accuracy_score(y_test, np.argmax(y_pred,axis=1)))
    cm = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
    np.save(MODEL_OUTPUT_FOLDER+label+'_confusion_matrix.npy',cm)
    save_roc_curve(label)
    

#%% Output dataframe to file 
cols, vals = [], []
for i, d in enumerate(classification_dict):
    for j in d.keys():
        
        for k in d[j].keys():
            
            cols.append(f'{j}_{k}')
            vals.append(d[j][k])
    cols.append('accuracy')
    vals.append(accuracy[i])

data = np.array(vals).reshape(len(model_list_names),int(len(cols)/number_of_models))
col_list = cols[:int(len(cols)/number_of_models)]
model_df = pd.DataFrame(data=data,columns=col_list,dtype=float)
model_df['model'] = model_list_names


model_df.to_pickle(MODEL_OUTPUT_FOLDER+'model_summary.pkl')


#%%






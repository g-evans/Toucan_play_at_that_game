#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Personalised dashboard to comapare image models created

"""

from dash import Dash, html, dcc, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import tensorflow as tf
import glob
from plotly.io import read_json
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_curve, auc
from sklearn.preprocessing import LabelBinarizer
from CNN_model_training import MODEL_OUTPUT_FOLDER,OUTPUT_FOLDER,LIST_OF_10_ANIMALS,ALL_CLASS_NAMES_DICT,read_df_from_file,get_array_from_image_loc
from sklearn.metrics import roc_auc_score,accuracy_score,classification_report,precision_recall_curve,precision_recall_fscore_support
from scikitplot.metrics import plot_roc
import seaborn as sns
import time

def read_test_sets_from_file():
    return [np.load(x,mmap_mode='r') for x in glob.iglob(OUTPUT_FOLDER+'**test**')] 

def clear_log():
    with open('/Users/guyevans/Downloads/tst.txt','w') as f: f.write('')

def log(value):
    with open('/Users/guyevans/Downloads/tst.txt','a+') as f: 
        f.write(time.strftime("%Y%m%d, %H:%M:%S:\t"))
        f.write(str(value)+'\n')

def get_a_number_of_toucan_images(df_images,num_to_show):    
    lst = [get_array_from_image_loc(x) for x in df_images['image_loc'][(df_images.category=='toucan')&(df_images.isdir==False)].sample(num_to_show)]
    return lst

'''

def read_test_sets_from_file():
    #test_set_list = [x for x in glob.iglob(OUTPUT_FOLDER+'/**test**.npy',recursive=True)] #dynamic version
    test_set_list = ['/Volumes/GE_2022/Image_analysis/Data/X_test_227.npy',     # keep specific as only two input shapes possible
                     '/Volumes/GE_2022/Image_analysis/Data/X_test_224.npy',
                     '/Volumes/GE_2022/Image_analysis/Data/y_test.npy']
    sets = []
    for i in test_set_list:
        sets.append(np.load(i))
    return sets

def predict_model_switch(model,X_test_227,X_test_224):
    if model.input.shape[1]==227:
        return model.predict(X_test_227)
    elif model.input.shape[1]==224:
        return model.predict(X_test_224)
    else: 
        raise Exception('Model input size mismatch')
'''
def filter_array_on_hoverdata(hoverdata,X_test,y_test,y_predict):
    log('filter_array_on_hoverdata')
    log(str(hoverdata))
    for i in range(11): 
        if hoverdata.get('x') == 'hold_value':
            return np.zeros((1,)+X_test_227.shape[1:])
    else: 
        real_filter = y_test == [i for i, x in enumerate(LIST_OF_10_ANIMALS) if x==hoverdata.get('y')][0]
        predict_filter = np.argmax(y_predict,axis=1)== [i for i, x in enumerate(LIST_OF_10_ANIMALS) if x==hoverdata.get('x')][0]
        
        double_filter = (real_filter) & (predict_filter)
        if sum(double_filter)==0:
            return np.zeros((1,)+X_test_227.shape[1:])
        else:
            return X_test[double_filter,:,:,:]

#%% Gathering data
# files
model_list_locs = [x for x in glob.iglob(MODEL_OUTPUT_FOLDER+'**.h5',recursive=True)]
model_list_names = [os.path.basename(x).split('.')[0].replace('_model','') for x in model_list_locs]

# formatted data
model_df = pd.read_pickle(MODEL_OUTPUT_FOLDER+'model_summary.pkl')
y_test = np.load(OUTPUT_FOLDER + 'y_test.npy')
X_test_227, X_test_224, y_test = read_test_sets_from_file()


#%% read from file
df_images = read_df_from_file()


#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children =[
        html.Div(
            children=[
                dcc.Graph(
                    id='accuracy',
                    hoverData={'points': [{'model': True}]}
                ), dcc.Graph(
                    id='confusion_matrix',
                    hoverData={'points': [{'x':'hold_value','y':True,'z':True}]}
                )
            ], style={'display': 'inline-block', 'width': '49%','horizontal-align': 'left','vertical-align': 'top'}
        ),
        html.Div(
            children=[
                dcc.Graph(
                        id='roc_curve',
                        hoverData={'points': [{'x':True}]}
                    ),
                dcc.Graph(
                    id='image_preview',
                    )
            ], style={'display': 'inline-block', 'width': '48%','horizontal-align': 'right','vertical-align': 'top'})
        ])

@app.callback(
    Output('accuracy', 'figure'),
    Input('accuracy', 'hoverData'))

def plot_accuracy(hoverData):
    label = hoverData['points'][0].get('label')

    models = model_df.model
    column_list = [x for x in model_df.columns[-10:] if 'support' not in x and x!='model']
    fig = go.Figure()
    # Change the bar mode
    fig.update_layout(barmode='group',title = '<b>Toucan play at that game</b>',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)'
                      )#color_discrete_sequence=px.colors.qualitative.Pastel 
    for i, col in enumerate(column_list):
        if col!='model': 
            fig.add_trace(go.Bar(name=col,x=model_df['model'],y=model_df[col],yaxis='y', offsetgroup= i,hovertemplate='%{x}<br>'+col+': %{y}<extra></extra>'))
    
    fig.update_yaxes( range=[0,1], tickformat='.2f', autorange=False)
    fig.update_layout(margin=dict(t=50, l=0))
    
    return fig

@app.callback(
    Output('confusion_matrix', 'figure'),
    Input('accuracy', 'hoverData'))

def plot_confusion_matix(hoverData):
    label = hoverData['points'][0].get('label')
    if label == None: label = 'Alexnet_10' 

    z = np.load(MODEL_OUTPUT_FOLDER+label+'_confusion_matrix.npy')[::-1] #invert to ensure correct chart
    
    x = [LIST_OF_10_ANIMALS[x] for x in range(11)]
    y = x[::-1]
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]
    
    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text,hovertemplate =
            'Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>')
    
    # add title
    fig.update_layout(title_text='Confusion matrix for model: '+str(label))
    
    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.1,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    
    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.4,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))
    
    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=150, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = False
    
    # fig.update_traces(customdata=df['model'])

    # fig.update_layout(margin={'l': 0, 'b': 0, 't': 40, 'r': 0}, hovermode='closest')
    # fig.update_xaxes(range=[0, 1])
    # fig.update_yaxes(range=[0, 1])
    # fig.update_layout(legend=dict(
    #                 yanchor="top",
    #                 y=0.99,
    #                 xanchor="right",
    #                 x=0.01
    #             ))
    return fig

@app.callback(
    Output('roc_curve', 'figure'),
    Input('accuracy', 'hoverData'))

def make_roc_curve(hoverData):
    label = hoverData['points'][0].get('label')   
    if label==None: label='Alexnet_10'
    fig = read_json(MODEL_OUTPUT_FOLDER+label+'_ROC.json',output_type='Figure')
    return fig


@app.callback(
    Output('image_preview', 'figure'),
    Input('confusion_matrix', 'hoverData'),
    Input('accuracy', 'hoverData')
    )

def update_image_preview (hoverData_cm,hoverData_accuracy):
    label = hoverData_accuracy['points'][0].get('label')
    predict_points = hoverData_cm['points'][0]
    max_images_to_show = 4
    log('update_image_preview:'+str(hoverData_cm['points'][0]))
    title= 'Examples'
    if label==None: 
        log('escaping due to no model selected')
        img_lst = get_a_number_of_toucan_images(df_images,4)
        
    elif predict_points.get('x')== None or predict_points.get('x')=='hold_value': 
        log('escaping due to no data passed')
        img_lst = get_a_number_of_toucan_images(df_images,4)
    
    elif predict_points.get('z')==0:  
        log('escaping due to zero value')
        img_lst = get_a_number_of_toucan_images(df_images,4)

    else: 
        
        log('hover_data: update_image_preview:\t\t'+str(predict_points.get('z')))
        y_predict = np.load(MODEL_OUTPUT_FOLDER+'y_test_predict_'+str(label)+'.npy')
        
        imgs_to_show = filter_array_on_hoverdata(predict_points,X_test_227,y_test,y_predict)
    
        log('number of images:\t'+str(imgs_to_show.shape[0]))
        number_of_images_to_show = min([imgs_to_show.shape[0],max_images_to_show])
        
        img_lst = [imgs_to_show[x,:,:,:] for x in np.random.choice(imgs_to_show.shape[0],number_of_images_to_show,replace=False)]
        title = 'Actual:         '+str(predict_points.get('y'))+'<br>'+'Predicted:    '+str(predict_points.get('x'))+'\t\t\t\t('+str(label)+' model)'

    fig = make_subplots(rows=1, cols=4)
    [fig.add_trace(go.Image(z=img_lst[i]),1,i+1) for i, x in enumerate(img_lst)]

    fig.update_layout(coloraxis_showscale=False,title=title)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    log('returning fig')
    return fig
    

if __name__ == "__main__":
    clear_log()
    app.run_server(debug=True)



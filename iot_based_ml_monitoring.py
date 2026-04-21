# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -- End of cell --

#! pip install kaleido==0.1.*
# -- End of cell --

df = pd.read_csv(r'C:\Users\alexm\Downloads\archive (5)\smart_manufacturing_data.csv')
df.head()
# -- End of cell --

df.dtypes
# -- End of cell --

df.columns
# -- End of cell --

df['timestamp'] = pd.to_datetime(df['timestamp'],format='%Y-%m-%d %H:%M:%S')
# -- End of cell --

df.describe()
# -- End of cell --

df.dtypes
# -- End of cell --

df['maintenance_required'].value_counts()
# -- End of cell --

from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=2,cols=3,subplot_titles=('Temperature','Humidity','Vibration','Pressure','Energy Consumption','Predicted Remaining Lifetime'))
fig.add_trace(go.Box(y=df.loc[df['maintenance_required']==1,'temperature'],marker_color='#636EFA',showlegend=False),row=1,col=1)
fig.add_trace(go.Box(y=df.loc[df['maintenance_required']==0,'temperature'],marker_color='#EF553B',showlegend=False),row=1,col=1)
fig.add_trace(go.Box(y=df.loc[df['maintenance_required']==1,'humidity'],marker_color='#636EFA',showlegend=False),row=1,col=2)
fig.add_trace(go.Box(y=df.loc[df['maintenance_required']==0,'humidity'],marker_color='#EF553B',showlegend=False),row=1,col=2)
fig.add_trace(go.Box(y=df.loc[df['maintenance_required']==1,'vibration'],marker_color='#636EFA',showlegend=False),row=1,col=3)
fig.add_trace(go.Box(y=df.loc[df['maintenance_required']==0,'vibration'],marker_color='#EF553B',showlegend=False),row=1,col=3)
fig.add_trace(go.Box(y=df.loc[df['maintenance_required']==1,'pressure'],marker_color='#636EFA',showlegend=False),row=2,col=1)
fig.add_trace(go.Box(y=df.loc[df['maintenance_required']==0,'pressure'],marker_color='#EF553B',showlegend=False),row=2,col=1)
fig.add_trace(go.Box(y=df.loc[df['maintenance_required']==1,'energy_consumption'],marker_color='#636EFA',showlegend=False),row=2,col=2)
fig.add_trace(go.Box(y=df.loc[df['maintenance_required']==0,'energy_consumption'],marker_color='#EF553B',showlegend=False),row=2,col=2)
fig.add_trace(go.Box(y=df.loc[df['maintenance_required']==0,'predicted_remaining_life'],marker_color='#636EFA',showlegend=False),row=2,col=3)
fig.add_trace(go.Box(y=df.loc[df['maintenance_required']==0,'predicted_remaining_life'],marker_color='#EF553B',showlegend=False),row=2,col=3)
fig.add_trace(go.Box(y=[None],x=[None],marker_color='#636EFA',name='No'),row=1,col=1,)
fig.add_trace(go.Box(y=[None],x=[None],marker_color='#EF553B',name='Yes'),row=1,col=1,)
fig.update_layout({'plot_bgcolor':'rgba(0,0,0,0)', 'paper_bgcolor':'rgba(0,0,0,0)','width':1200})
fig.update_xaxes(showticklabels=False,linecolor='black',linewidth=1)
fig.update_yaxes(linecolor='black',linewidth=1,ticks='outside',tickcolor='black',tickwidth=1,ticklen=5)
# fig.show('svg')
# -- End of cell --

import plotly.express as px
fig = make_subplots(rows=1,cols=3,subplot_titles=('Machine Status','Anomaly Flag','Failure Type'),column_widths=[0.25, 0.25, 0.5])
fig1 = px.histogram(df.groupby('machine_status')['maintenance_required'].value_counts().reset_index(),x='machine_status',y='count',color='maintenance_required',barmode='group')
fig1.update_traces(showlegend=False)
fig2 = px.histogram(df.groupby('anomaly_flag')['maintenance_required'].value_counts().reset_index(),x='anomaly_flag',y='count',color='maintenance_required',barmode='group')
fig2.update_traces(showlegend=False)
fig3 = px.histogram(df.groupby('failure_type')['maintenance_required'].value_counts().reset_index(),x='failure_type',y='count',color='maintenance_required',barmode='group')
for trace in fig1.data:
    fig.add_trace(trace,row=1,col=1)
for trace in fig2.data:
    fig.add_trace(trace,row=1,col=2)
for trace in fig3.data:
    fig.add_trace(trace,row=1,col=3)
fig.update_layout({'width':1400,'legend':{'orientation':'h','xanchor':'center','x':0.5,'y':1.2}})
fig.update_xaxes(linecolor='black',linewidth=1)
fig.update_yaxes(linecolor='black',linewidth=1,ticks='outside',tickcolor='black',tickwidth=1,ticklen=5)
# fig.show('svg')
# -- End of cell --

fig=px.scatter(df.loc[df['machine_id']==23],x='timestamp',y='temperature',size='energy_consumption',color='anomaly_flag',facet_col='maintenance_required',width=1200)
# fig.show('svg')
# -- End of cell --

fig=px.scatter(df.loc[df['machine_id']==23],x='timestamp',y='humidity',size='vibration',color='machine_status',facet_col='maintenance_required',width=1200)
# fig.show('svg')
# -- End of cell --

fig = px.scatter(df.loc[df['machine_id']==23],x='timestamp',y='pressure',size='predicted_remaining_life',color='failure_type',facet_col='maintenance_required',width=1200)
# fig.show('svg')
# -- End of cell --

df['failure_type'] = df['failure_type'].map({'Normal':0,'Vibration Issue':1,'Overheating':2,'Electrical Fault':3,'Pressure Drop':4})
# -- End of cell --

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
corr_df = df.drop(columns=['machine_id']).select_dtypes(include=[int,float]).corr()
mask = np.triu(np.ones_like(corr_df))
np.fill_diagonal(mask,0)
sns.heatmap(corr_df,annot=True,cmap='Blues',mask=mask)
# plt.show()
# -- End of cell --

y = df['maintenance_required']
X = df.drop(columns=['failure_type','downtime_risk','timestamp','machine_id','maintenance_required'])
# -- End of cell --

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
print("Training Data: ",X_train.shape,y_train.shape)
print("Testing Data: ",X_test.shape,y_test.shape)
# -- End of cell --

from sklearn.metrics import confusion_matrix,f1_score,classification_report,accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
scores = []
def checkModel(model):
    clf = Pipeline([('scaler',StandardScaler()),('model',model)])
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test,y_pred))
    scores.append({'model':model.__class__.__name__,'accuracy':accuracy_score(y_test,y_pred),'f1_score':f1_score(y_test,y_pred), 'pipeline': clf})
    # sns.heatmap(confusion_matrix(y_test,y_pred,normalize='all'),annot=True)
# -- End of cell --

from sklearn.svm import SVC
checkModel(SVC())
# -- End of cell --

from sklearn.neighbors import KNeighborsClassifier
checkModel(KNeighborsClassifier())
# -- End of cell --

from sklearn.naive_bayes import GaussianNB
checkModel(GaussianNB())
# -- End of cell --

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
checkModel(AdaBoostClassifier())
# -- End of cell --

checkModel(RandomForestClassifier())
# -- End of cell --

from sklearn.tree import DecisionTreeClassifier
checkModel(DecisionTreeClassifier())
# -- End of cell --

from sklearn.linear_model import LogisticRegression
checkModel(LogisticRegression())
# -- End of cell --

from sklearn.neural_network import MLPClassifier
checkModel(MLPClassifier())
# -- End of cell --

import pickle
df_scores = pd.DataFrame(scores)
best_model_row = df_scores.loc[df_scores['accuracy'].idxmax()]
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model_row['pipeline'], f)
print(f"\nBest model ({best_model_row['model']}) saved to best_model.pkl with accuracy {best_model_row['accuracy']:.4f}")
# -- End of cell --


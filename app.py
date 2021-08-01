import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

#from kmodes.kmodes import KModes

#import pickle
import base64

import io

column = ['aging', 'subscribe', 'order', 'Tagihan']


def get_table_download_link(df):
    towrite = io.BytesIO()
    df.to_excel(towrite, encoding='utf-8', index=False, header=True, engine='xlsxwriter')
    towrite.seek(0)  # reset pointer
    #csv = df.to_csv(index=False,sep=';')
    b64 = base64.b64encode(towrite.read()).decode()
    new_filename = "datahasil.xlsx"
    href= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{new_filename}">Download file hasil clustering</a>'

    #href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download file hasil clustering</a>'
    return href
#end of get_table_download_link
    
    
def eda():
    st.header('Datasets')
    df = pd.read_csv('data_pelanggan.csv',sep=';')
    
    st.subheader('Data Awal '+str(df.shape))
    st.write(df.sample(10));

    
    if st.checkbox("Show Columns Histogram"):
        selected_columns = st.selectbox("Select Column",column)
        fig4= plt.figure()
        if selected_columns == 'Tagihan':
            sns.histplot(x = selected_columns, data=df,bins=15)
        elif selected_columns != '':
            sns.histplot(x = selected_columns, data=df)
        plt.xticks(rotation=45,ha='right')
        st.write(fig4)

def kmeans():
    st.header('K-Means')
    cols = ['Tagihan','aging','subscribe','order']
    
    df_master = pd.read_csv('data_pelanggan.csv',sep=';')     
    df1 = df_master[cols].copy()
    
    scaler = preprocessing.StandardScaler()
    model = scaler.fit(df1)
    df1 = model.transform(df1)
    pca = PCA(2) #mengubah menajdi 2 kolom
    df1 = pca.fit_transform(df1) #Transform data
    
    st.subheader('Pemilihan nilai K Menggunakan Elbow Method')
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    dbi = []
    slh = []
    
    K = range(2,9)
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(df1)
        kmeanModel.fit(df1)
        distortions.append(sum(np.min(cdist(df1, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / df1.shape[0])
        inertias.append(kmeanModel.inertia_)
        mapping1[k] = sum(np.min(cdist(df1, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / df1.shape[0]
        mapping2[k] = kmeanModel.inertia_
        #evaluasi silhouette
        slh.append(metrics.silhouette_score(df1,kmeanModel.labels_))
        #evaluasi DBI
        dbi.append(metrics.davies_bouldin_score(df1,kmeanModel.labels_))
     
     
    st.text('Metode Distortion')
    fig = plt.figure(figsize=(4,2))
    plt.plot(K,distortions,'bx-')
    plt.xlabel("Nilai K")
    plt.ylabel("Distortion")
    st.write(fig)
    
    
    st.text('Metode Inertia')
    fig2 = plt.figure(figsize=(4,2))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel("Nilai K")
    plt.ylabel('Inertia')
    st.write(fig2)
    
    st.text('Metode DBI')
    fig2 = plt.figure(figsize=(4,2))
    plt.plot(K, dbi, 'bx-')
    plt.xlabel("Nilai K")
    plt.ylabel('DBI')
    st.write(fig2)
    
    st.text('Metode Silhouette')
    fig2 = plt.figure(figsize=(4,2))
    plt.plot(K, slh, 'bx-')
    plt.xlabel("Nilai K")
    plt.ylabel('Silhouette')
    st.write(fig2)

    st.header('K-Means Modelling')
    k_value  = st.slider('Nilai K', min_value=3, max_value=10, step=1, value=4)
    

    model = KMeans(n_clusters=k_value,random_state=101) # isnisialisasi Kmeans dgn  nilai K yg dipilih
    model.fit_predict(df1) #proses Clustering
    label = model.predict(df1) #proses Clustering

    #center = pca.fit_transform(model.cluster_centers_)
    center = model.cluster_centers_
    
    #dibuat menjadi dataFrame
    #df_master['x1'] = dfnp[:,0]
    #df_master['y1'] = dfnp[:,1]
    
    df_master['x1'] = df1[:,0]
    df_master['y1'] = df1[:,1]   
    df_master['cluster'] = label
    
    
    
    fig3= plt.figure()
    sns.scatterplot(x='x1', y='y1',hue='cluster',data=df_master,alpha=1, s=40, palette='deep')
    plt.scatter(x=center[:, 0], y=center[:, 1], s=100, c='black', ec='red',label='centroid')
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    st.write(fig3)
    
    fig4= plt.figure()
    ax = sns.countplot(x ='cluster', data=df_master)
    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height()
        value = int(p.get_height())
        ax.text(_x, _y, value, ha="center")  
    st.write(fig4)
    
    cluster = df_master['cluster'].unique()
    cluster.sort()

    
    choice = st.selectbox("Pilih Kluster",cluster)
    res = df_master.loc[df_master['cluster'] == choice]
    st.subheader('Cluster '+str(choice)+': '+str(res.shape[0])+' data')
    st.write(res.sample(10))
     
    
    
    
def apps():
    cols = ['Tagihan','aging','subscribe','order']
    k_value = int(st.text_input('Nilai K:',value=4))
    data = st.file_uploader("Upload a Dataset", type=["csv"])
    if data is not None:
        
        
        df = pd.read_csv(data,sep=';')
        st.dataframe(df)
        

        df1 = df[cols].copy() 
        scaler = preprocessing.StandardScaler()
        model = scaler.fit(df1)
        df1 = model.transform(df1)
        
        pca = PCA(2)
        df1 = pca.fit_transform(df1)

        
        model = KMeans(n_clusters=k_value,random_state=101)
        model.fit(df1)
        label = model.predict(df1)
  
        center = model.cluster_centers_
        #center = pca.fit_transform(model.cluster_centers_)
        
        
        #dibuat menjadi dataFrame
        df['x1'] = df1[:,0]
        df['y1'] = df1[:,1]
        df['cluster'] = label
        st.write('Proses Dimulai...')
        for index, row in df.iterrows():
            st.write(str(row['No'])+'... cluster: ',str(row['cluster']))
        st.write('Proses Selesai')
        
        st.subheader('Data Hasil Kluster')
        st.dataframe(df)
        
        fig3= plt.figure()
        sns.scatterplot(x='x1', y='y1',hue='cluster',data=df,alpha=1, s=40, palette='deep')
        plt.scatter(x=center[:, 0], y=center[:, 1], s=100, c='black', ec='red',label='centroid')
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        st.write(fig3)

        fig4= plt.figure()
        ax = sns.countplot(x ='cluster', data=df)
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = int(p.get_height())
            ax.text(_x, _y, value, ha="center") 
        st.write(fig4)
        
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)
        
        
    
#end of apps
    
def main():
    """ Streamlit Pelanggaran Pilkada Jabar """

    activities = ['EDA','K-Means','Aplikasi Perhitungan']
    choice = st.sidebar.selectbox("Select Activities",activities)
    
    if choice == 'EDA':
        eda()
    elif choice == 'K-Means':
        kmeans()
    elif choice == 'Aplikasi Perhitungan':
        apps()
        

if __name__ == '__main__':
    main()
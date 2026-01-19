import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import folium
from streamlit_folium import st_folium
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def plot_graph(data, plot_key):
    plot_type = st.selectbox(
        'Pilih jenis plot:',
        ('Line Plot', 'Bar Plot', 'Scatter Plot', 'Histogram'),
        key=f'plot_type_selectbox_{plot_key}'
    )
    color = st.color_picker('Pilih warna plot', '#00f900', key=f'color_picker_{plot_key}')
    x_column = st.selectbox('Pilih kolom untuk sumbu X', data.columns, key=f'x_column_selectbox_{plot_key}')
    
    if plot_type != 'Histogram':
        y_column = st.selectbox('Pilih kolom untuk sumbu Y', data.columns, key=f'y_column_selectbox_{plot_key}')
    else:
        y_column = None
    plt.clf() 
    if plot_type == 'Line Plot':
        st.subheader('Line Plot')
        plt.plot(data[x_column], data[y_column] if y_column else data, color=color)
        plt.xticks(rotation=45)
        plt.title('Line Plot')

    elif plot_type == 'Bar Plot':
        st.subheader('Bar Plot')
        plt.bar(data[x_column], data[y_column] if y_column else data, color=color)
        plt.xticks(rotation=45)
        plt.title('Bar Plot')

    elif plot_type == 'Scatter Plot':
        st.subheader('Scatter Plot')
        plt.scatter(data[x_column], data[y_column] if y_column else data, color=color)
        plt.xticks(rotation=45)
        plt.title('Scatter Plot')

    elif plot_type == 'Histogram':
        st.subheader('Histogram')
        plt.hist(data[x_column], bins=10, color=color)
        plt.xticks(rotation=45)
        plt.title('Histogram')

    st.pyplot(plt)

def clean_data(df):
    df_drop = df.dropna()
    df_drop = df_drop.drop_duplicates()
    return df_drop

def extract_time_features(df):
    df['tapInTime'] = pd.to_datetime(df['tapInTime'])
    df['tapOutTime'] = pd.to_datetime(df['tapOutTime'])
    df['tapInHour'] = df['tapInTime'].dt.hour
    df['tapOutHour'] = df['tapOutTime'].dt.hour
    df['tapDay'] = df['tapInTime'].dt.dayofweek
    df['tapDay'] = df['tapDay'].replace({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    return df

def get_customer_data(df):
    customer = df[['payCardName', 'payCardBirthDate', 'payCardBank', 'payCardSex']]
    customer['age'] = 2025 - customer['payCardBirthDate']
    return customer

def count_banks(customer):
    cust_bank = customer.payCardBank.value_counts().reset_index(name='count')
    return cust_bank

def customer_age(customer):
    cust_age = customer[['payCardName', 'age']].groupby('age').size().reset_index(name='count')
    return cust_age

def customer_sex(customer):
    cust_sex = customer.groupby(['payCardSex', 'age'])['payCardName'].count().reset_index(name='count').sort_values(by='count', ascending=False)
    return cust_sex

def calculate_clv_metrics(data):
    clv = data[['payCardName', 'tapOutTime', 'payAmount']]
    max_date = clv['tapOutTime'].max()
    clv['Recency'] = (max_date - clv['tapOutTime']).dt.days
    frequency = clv['payCardName'].value_counts().reset_index()
    frequency.columns = ['payCardName', 'Frequency']
    clv = clv.merge(frequency, on='payCardName')
    clv.drop_duplicates(subset='payCardName', keep='first', inplace=True)
    clv.drop(columns=['tapOutTime', 'payAmount'], inplace=True)
    return clv

def calculate_trip_duration(df):
    df['tapInHour'] = df['tapInTime'].dt.hour
    df['tapOutHour'] = df['tapOutTime'].dt.hour
    df['tapOutHour'] = df['tapOutHour'].replace(0, 24)
    df['tripDuration'] = df['tapOutHour'] - df['tapInHour']
    trip_data = df.groupby(['tapInStopsName', 'tapOutStopsName', 'tripDuration'])['transID'].count().reset_index(name='trip')
    trip_data = trip_data.sort_values(by='tripDuration', ascending=False, ignore_index=True)
    
    return trip_data

def calculate_tapin_count(df):
    tapIn_counts = df.groupby(['tapInStopsName', 'tapInHour']).size().reset_index(name='tapInCounts')
    tapIn_counts_sorted = tapIn_counts.sort_values(by='tapInCounts', ascending=False, ignore_index=True)
    return tapIn_counts_sorted.head(5)

def calculate_tapout_count(df):
    tapOut_counts = df.groupby(['tapOutStopsName', 'tapOutHour']).size().reset_index(name='tapOutCounts')
    tapOut_counts_sorted = tapOut_counts.sort_values(by='tapOutCounts', ascending=False, ignore_index=True)
    return tapOut_counts_sorted.head(5)

def calculate_transaction_counts(df):
    transcount = pd.DataFrame(df[['tapInStopsName', 'tapOutStopsName']].groupby(['tapInStopsName', 'tapOutStopsName']).size().reset_index(name='TransactionCount').sort_values(by='TransactionCount', ascending=False, ignore_index=True))
    return transcount

def customer_transaction_counts(df, transcount):
    cor = df[['tapInStopsName', 'tapInStopsLat', 'tapInStopsLon', 'tapOutStopsName', 'tapOutStopsLat', 'tapOutStopsLon']]
    cor.drop_duplicates(inplace=True, ignore_index=True)
    res = cor.merge(transcount, on=['tapInStopsName', 'tapOutStopsName'])
    res.sort_values(by='TransactionCount', ascending=False, ignore_index=True, inplace=True)
    
    return res.head(10)

def plot_map_with_markers(df, transcount):
    res = customer_transaction_counts(df, transcount)
    center_latitude = -6.1751
    center_longitude = 106.8272
    zoom_level = 10
    m = folium.Map(location=[center_latitude, center_longitude], zoom_start=zoom_level)
    
    for index, row in res.iterrows():
        folium.Marker(
            location=[row['tapInStopsLat'], row['tapInStopsLon']],
            icon=folium.Icon(icon='cloud', color='green'),
            popup=row['tapInStopsName']
        ).add_to(m)

        folium.Marker(
            location=[row['tapOutStopsLat'], row['tapOutStopsLon']],
            icon=folium.Icon(icon='cloud', color='red'),
            popup=row['tapOutStopsName']
        ).add_to(m)

        folium.PolyLine(
            locations=[[row['tapInStopsLat'], row['tapInStopsLon']], [row['tapOutStopsLat'], row['tapOutStopsLon']]],
            color='blue'
        ).add_to(m)

    st_folium(m, width=700, height=500)

def perform_kmeans(df, clv_, n_clusters_min=2, n_clusters_max=10):
    clv = df.merge(clv_, on='payCardName')
    clv.rename(columns={'payAmount': 'Value'}, inplace=True)
    columns_to_scale = ['Recency', 'Frequency', 'Value']
    scaler = MinMaxScaler()
    clv[columns_to_scale] = scaler.fit_transform(clv[columns_to_scale])

    range_n_clusters = range(n_clusters_min, n_clusters_max + 1)
    
    ssd = []
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(clv[['Recency', 'Frequency', 'Value']])
        ssd.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range_n_clusters, ssd, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances (SSD)')
    plt.title('Elbow Curve for K-Means Clustering')
    st.pyplot(plt) 

    silhouette_scores = []
    for num_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(clv[['Recency', 'Frequency', 'Value']])
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(clv[['Recency', 'Frequency', 'Value']], cluster_labels)
        silhouette_scores.append(silhouette_avg)

    plt.figure(figsize=(8, 6))
    plt.plot(range_n_clusters, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for K-Means Clustering')
    st.pyplot(plt)  
    n_clusters = st.slider('Select number of clusters (k)', min_value=n_clusters_min, max_value=n_clusters_max, value=3, step=1)
    
    kmeans = KMeans(n_clusters=n_clusters)
    clv['Cluster'] = kmeans.fit_predict(clv[['Recency', 'Frequency', 'Value']])

    return clv, kmeans

def plot_customer_clustering(data):
    plot_type = st.selectbox(
        'Choose the plot projection:',
        ('2D', '3D')
    )
    
    if plot_type == '2D':
        st.subheader('2D Scatter Plot based on Recency and Frequency')
        plt.figure(figsize=(8, 6))
        plt.scatter(data['Recency'], data['Frequency'], c=data['Cluster'], cmap='viridis')
        plt.xlabel('Recency')
        plt.ylabel('Frequency')
        plt.title('Customer Clustering based on Recency and Frequency')
        plt.colorbar(label='Cluster')
        st.pyplot(plt)

    elif plot_type == '3D':
        st.subheader('3D Scatter Plot based on Recency, Frequency, and Value')
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data['Recency'], data['Frequency'], data['Value'], c=data['Cluster'], cmap='viridis')
        ax.set_xlabel('Recency')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Value')
        ax.set_title('Customer Clustering based on Recency, Frequency, and Value')
        plt.colorbar(ax.scatter(data['Recency'], data['Frequency'], data['Value'], c=data['Cluster'], cmap='viridis'))
        st.pyplot(fig)

def main():
    st.title('Analysis Customer Transjakarta')
    st.sidebar.header("Upload CSV & Select Options")
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        section_sidebar = st.sidebar.radio("Select Section", ["EDA", "Clustering"])
        if section_sidebar == 'EDA':
            section_eda = st.selectbox("Select Section", ["Preview Data", "Data Customer Time", "Customer Age", "Customer Banks Counts", "Recency and Frequency Analysis", "Trip Analysis", "Geographic Locations", ])
            if section_eda == "Preview Data":
                df = load_data(file)
                st.subheader('Data Preview')
                st.write(df)

                df_cleaned = clean_data(df)
                st.subheader('Cleaned Data')
                st.write(df_cleaned)

            elif section_eda == "Data Customer Time": 
                df_cleaned = clean_data(load_data(file))
                df_time_features = extract_time_features(df_cleaned)
                st.subheader("Data with Time Features")
                df_time = df_time_features[['payCardName','tapInTime', 'tapOutTime', 'tapInHour', 'tapOutHour', 'tapDay']].copy()
                st.write(df_time.head())

            elif section_eda == "Customer Age":
                df_cleaned = clean_data(load_data(file))
                df_time_features = extract_time_features(df_cleaned)
                customer_data = get_customer_data(df_time_features)
                st.subheader("Customer Data with Age")
                st.write(customer_data.head())

                cust_age = customer_age(customer_data)
                st.title('Plot Customer Age')
                plot_graph(cust_age, 'cust_age')  

            elif section_eda == "Customer Banks Counts":
                df_cleaned = clean_data(load_data(file))
                df_time_features = extract_time_features(df_cleaned)
                customer_data = get_customer_data(df_time_features)                
                bank_counts = count_banks(customer_data)
                st.subheader("Bank Type Counts")
                st.write(bank_counts.head())
                st.title('Plot Bank Counts')
                plot_graph(bank_counts, 'bank_counts')

            elif section_eda == "Recency and Frequency Analysis":
                df_cleaned = clean_data(load_data(file))
                df_time_features = extract_time_features(df_cleaned)
                customer_data = get_customer_data(df_time_features)       
                df_eda = df_time_features.merge(customer_data[['payCardName', 'age']], on='payCardName')

                data = df_eda[['payCardBank', 'payCardName', 'payCardSex',
                            'payCardBirthDate', 'tapInStops', 'tapInStopsName', 'tapInTime', 'tapOutStops', 
                            'tapOutStopsName', 'tapOutTime', 'payAmount', 'tapInHour', 'tapOutHour', 'tapDay', 'age']]

                cust_clv = calculate_clv_metrics(data)  
                clv_ = df_eda.groupby('payCardName')['payAmount'].sum().reset_index().sort_values(by='payAmount', ascending=False, ignore_index=True)
                st.write(cust_clv.head())

            elif section_eda == "Trip Analysis":                
                df_cleaned = clean_data(load_data(file))
                df_time_features = extract_time_features(df_cleaned)
                customer_data = get_customer_data(df_time_features)       
                df_eda = df_time_features.merge(customer_data[['payCardName', 'age']], on='payCardName')
                cust_trip_duration = calculate_trip_duration(df_eda)
                st.write(cust_trip_duration)

                section_trip = st.radio('select trip', ['tapIn', 'tapOut'])
                if section_trip == "tapIn":
                    transcount = calculate_transaction_counts(df_eda)
                    cust_tapin = calculate_tapin_count(df_eda)
                    st.write(cust_tapin)
                    plot_graph(cust_tapin, 'customer tapin transaction')

                    cus_transaction = customer_transaction_counts(df_eda, transcount)
                    cus_transaction = cus_transaction[['tapInStopsName', 'tapOutStopsName', 'TransactionCount']]
                    st.write(cus_transaction)

                elif section_trip == "tapOut":
                    transcount = calculate_transaction_counts(df_eda)
                    cust_tapout = calculate_tapout_count(df_eda)
                    st.write(cust_tapout)
                    plot_graph(cust_tapout, 'customer tapout transaction')

                    cus_transaction = customer_transaction_counts(df_eda, transcount)
                    cus_transaction = cus_transaction[['tapInStopsName', 'tapOutStopsName', 'TransactionCount']]
                    st.write(cus_transaction)

            elif section_eda == "Geographic Locations":
                df_cleaned = clean_data(load_data(file))
                df_time_features = extract_time_features(df_cleaned)
                customer_data = get_customer_data(df_time_features)       
                df_eda = df_time_features.merge(customer_data[['payCardName', 'age']], on='payCardName')
                transcount = calculate_transaction_counts(df_eda)
                longLat = customer_transaction_counts(df_eda, transcount)
                longLat_stasiun = longLat[['tapInStopsName', 'tapInStopsLon', 'tapOutStopsName', 'tapOutStopsLon']]
                st.write(longLat_stasiun)
                m = plot_map_with_markers(df_eda, transcount)
                st.write(m)

        elif section_sidebar == "Clustering":
            df_cleaned = clean_data(load_data(file))
            df_time_features = extract_time_features(df_cleaned)
            customer_data = get_customer_data(df_time_features)       
            df_eda = df_time_features.merge(customer_data[['payCardName', 'age']], on='payCardName')
            data = df_eda[['payCardBank', 'payCardName', 'payCardSex',
                            'payCardBirthDate', 'tapInStops', 'tapInStopsName', 'tapInTime', 'tapOutStops', 
                            'tapOutStopsName', 'tapOutTime', 'payAmount', 'tapInHour', 'tapOutHour', 'tapDay', 'age']]

            cust_clv = calculate_clv_metrics(data)  
            clv_ = df_eda.groupby('payCardName')['payAmount'].sum().reset_index().sort_values(by='payAmount', ascending=False, ignore_index=True)
            df_result, kmeans_model = perform_kmeans(cust_clv,clv_, n_clusters_min=2, n_clusters_max=10)
            st.subheader('Clustering Result')
            st.write(df_result.head())

            plot_customer_clustering(df_result)

if __name__ == "__main__":
    main()

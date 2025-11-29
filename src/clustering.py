
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_listeners(df):
    print("Running K-Means Clustering on Artists...")
    
    artist_stats = df.groupby('master_metadata_album_artist_name').agg({
        'ms_played': 'sum',
        'is_skipped': 'mean',
        'ts': 'count'
    }).reset_index()
    
    artist_stats.columns = ['Artist', 'Total_Ms', 'Skip_Rate', 'Play_Count']
    data = artist_stats[artist_stats['Play_Count'] > 20].copy()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(data[['Skip_Rate', 'Play_Count']])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X)
    
    print("Clustering Complete.")
    return data

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Read Datasets
df1 = pd.read_csv('datasets/Data/features_30_sec.csv')
df2 = pd.read_csv('datasets/Data/features_3_sec.csv')
print("Processing Dataset 1:")

#Drop unnecessary columns 
drop_columns_1 = ["chroma_stft_mean", "chroma_stft_var", "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var", "length"]
df1_cleaned = df1.drop(columns=drop_columns_1)
df1_cleaned = df1_cleaned.dropna()

drop_columns_2 = ["chroma_stft_mean", "chroma_stft_var", "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var" , "length"]
df2_cleaned = df2.drop(columns=drop_columns_2)  
df2_cleaned = df2_cleaned.dropna()

# Encode labels
le = LabelEncoder()
df1_cleaned["label"] = le.fit_transform(df1_cleaned["label"])
df2_cleaned["label"] = le.transform(df2_cleaned["label"])

# Scale features
scaler = StandardScaler()
feature_cols = [c for c in df1_cleaned.columns if c not in ("label", "filename")]
df1_cleaned[feature_cols] = scaler.fit_transform(df1_cleaned[feature_cols])
df2_cleaned[feature_cols] = scaler.transform(df2_cleaned[feature_cols])

df1_cleaned.to_csv('datasets/Data/features_30_sec_cleaned.csv', index=False)
df2_cleaned.to_csv('datasets/Data/features_3_sec_cleaned.csv', index=False)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Preprocessing complete. Cleaned files saved.")
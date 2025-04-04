import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('/mnt/data/train.csv')

# Step 1: Hapus fitur dengan missing value besar
fitur_dihapus = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
df.drop(columns=fitur_dihapus, inplace=True)

# Step 2: Tangani missing value
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('None')  # kategori diisi None
    else:
        df[col] = df[col].fillna(df[col].median())  # numerik diisi median

# Step 3: Encoding untuk fitur nonnumerik (kategorikal)
le_dict = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # simpan encoder kalau mau inverse_transform nanti

# Step 4: Pisahkan fitur (X) dan target/label (y)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Step 5: Split train-test 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

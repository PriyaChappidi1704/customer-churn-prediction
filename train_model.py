import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('Telco-Customer-Churn.csv')

# 1. Standardize all column names
df.columns = df.columns.str.strip().str.replace(' ', '').str.replace('_', '').str.lower()

# 2. Map the newer dataset column names back to standard names
rename_map = {
    'tenuremonths': 'tenure',
    'churnlabel': 'churn'
}
df.rename(columns=rename_map, inplace=True)

# 3. If there are multiple 'churn' columns, keep only the first one
df = df.loc[:, ~df.columns.duplicated()]

# Fallback just in case 'churn' is missing but 'churnvalue' exists
if 'churn' not in df.columns and 'churnvalue' in df.columns:
    df['churn'] = df['churnvalue']

# 4. Clean data
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
df.dropna(subset=['totalcharges', 'tenure', 'monthlycharges', 'contract', 'churn'], inplace=True)

# 5. Select the 4 features
features = ['tenure', 'monthlycharges', 'totalcharges', 'contract']
X = df[features].copy()

# 6. Handle Churn target (Foolproof conversion regardless of data type)
y = df['churn'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)

# 7. Encode the categorical 'contract' column
le = LabelEncoder()
X['contract'] = le.fit_transform(X['contract'].astype(str))

# 8. Handle Imbalanced Data using SMOTE
print("Applying SMOTE to handle imbalance...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 9. Train the Model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=15, max_depth=5, random_state=42)
model.fit(X_resampled, y_resampled)

# 10. Save the model and encoder
joblib.dump(model, 'churn_model.pkl')
joblib.dump(le, 'contract_encoder.pkl')
print("Success! Model saved as churn_model.pkl")
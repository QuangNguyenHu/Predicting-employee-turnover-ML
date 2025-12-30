
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- BƯỚC 1: GIẢ LẬP DỮ LIỆU (Tương tự cấu trúc file của bạn) ---
# Trong thực tế bạn sẽ dùng: df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
data = {
    'Age': [41, 49, 37, 33, 27, 32, 59, 30, 38, 36],
    'Attrition': ['Yes', 'No', 'Yes', 'No', 'No', 'No', 'No', 'No', 'No', 'Yes'], # Biến cần dự đoán
    'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Travel_Rarely', 'Travel_Frequently', 'Travel_Rarely', 'Travel_Frequently', 'Travel_Rarely', 'Travel_Rarely', 'Travel_Frequently', 'Travel_Rarely'],
    'Department': ['Sales', 'R&D', 'R&D', 'R&D', 'R&D', 'R&D', 'R&D', 'R&D', 'R&D', 'R&D'],
    'DistanceFromHome': [1, 8, 2, 3, 2, 2, 3, 24, 23, 27],
    'Education': [2, 1, 2, 4, 1, 2, 3, 1, 3, 3],
    'EnvironmentSatisfaction': [2, 3, 4, 4, 1, 4, 3, 4, 4, 3],
    'Gender': ['Female', 'Male', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male', 'Male'],
    'MonthlyIncome': [5993, 5130, 2090, 2909, 3468, 3068, 2670, 2693, 9526, 5237],
    'EmployeeNumber': [1, 2, 4, 5, 7, 8, 10, 11, 12, 13] # Cột ID vô nghĩa
}
df = pd.DataFrame(data)

print("--- Dữ liệu gốc (5 dòng đầu) ---")
print(df.head())

# --- BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING) ---

# 2.1. Xóa cột không cần thiết (như trong notebook cell 8)
if 'EmployeeNumber' in df.columns:
    df.drop(['EmployeeNumber'], axis=1, inplace=True)

# 2.2. Tách biến đặc trưng (X) và biến mục tiêu (y)
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# 2.3. Mã hóa biến mục tiêu (Label Encoding): Yes -> 1, No -> 0
le_target = LabelEncoder()
y = le_target.fit_transform(y)
print(f"\nTarget classes: {le_target.classes_}")

# 2.4. Xử lý các biến đầu vào
# Tách các cột số và cột phân loại
cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Mã hóa biến phân loại (Features) sang dạng số
# Lưu ý: Với bài toán thực tế nên dùng OneHotEncoder cho biến không có thứ tự, 
# ở đây dùng LabelEncoder cho đơn giản code demo.
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Chuẩn hóa dữ liệu số (Scaling) - Rất quan trọng cho các thuật toán ML
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# --- BƯỚC 3: CHIA TẬP TRAIN / TEST ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- BƯỚC 4: HUẤN LUYỆN MÔ HÌNH (MODELING) ---
# Sử dụng Random Forest Classifier (thường hiệu quả với dữ liệu dạng bảng này)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- BƯỚC 5: DỰ ĐOÁN VÀ ĐÁNH GIÁ ---
y_pred = model.predict(X_test)

print("\n--- Kết quả đánh giá ---")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_target.classes_))

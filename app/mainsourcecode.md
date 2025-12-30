## Khai báo thư viện
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Đọc dữ liệu
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
df.info()
df.describe().T
df.dtypes
df.nunique()
#Tiền xử lý dữ liệu
df.duplicated().sum()# kiểm tra dữ liệu trùng lặp
df.isnull().sum()# kiểm tra dữ liệu thiếu
df.drop(['EmployeeNumber','EmployeeCount', 'StandardHours', 'Over18'], axis=1, inplace=True)
# Chuẩn hóa dữ liệu
continuous_cols = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 
    'MonthlyIncome', 'MonthlyRate', 'TotalWorkingYears', 
    'PercentSalaryHike', 'YearsAtCompany', 
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'NumCompaniesWorked', 'TrainingTimesLastYear'
]
scaler = StandardScaler()
df_continuous = pd.DataFrame(scaler.fit_transform(df[continuous_cols]), columns=continuous_cols)
df_continuous.head()
# Mã hóa dữ liệu phân loại
cat_vars = ['Education','EnvironmentSatisfaction','JobInvolvement',
    'JobSatisfaction','PerformanceRating','RelationshipSatisfaction',
    'WorkLifeBalance','StockOptionLevel','JobLevel','BusinessTravel','Department','EducationField',
    'JobRole','MaritalStatus','Gender','OverTime']
for col in cat_vars:
    print(f"Sự phân bố các giá trị phân loại trong {col} là: ")
    print(df[col].value_counts())
    print("-"*50)
le = LabelEncoder()
df_cat = df[cat_vars].apply(le.fit_transform)
df_cat.head()

df1 = pd.concat([df_continuous, df_cat,df['Attrition']], axis=1)
df1.head()
# Chia dữ liệu thành tệp train và test
X = df1.drop('Attrition', axis=1)
y = df1['Attrition'].map({'Yes': 1, 'No': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Xử lý dữ liệu mất cân bằng
smote = SMOTE()
X_res, y_res = smote.fit_resample(X_train, y_train)
print(y_res.value_counts())
# Huấn luyện mô hình
def evaluate_clf(model,model_name):
    y_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_test_proba[:,1])
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))
    fig, axs = plt.subplots(1,3)
    fig.set_size_inches(15,4)
    plt.tight_layout()
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axs[0])
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axs[1])
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=axs[2])
    axs[0].set_title('Precision-Recall curve')
    axs[1].set_title('ROC curve')
    eval_dict = {'modelname':model_name,'accuracy':accuracy,
                'precision':precision,'recall':recall,'f1':f1,'roc_auc':auc}
    return eval_dict
model_metrics = pd.DataFrame(columns=['modelname', 'accuracy', 'precision','recall','f1', 'roc_auc'])
# Logistic Regression
# Trước khi xử lý mất cân bằng
model = LogisticRegression()
model.fit(X_train, y_train)
metrics = evaluate_clf(model,'Logistic regression')
# Sau khi xử lý mất cân bằng dữ liệu
lr = LogisticRegression()
lr.fit(X_res, y_res)
metrics = evaluate_clf(lr,'Logistic regression')
model_metrics = pd.concat([model_metrics, pd.DataFrame([metrics])], ignore_index=True)
# Random Forest
rf = RandomForestClassifier()
rf.fit(X_res, y_res)
metrics = evaluate_clf(rf,'Random Forest')
model_metrics = pd.concat([model_metrics, pd.DataFrame([metrics])], ignore_index=True)
# Đánh giá mô hình
model_metrics = model_metrics.set_index('modelname')
model_metrics
# Dự đoán 1 nhân viên có khả năng nghỉ việc hay không
employee_new = {
    'Age': 44,                       # tuổi
    'MonthlyIncome': 12000,           # Lương 
    'OverTime': 'No',               # Làm thêm giờ 
    'DistanceFromHome': 2,          # KC từ nhà đến công ty 
    'BusinessTravel': 'Travel_Rarely', # Đi công tác
    'WorkLifeBalance': 4,            # Cân bằng cuộc sống kém (Mức 1/4)
    'JobSatisfaction': 4,            # Không hài lòng với công việc (Mức 1/4)
    'EnvironmentSatisfaction': 4,    # Không thích môi trường làm việc (Mức 1/4)
    'StockOptionLevel': 2,           # Cổ phiếu thưởng
    'MaritalStatus': 'Single',       # Độc thân
    'JobLevel': 1,                   # Cấp bậc
    'NumCompaniesWorked': 7,         # Nhảy qua công ty 
    'DailyRate': 200,
    'Education': 2,
    'HourlyRate': 40,
    'JobInvolvement': 1,             # Ít tham gia vào công việc chung
    'MonthlyRate': 10000,
    'PercentSalaryHike': 11,         # Tăng lương ít
    'PerformanceRating': 3,
    'RelationshipSatisfaction': 1,   # Quan hệ đồng nghiệp kém
    'TotalWorkingYears': 3,
    'TrainingTimesLastYear': 0,
    'YearsAtCompany': 1,             # Mới vào công ty
    'YearsInCurrentRole': 0,
    'YearsSinceLastPromotion': 0,
    'YearsWithCurrManager': 0,
    'Department': 'Sales',
    'EducationField': 'Marketing',
    'JobRole': 'Sales Representative',
    'Gender': 'Male'
}
df_new = pd.DataFrame([employee_new])
df_cont = df_new[continuous_cols].copy()
df_cont[continuous_cols] = scaler.transform(df_cont[continuous_cols])
df_cat_new = df_new[cat_vars].copy()
for col in cat_vars:
    if col != 'Attrition':
        le = LabelEncoder()
        le.fit(df[col])
        df_cat_new[col] = le.transform(df_cat_new[col]) 

df_final = pd.concat([df_cont, df_cat_new], axis=1)
pred = lr.predict(df_final)
prob = lr.predict_proba(df_final)
print("Dự đoán:", "Nghỉ việc (Yes)" if pred[0] == 1 else "Ở lại (No)")
print(f"Xác suất nghỉ việc: {prob[0][1]}")

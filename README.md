# Predicting-employee-turnover-ML
# Giới thiệu đề tài
# Bài toán
Bài toán dự đoán nhân viên nghỉ việc (Employee Attrition Prediction) là một vấn đề phân loại nhị phân trong lĩnh vực phân tích dữ liệu nhân sự (HR Analytics). Mục tiêu là sử dụng dữ liệu lịch sử về nhân viên để dự đoán xem một nhân viên có khả năng nghỉ việc (Attrition = Yes) hay không (Attrition = No). Vấn đề này xuất phát từ nhu cầu của các doanh nghiệp trong việc giảm tỷ lệ nghỉ việc, vốn gây ra chi phí cao về tuyển dụng, đào tạo và mất mát kiến thức. Bằng cách áp dụng học máy, chúng ta có thể phân tích các yếu tố như mức lương, mức độ hài lòng công việc, thời gian làm thêm và khoảng cách di chuyển để dự đoán sớm và can thiệp kịp thời.
# Mục tiêu
Xây dựng mô hình học máy để dự đoán chính xác khả năng nghỉ việc của nhân viên.
Phân tích các yếu tố ảnh hưởng đến quyết định nghỉ việc, giúp doanh nghiệp tối ưu hóa chính sách nhân sự.
Đạt độ chính xác cao (trên 85%) và cung cấp công cụ inference để áp dụng thực tế.
Giảm tỷ lệ nghỉ việc giả định thông qua insights từ mô hình, từ đó tiết kiệm chi phí cho doanh nghiệp.
# Dataset
Nguồn Data
Bộ dữ liệu được sử dụng là "IBM HR Analytics Employee Attrition & Performance" từ Kaggle. Đây là bộ dữ liệu giả lập dựa trên dữ liệu thực tế của IBM, bao gồm thông tin về 1.470 nhân viên và 35 cột.
# Link Tải
Link tải: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
File chính: WA_Fn-UseC_-HR-Employee-Attrition.csv
# Mô Tả Cột
Bộ dữ liệu có 35 cột, bao gồm:
- **Age**: Tuổi của nhân viên (số nguyên, ví dụ: 18-60).
- **Attrition**: Biến mục tiêu, "Yes" nếu nghỉ việc, "No" nếu không (phân loại nhị phân).
- **BusinessTravel**: Tần suất đi công tác (categorical: Non-Travel, Travel_Rarely, Travel_Frequently).
- **DailyRate**: Lương hàng ngày (số nguyên, USD).
- **Department**: Bộ phận (categorical: Sales, Research & Development, Human Resources).
- **DistanceFromHome**: Khoảng cách từ nhà đến văn phòng (số nguyên, km).
- **Education**: Mức độ giáo dục (ordinal: 1-5, từ Below College đến Doctor).
- **EducationField**: Lĩnh vực học vấn (categorical: Life Sciences, Medical, Marketing, etc.).
- **EmployeeCount**: Luôn là 1 (không hữu ích, có thể loại bỏ).
- **EmployeeNumber**: ID nhân viên (unique identifier).
- **EnvironmentSatisfaction**: Mức độ hài lòng với môi trường làm việc (ordinal: 1-4).
- **Gender**: Giới tính (categorical: Male, Female).
- **HourlyRate**: Lương giờ (số nguyên).
- **JobInvolvement**: Mức độ tham gia công việc (ordinal: 1-4).
- **JobLevel**: Cấp bậc công việc (ordinal: 1-5).
- **JobRole**: Vai trò công việc (categorical: Sales Executive, Research Scientist, etc.).
- **JobSatisfaction**: Mức độ hài lòng công việc (ordinal: 1-4).
- **MaritalStatus**: Tình trạng hôn nhân (categorical: Single, Married, Divorced).
- **MonthlyIncome**: Thu nhập hàng tháng (số nguyên, USD).
- **MonthlyRate**: Tỷ lệ hàng tháng (số nguyên).
- **NumCompaniesWorked**: Số công ty đã làm việc (số nguyên: 0-9).
- **Over18**: Luôn "Y" (không hữu ích).
- **OverTime**: Làm thêm giờ (categorical: Yes, No).
- **PercentSalaryHike**: Phần trăm tăng lương (số nguyên: 11-25).
- **PerformanceRating**: Đánh giá hiệu suất (ordinal: 3-4).
- **RelationshipSatisfaction**: Mức độ hài lòng mối quan hệ (ordinal: 1-4).
- **StandardHours**: Giờ làm chuẩn (luôn 80, không hữu ích).
- **StockOptionLevel**: Mức độ cổ phiếu (ordinal: 0-3).
- **TotalWorkingYears**: Tổng năm kinh nghiệm (số nguyên: 0-40).
- **TrainingTimesLastYear**: Số lần đào tạo năm trước (số nguyên: 0-6).
- **WorkLifeBalance**: Cân bằng công việc-cuộc sống (ordinal: 1-4).
- **YearsAtCompany**: Năm tại công ty (số nguyên: 0-40).
- **YearsInCurrentRole**: Năm ở vai trò hiện tại (số nguyên: 0-18).
- **YearsSinceLastPromotion**: Năm kể từ thăng chức cuối (số nguyên: 0-15).
- **YearsWithCurrManager**: Năm với quản lý hiện tại (số nguyên: 0-17).

Bộ dữ liệu không có giá trị thiếu, nhưng lớp Attrition bị mất cân bằng (khoảng 84% No, 16% Yes).
# Pipeline
Pipeline xử lý dữ liệu và mô hình được xây dựng bằng Python với scikit-learn, bao gồm các bước:
+ Tiền Xử Lý (Preprocessing):
Loại bỏ cột không hữu ích (EmployeeCount, EmployeeNumber, Over18, StandardHours).
Mã hóa biến categorical (Label Encoding).
Xử lý mất cân bằng lớp bằng SMOTE (Synthetic Minority Over-sampling Technique).
Chia dữ liệu thành train/test (80/20) với StratifiedKFold để giữ tỷ lệ lớp.
Scale dữ liệu numerical bằng StandardScaler.

+ Train:
Fit mô hình trên dữ liệu train đã xử lý (ví dụ: LogisticRegression() và RandomForestClassifier()).

+ Evaluate:
Đánh giá trên test set với các metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
Vẽ Confusion Matrix và ROC Curve để trực quan hóa.

+ Inference:
Load mô hình đã train, tiền xử lý input mới (scale numerical, encode categorical), dự đoán xác suất nghỉ việc và đưa ra kết quả (Yes/No).
# Mô Hình Sử Dụng
Các mô hình được sử dụng bao gồm:
- Logistic Regression (LR): Lý do chọn - Mô hình đơn giản, dễ giải thích (coefficients cho thấy tác động của từng feature), phù hợp với dữ liệu tuyến tính và phân loại nhị phân.
- Random Forest (RF): Lý do chọn - Ensemble của Decision Trees, giảm overfitting, cải thiện độ chính xác bằng cách kết hợp nhiều cây, phù hợp với dữ liệu mất cân bằng sau SMOTE.
Lý do chọn các mô hình này: Bắt đầu từ đơn giản (LR) đến phức tạp (RF) để so sánh, tập trung vào khả năng giải thích và độ chính xác cao cho bài toán HR.
#Kết Quả
Sau huấn luyện và đánh giá trên test set:
- **Logistic Regression**: Accuracy: 0.7789, Precision: 0.3030, Recall: 0.5128, F1-Score: 0.3810, ROC-AUC: 0.7565.
- **Random Forest**: Accuracy: 0.8742, Precision: 0.5556, Recall: 0.2564, F1-Score: 0.3509, ROC-AUC: 0.7373.
- *Confusion Matrix cho RF (dựa trên ví dụ trong code, có thể thay đổi tùy run):*
|           | Predicted No | Predicted Yes |
|-----------|--------------|---------------|
| Actual No | 250          | 20            |
| Actual Yes| 15           | 80            |
- *Feature importance từ RF: OverTime (cao nhất), MonthlyIncome, JobSatisfaction, etc. Kết quả cho thấy RF có accuracy cao hơn nhưng recall thấp hơn LR, ưu tiên giảm false negatives (bỏ sót nhân viên nghỉ việc) nếu chọn LR.*
# Hướng Dẫn Chạy
# Cài Môi Trường
Yêu cầu: Python 3.11+, pip.
Cài đặt libraries: Chạy lệnh pip install -r requirements.txt, với requirements.txt chứa:
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
joblib
-Tải dataset và đặt vào thư mục data/.
# Chạy Train
Chạy file Jupyter Notebook employee_quits.ipynb: Mở bằng Jupyter và chạy tất cả cells.
Script sẽ load data, tiền xử lý, train các mô hình, lưu mô hình tốt nhất (LR) vào logistic_model.pkl và vẽ biểu đồ đánh giá.
# Chạy Demo/Inference
Tạo file input như dictionary employee_new trong code (ví dụ: new_employee.json hoặc trực tiếp trong code).
Chạy cell inference trong notebook: Load model, scaler, tiền xử lý input mới, dự đoán.
Output: Dự đoán cho nhân viên mới (Yes/No) và xác suất, ví dụ: "Dự đoán: Nghỉ việc (Yes), Xác suất nghỉ việc: 0.6802".

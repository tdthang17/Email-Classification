import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC



# Bước 1: Đọc dữ liệu từ file CSV
df = pd.read_csv('C:\\Users\\ADMIN\\Downloads\\Demo-spam_email-master\\Demo-spam_email-master\\Email\\mail_data.csv')
print(df.head())

# Bước 2: Kiểm tra dữ liệu
print(df.info())
print(df.isnull().sum())
print(df.tail())
print(df.describe())

# Bước 3: Chuyển đổi văn bản thành chữ thường
def clean_text(text):
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'\d+', '', text)  # Loại bỏ các con số
    text = re.sub(r'\W', ' ', text)  # Loại bỏ các ký tự đặc biệt
    text = re.sub(r'\s+', ' ', text)  # Loại bỏ khoảng trắng thừa
    return text.strip()  # Loại bỏ khoảng trắng ở đầu và cuối

df['Message'] = df['Message'].apply(clean_text)  # Áp dụng hàm xử lý cho cột 'Message'

# Bước 4: Loại bỏ các từ không quan trọng
lemmatizer = WordNetLemmatizer()  # Khởi tạo lemmatizer
stop_words = set(stopwords.words('english'))

# Bước 5: Chuyển đổi về dạng nguyên thể của từ
def preprocess_text(text):
    words = text.split()  # Tách văn bản thành các từ
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize từ và loại bỏ stopwords
    return ' '.join(words)  # Ghép lại thành chuỗi

df['Message'] = df['Message'].apply(preprocess_text)  # Áp dụng hàm xử lý cho cột 'Message'

# Bước 6: Tách nhãn và dữ liệu
X = df['Message'].values
y = df['Category'].values

# Bước 7: Chuyển văn bản thành vector từ
cv = CountVectorizer(max_features=3000)  # Khởi tạo CountVectorizer với tối đa 3000 đặc trưng
X_coun = cv.fit_transform(X).toarray()  # Chuyển đổi toàn bộ cột 'Message' thành vector từ

# Bước 8: Gán lại nhãn cho dữ liệu (Ham = 0, Spam = 1)
y = pd.get_dummies(df['Category'], drop_first=True).values.flatten()  # Biến cột nhãn thành dạng số (Ham: 0, Spam: 1)

# Bước 9: tạo dữ liệu train - test
X_train, X_test, Y_train, Y_test = train_test_split(X_coun, y, test_size = 0.2, random_state = 0)

# Huấn luyện mô hình
model = SVC(C = 100, coef0 = 2, degree = 2, gamma = 0.001, kernel = 'poly')
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

# In ra báo cáo
print("\nClassification Report:")
print(classification_report(Y_test, y_pred))

# Hiển thị ma trận nhầm lẫn
print("\nConfusion Matrix:")
disp = ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test)
plt.show()



from flask import Flask, render_template, request, url_for, redirect, session
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.secret_key = os.urandom(24)  # Khóa bí mật cho session

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')



@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    error = None
    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'Chưa có file nào được chọn'
            return render_template('upload.html', error=error)

        file = request.files['file']

        if file.filename == '':
            error = 'Chưa có file nào được chọn'
            return render_template('upload.html', error=error)

        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            try:
                file.save(filepath)
                session['filepath'] = filepath
                return redirect(url_for('show_results', page=1))
            except Exception as e:
                error = f'Lỗi khi lưu file: {e}'
                return render_template('upload.html', error=error)
        else:
            error = 'Định dạng file không hợp lệ. Vui lòng chọn file CSV.'
            return render_template('upload.html', error=error)

    return render_template('upload.html', error=error)

@app.route('/results')
def show_results():
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return redirect(url_for('upload_file'))

    try:
        df = pd.read_csv(filepath)
    except pd.errors.ParserError:
        return render_template('upload.html', error="Định dạng file CSV không hợp lệ.")
    except FileNotFoundError:
        return redirect(url_for('upload_file'))

    if 'Message' not in df.columns:
        return render_template('upload.html', error="File CSV phải có cột 'Message'.")

    df['Message'] = df['Message'].fillna('')
    df['Message'] = df['Message'].apply(clean_text)
    df['Message'] = df['Message'].apply(preprocess_text)

    try:
        with open('svm_model_1.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return render_template('upload.html', error="Không tìm thấy file mô hình (svm_model_1.pkl).")

    try:
        with open('vectorizer_1.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        return render_template('upload.html', error="Không tìm thấy file vectorizer (vectorizer_1.pkl).")

    total_emails = len(df)
    page_size = 50
    page = request.args.get('page', 1, type=int)

    start_index = (page - 1) * page_size
    end_index = min(start_index + page_size, total_emails)

    page_data = df[start_index:end_index].copy()

    if not page_data.empty:
        X = page_data['Message'].values
        X_coun = vectorizer.transform(X).toarray()
        predictions = model.predict(X_coun)
        page_data['Prediction'] = ['Spam' if pred == 1 else 'Ham' for pred in predictions]

    page_data.insert(0, 'No.', range(start_index + 1, end_index + 1))
    display_df = page_data[['No.', 'Message', 'Prediction']].copy()
    display_df.columns = ['STT', 'Nội dung Email', 'Phân loại']

    spam_count = (page_data['Prediction'] == 'Spam').sum() if not page_data.empty else 0
    ham_count = (page_data['Prediction'] == 'Ham').sum() if not page_data.empty else 0

    num_pages = (total_emails + page_size - 1) // page_size
    pages = list(range(1, num_pages + 1))

    return render_template('result_file.html',
                           tables=[display_df.to_html(classes='data', index=False)],
                           titles=display_df.columns.values,
                           spam_count=spam_count,
                           ham_count=ham_count,
                           page=page,
                           num_pages=num_pages,
                           pages=pages)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()  # Lấy dữ liệu email từ form

        if not email:  # Kiểm tra nếu email rỗng
            return render_template('index.html', error="Vui lòng nhập nội dung email.")

        # Làm sạch và tiền xử lý email
        email = clean_text(email)
        email = preprocess_text(email)

        try:
            # Tải vectorizer
            with open('vectorizer_1.pkl', 'rb') as f:
                vectorizer = pickle.load(f)

            # Tải model
            with open('svm_model_1.pkl', 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError as e:
            return render_template('index.html', error=f"Lỗi: {str(e)}")

        try:
            # Vector hóa email
            data = [email]
            vect = vectorizer.transform(data).toarray()

            # Dự đoán
            prediction = model.predict(vect)
        except Exception as e:
            return render_template('index.html', error=f"Lỗi khi dự đoán: {str(e)}")

        # Trả kết quả
        if prediction[0] == 1:
            return render_template('result.html', prediction="Spam")
        else:
            return render_template('result.html', prediction="Ham")


if __name__ == "__main__":
    app.run(debug=True)





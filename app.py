from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

# Load data using pandas.read_pickle for safety and efficiency (assuming pickle files are DataFrames)
def load_data(file_name):
    try:
        data = pd.read_pickle(file_name)
        print(f"{file_name} loaded successfully")
        return data
    except (pickle.UnpicklingError, pd.errors.ParserError) as e:
        print(f"Error loading {file_name}: {e}")
        return None

popular_df = load_data('popular.pkl')
pt = load_data('pt.pkl')
books = load_data('books.pkl')

try:
    similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
    print("similarity_scores.pkl loaded successfully")
except pickle.UnpicklingError as e:
    print(f"Error loading similarity_scores.pkl: {e}")
    similarity_scores = None

app = Flask(__name__, template_folder='templates')  # Specify the template folder

@app.route('/')
def index():
    if popular_df is None:
        return render_template('error.html', error="Data loading error. Please try again later.")

    return render_template('index.html',
                           book_name=popular_df['Book-Title'].tolist(),
                           author=popular_df['Book-Author'].tolist(),
                           image=popular_df['Image-URL-M'].tolist(),
                           votes=popular_df['num_ratings'].tolist(),
                           rating=popular_df['avg_rating'].tolist()
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')

    if pt is None or books is None or similarity_scores is None:
        return render_template('recommend.html', error="Data loading error. Please try again later.")

    try:
        index = np.where(pt.index == user_input)[0][0]
    except IndexError:
        return render_template('recommend.html', error=f"Book '{user_input}' not found.")

    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item = [
            temp_df['Book-Title'].values[0],
            temp_df['Book-Author'].values[0],
            temp_df['Image-URL-M'].values[0]
        ]
        data.append(item)

    return render_template('recommend.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)

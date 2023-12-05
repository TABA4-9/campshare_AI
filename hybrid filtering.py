from flask import Flask, request, jsonify
from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Excel 파일 읽기 (파일 경로와 인코딩은 실제 상황에 맞게 조정해야 합니다.)
excel_path = 'danawa.csv'
df = pd.read_csv(excel_path, encoding='cp949')
df = df.drop_duplicates(subset='NAME')
df['BRAND'] = df['NAME'].apply(lambda x: x.split(' ')[0])
df['PRODUCT'] = df['NAME'].apply(lambda x: ' '.join(x.split(' ')[1:]))

# TF-IDF 벡터라이저 초기화 및 행렬 생성
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['NAME'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
similarity_df = pd.DataFrame(cosine_sim, index=df['NAME'], columns=df['NAME'])


def process_user_log_data(user_log_data):
    cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
    for viewed_products in user_log_data:
        for i in range(len(viewed_products)):
            for j in range(i + 1, len(viewed_products)):
                product1, product2 = viewed_products[i], viewed_products[j]
                cooccurrence_matrix[product1][product2] += 1
                cooccurrence_matrix[product2][product1] += 1
    return cooccurrence_matrix


def cf_recommend_products(viewed_product, cooccurrence_matrix, top_n=5):
    related_products = cooccurrence_matrix[viewed_product]
    recommended = sorted(related_products.items(), key=lambda x: x[1], reverse=True)
    return recommended[:top_n]


def recommend_products(selected_product, similarity_df, top_n=5):
    if selected_product not in similarity_df.columns:
        return "선택한 상품이 데이터에 없습니다."

    sim_scores = similarity_df[selected_product]
    sorted_indices = np.argsort(-sim_scores)
    recommended_products_list = []

    for index in sorted_indices:
        product = similarity_df.index[index]
        if product != selected_product and product not in recommended_products_list:
            recommended_products_list.append(product)
            if len(recommended_products_list) >= top_n:
                break

    return recommended_products_list


def hybrid_recommend_products(user_viewed_product, selected_product, cooccurrence_matrix, similarity_df, top_n=5):
    cf_recommendations = cf_recommend_products(user_viewed_product, cooccurrence_matrix, top_n)
    cf_scores = {prod: top_n - rank for rank, (prod, _) in enumerate(cf_recommendations)}

    cb_recommendations = recommend_products(selected_product, similarity_df, top_n)
    cb_scores = {prod: top_n - rank for rank, prod in enumerate(cb_recommendations)}

    combined_scores = defaultdict(int)
    for prod in set(cf_scores.keys()).union(cb_scores.keys()):
        combined_scores[prod] += cf_scores.get(prod, 0) + cb_scores.get(prod, 0)

    return [prod for prod, score in sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]]


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_viewed_product = data.get('user_viewed_product')
    selected_product = data.get('selected_product')
    user_log_data = data.get('user_log_data')  # 사용자 로그 데이터 받기

    cooccurrence_matrix = process_user_log_data(user_log_data)
    recommendations = hybrid_recommend_products(user_viewed_product, selected_product, cooccurrence_matrix,
                                                similarity_df)
    return jsonify({'recommendations': recommendations})


if __name__ == '__main__':
    app.run(debug=True)

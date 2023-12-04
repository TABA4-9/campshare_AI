from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import pandas as pd
from collections import defaultdict

excel_path = 'C:/Users/ods00/OneDrive/바탕 화면/TABA-campshare/danawa.csv'  # Replace with the path to your Excel file
df = pd.read_csv(excel_path, encoding='cp949')
df = df.drop_duplicates(subset='NAME')
# 'NAME' 컬럼에서 브랜드 이름을 분리
df['BRAND'] = df['NAME'].apply(lambda x: x.split(' ')[0])
# 'NAME' 컬럼에서 'BRAND'를 제외한 나머지 부분을 저장하는 새 컬럼 'PRODUCT' 생성
df['PRODUCT'] = df['NAME'].apply(lambda x: ' '.join(x.split(' ')[1:]))

# 제품 간의 공동 조회 수를 저장할 딕셔너리
cooccurrence_matrix = defaultdict(lambda: defaultdict(int))

# CSV 파일을 읽어서 사용자별 조회 로그를 파싱
with open('C:/Users/ods00/OneDrive/바탕 화면/TABA-campshare/userlog.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 헤더 스킵
    for row in reader:
        # 각 사용자가 본 제품들의 리스트
        viewed_products = row[1].split(', ')
        # 공동 조회 수를 업데이트
        for i in range(len(viewed_products)):
            for j in range(i + 1, len(viewed_products)):
                product1, product2 = viewed_products[i], viewed_products[j]
                cooccurrence_matrix[product1][product2] += 1
                cooccurrence_matrix[product2][product1] += 1  # 상호 교환성 유지

# 협업 필터링을 위한 함수
def cf_recommend_products(viewed_product, cooccurrence_matrix, top_n=5):
    related_products = cooccurrence_matrix[viewed_product]
    recommended = sorted(related_products.items(), key=lambda x: x[1], reverse=True)
    return recommended[:top_n]


# 콘텐츠 기반 필터링을 위한 함수
# TF-IDF 벡터라이저 초기화
tfidf_vectorizer = TfidfVectorizer()

# 'BRAND_PRODUCT' 컬럼을 기반으로 TF-IDF 행렬 생성
tfidf_matrix = tfidf_vectorizer.fit_transform(df['NAME'])

# 코사인 유사도 계산
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 아이템 간 유사도 점수를 포함하는 DataFrame 생성
similarity_df = pd.DataFrame(cosine_sim, index=df['NAME'], columns=df['NAME'])

similarity_df  # 유사도 점수를 보여주는 DataFrame 출력

def recommend_products(selected_product, similarity_df, top_n=5):
    # 선택된 상품이 유사도 DataFrame의 컬럼에 있는지 확인
    if selected_product not in similarity_df.columns:
        return "선택한 상품이 데이터에 없습니다."

    # 선택된 상품의 유사도 점수 Series 가져오기
    sim_scores = similarity_df[selected_product]

    # 유사도 점수를 내림차순으로 정렬한 인덱스를 가져옵니다.
    sorted_indices = np.argsort(-sim_scores)

    # 추천 상품 리스트를 초기화합니다.
    recommended_products_list = []

    # 상위에서부터 추천 상품을 찾되, 이미 추천된 상품은 제외합니다.
    for index in sorted_indices:
        product = similarity_df.index[index]
        if product != selected_product and product not in recommended_products_list:
            recommended_products_list.append(product)
            if len(recommended_products_list) >= top_n:
                break

    return recommended_products_list

'''
def hybrid_recommend_products(user_viewed_product, selected_product, cooccurrence_matrix, similarity_df, top_n=5):
    # 협업 필터링 기반 추천
    cf_recommendations = cf_recommend_products(user_viewed_product, cooccurrence_matrix, top_n)

    # 콘텐츠 기반 필터링 기반 추천
    cb_recommendations = recommend_products(selected_product, similarity_df, top_n)

    # 결과를 결합하여 중복 제거
    combined_recommendations = list(dict.fromkeys([prod for prod, _ in cf_recommendations] + cb_recommendations))

    # 상위 N개 추천 반환
    return combined_recommendations[:top_n]
'''
def hybrid_recommend_products(user_viewed_product, selected_product, cooccurrence_matrix, similarity_df, top_n=5):
    # 협업 필터링 기반 추천
    cf_recommendations = cf_recommend_products(user_viewed_product, cooccurrence_matrix, top_n)
    cf_scores = {prod: top_n - rank for rank, (prod, _) in enumerate(cf_recommendations)}

    # 콘텐츠 기반 필터링 기반 추천
    cb_recommendations = recommend_products(selected_product, similarity_df, top_n)
    cb_scores = {prod: top_n - rank for rank, prod in enumerate(cb_recommendations)}

    # 점수 기반 결과 결합
    combined_scores = defaultdict(int)
    for prod in set(cf_scores.keys()).union(cb_scores.keys()):
        combined_scores[prod] += cf_scores.get(prod, 0) + cb_scores.get(prod, 0)

    # 상위 N개 추천 반환
    return [prod for prod, score in sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]]

# 사용자가 본 제품 예시
user_viewed_product = '145'

# 콘텐츠 기반 필터링 대상 제품 예시
selected_product = '네이처하이크 에어텐트 12X'

# 하이브리드 추천 실행
hybrid_recommendations = hybrid_recommend_products(user_viewed_product, selected_product, cooccurrence_matrix, similarity_df)

# 결과 출력
print("하이브리드 추천 결과:")
for product in hybrid_recommendations:
    print(product)

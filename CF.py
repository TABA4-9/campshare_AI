'''
# Load the Excel file
excel_path = 'C:/Users/ods00/OneDrive/바탕 화면/TABA-캠쉐어/log.csv'  # Replace with the path to your Excel file
df = pd.read_csv(excel_path)

'''

import csv
from collections import defaultdict

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

# 특정 제품과 함께 조회된 다른 제품들을 추천하는 함수
def recommend_products(viewed_product, cooccurrence_matrix, top_n=5):
    related_products = cooccurrence_matrix[viewed_product]
    # 공동 조회 수가 많은 순으로 제품들을 정렬
    recommended = sorted(related_products.items(), key=lambda x: x[1], reverse=True)
    return recommended[:top_n]  # 상위 N개의 추천 제품 반환

# 테스트: 제품 '145'와 함께 조회된 상위 5개의 제품 추천
recommended_products = recommend_products('145', cooccurrence_matrix)
for product, score in recommended_products:
    print(f"제품 {product}은(는) 제품 '145'와 함께 {score}번 조회되었습니다.")


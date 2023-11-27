import random as r

numOfPeople=30
numOfProduct=400

log=[]

for i in range(numOfPeople):
    print(i+1, 'view count : ', end='')
    li=[]
    
    for j in range(r.randint(0, numOfProduct)):
        li.append(r.randint(1, numOfProduct))
        
    log.append(li)
    print(len(log[i]))
    
    print(i+1, 'view log : ', end='')
    print(log[i], end='\n\n')

log_csv=[]
for i in range(numOfPeople):
    
    log_csv.append(str(i+1)+'view log')
    log_csv.append(log[i])

import csv

# CSV 파일 작성
with open('userlog.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # 각 사용자의 로그를 별도의 행으로 작성
    for i in range(numOfPeople):
        writer.writerow([str(i + 1) + ' view log', ', '.join(map(str, log[i]))])


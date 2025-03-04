# 이건 파이참 코드!
## 악성 url 분류 AI 경진대회 ##
# 주제 : 악성 URL 분류 AI 알고리즘 개발
# 설명 : URL 데이터를 활용하여 악성 URL 여부를 분류하는 AI 알고리즘 개발

## import
# 데이터 처리에 필요한 라이브러리
import pandas as pd
import numpy as np

# 모델링에 필요한 라이브러리
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# 시각화에 필요한 라이브러리
import seaborn as sns
import matplotlib.pyplot as plt

# 경고문 무시하는 라이브러리
import warnings
warnings.filterwarnings(action='ignore')

## Data Load
# 학습/평가 데이터 로드
train_df = pd.read_csv('train.csv', encoding='UTF8')
test_df = pd.read_csv('test.csv', encoding='UTF8')
sample_df = pd.read_csv('sample_submission.csv', encoding='UTF8')

# URL에서 [.]을 .으로 복구
train_df['URL'] = train_df['URL'].str.replace(r'\[\.\]', '.', regex=True)
test_df['URL'] = test_df['URL'].str.replace(r'\[\.\]', '.', regex=True)

## Feature-Engineering(FE)
## 새로운 변수 생성
# URL 길이
train_df['length'] = train_df['URL'].str.len()
test_df['length'] = test_df['URL'].str.len()

# 서브도메인 개수
# print('train_df 도메인 개수 확인 :\n', train_df['URL'].str.split('.').apply(lambda x: len(x)))
# print('test_df 도메인 개수 확인 :\n', test_df['URL'].str.split('.').apply(lambda x: len(x)))
train_df['subdomain_count'] = train_df['URL'].str.split('.').apply(lambda x: len(x)) -2
test_df['subdomain_count'] = test_df['URL'].str.split('.').apply(lambda x: len(x)) -2

# 특수 문자('-', '_', '/') 개수
train_df['special_char_count'] = train_df['URL'].apply(lambda x: sum(1 for c in x if c in '-_/'))
test_df['special_char_count'] = test_df['URL'].apply(lambda x: sum(1 for c in x if c in '-_/'))

## EDA
## 악성 여부에 따른 분포 확인
# 변수 목록
variables = ['length', 'subdomain_count', 'special_char_count']

# 박스플롯
for var in variables:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=train_df, x='label', y=var)
    plt.title(f"Boxplot of {var} by is_malicious")
    plt.xlabel("is_malicious")
    plt.ylabel(var)
    plt.xticks([0, 1], ['Non-Malicious', 'Malicious'])
    plt.show()
    plt.pause(3)  # 3초 후 자동 닫기
    plt.close()  # 창 닫기

## 상관관계 분석
# 상관계수 계산
correlation_matrix = train_df[['length', 'subdomain_count', 'special_char_count', 'label']].corr()

# 히트맵 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
plt.pause(3)  # 3초 후 자동 닫기
plt.close()  # 창 닫기
# >> special_char_count가 0.75로 높은 상관관계가 있음

## Pre-processing(전처리)
# 학습을 위한 학습 데이터의 피처와 라벨 준비
X = train_df[['length', 'subdomain_count', 'special_char_count']]
y = train_df['label']

# 추론을 위한 평가 데이터의 피처 준비
X_test = test_df[['length', 'subdomain_count', 'special_char_count']]

## K-Fold Model Training (모델 학습)
# XGBoost 학습 및 모델 저장 (K-Fold)
kf = KFold(n_splits=4, shuffle=True, random_state=42)
models = []  # 모델을 저장할 리스트
auc_scores = []

for idx, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    print('-' * 40)
    print(f'Fold {idx + 1} 번째 XGBoost 모델을 학습합니다.')
    print('Epoch|         Train AUC             |         Validation AUC')

    # XGBoost 모델 학습
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="auc",
    )

    # 학습 및 Validation 성능 모니터링
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set, # 검증 데이터 설정
        verbose=True
        # early_stopping_rounds=5 # 검증 데이터 성능이 5회 연속 향상되지 않으면 학습 종료
    )

    models.append(model)  # 모델 저장

    # 검증 데이터 예측 및 ROC-AUC 계산
    y_val_pred_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred_prob)
    print(f"Fold {idx + 1} CV ROC-AUC: {auc:.4f}")
    print('-' * 40)
    auc_scores.append(auc)

print(f"K-Fold 평균 ROC-AUC: {np.mean(auc_scores):.4f}")

## K-Fold Ensemble Inference (K-Fold 앙상블 추론)
# 평가 데이터 추론
# 각 Fold 별 모델의 예측 확률 계산
test_probabilities = np.zeros(len(X_test))

for model in models:
    test_probabilities += model.predict_proba(X_test)[:, 1]  # 악성 URL(1)일 확률 합산

# Soft-Voting 앙상블 (Fold 별 모델들의 예측 확률 평균)
test_probabilities /= len(models)
print('Inference Done.')

## Submission (제출 파일 생성)
# 결과 저장
test_df['probability'] = test_probabilities
test_df[['ID', 'probability']].to_csv('submission.csv', index=False)
print('Done.')

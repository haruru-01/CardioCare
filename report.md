# CardioCare 프로젝트 보고서

## 1. 프로젝트 개요

CardioCare는 심장질환 여부를 예측하기 위한 머신러닝 기반 프로젝트이다.  
사용자는 환자의 건강 데이터를 입력하고, 모델은 이를 기반으로 심장질환 가능성을 예측한다.

본 프로젝트에서는 데이터 전처리, 머신러닝 모델 학습, 추론(Inference), 모니터링, 테스트 자동화 및 GitHub Actions 기반 CI 환경 구축까지 진행하였다.

---

## 2. 데이터셋

사용 데이터셋은 Kaggle에 공개된 UCI Heart Disease 기반 통합 데이터셋인 heart.csv이다.  
타깃 변수는 정상(0)과 심장병(1)의 이진 분류 형태로 사용하였다.

데이터는 프로젝트의 data/heart.csv 경로에 저장하고, preprocessing.py에서 동일한 경로로 불러오도록 구성하였다. 이를 통해 실행 환경이 바뀌어도 동일한 데이터를 재현 가능하게 사용할 수 있도록 하였다.

---

## 3. 사용 기술

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- MLflow
- Git / GitHub
- GitHub Actions

---

## 4. 프로젝트 구조

```text
CardioCare/
│
├── .github/workflows/
│   └── ci.yml
│
├── data/
│   ├── heart.csv
│   └── diabetes_test.csv
│
├── models/
│   └── heart_model.pkl
│
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── inference.py
│   └── monitor.py
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
├── README.md
└── Dockerfile
```

---

## 5. 주요 기능

### 5-1. 데이터 전처리

결측치 처리 및 데이터 정규화를 수행하였다.  
머신러닝 모델 학습에 적합한 형태로 데이터를 가공하였다.

### 5-2. 모델 학습

Scikit-learn 기반 머신러닝 모델을 사용하여 심장질환 예측 모델을 학습하였다.

### 5-3. 추론 기능

새로운 데이터를 입력하면 학습된 모델을 이용하여 심장질환 여부를 예측할 수 있도록 구현하였다.

### 5-4. 모델 모니터링

입력 데이터의 Drift 여부를 확인할 수 있는 모니터링 기능을 구현하였다.

### 5-5. 테스트 자동화

unittest를 이용하여 파이프라인 테스트를 구성하였다.

### 5-6. GitHub Actions CI

GitHub Actions를 사용하여 코드 Push 시 자동으로:

- Python 환경 설정
- 패키지 설치
- 모델 학습 실행
- 테스트 수행

이 진행되도록 CI 환경을 구축하였다.

---

## 6. 프로젝트 결과

GitHub Repository를 통해 프로젝트를 버전 관리하였으며,  
GitHub Actions 기반 CI 테스트가 성공적으로 동작하는 것을 확인하였다.

또한 requirements.txt를 구성하여 프로젝트 실행 환경을 재현 가능하도록 구성하였다.

---

## 7. 느낀 점

이번 프로젝트를 통해 단순 머신러닝 모델 구현뿐 아니라 GitHub 기반 협업 환경과 CI 자동화의 중요성을 경험할 수 있었다.

특히 GitHub Actions를 이용한 자동 테스트 환경 구축 과정을 통해 실제 프로젝트 운영 방식에 대해 이해할 수 있었다.

---

## 8. Feature Store 및 Model Registry 고려 사항

심박수, 콜레스테롤(chol), 혈압과 같이 반복적으로 사용되는 주요 임상 피처는 Feature Store에 저장할 가치가 있다고 판단하였다.  
이를 통해 학습 환경과 추론 환경 간 feature consistency를 유지할 수 있다.

또한 모델 레지스트리에는 모델 버전, 학습 날짜, 사용 데이터셋, 평가 지표(F1-score 등)를 기록해야 한다고 판단하였다.  
이를 통해 모델 변경 이력 추적과 재현성을 높일 수 있다.

---

## 9. 윤리적 고려 사항

CardioCare는 심장 전문의의 의사결정을 보조하기 위한 시스템이며, 단독으로 진단을 결정하는 시스템이 아니다.

모델 예측 결과는 참고 자료로 활용되어야 하며, 실제 의료 판단은 반드시 전문 의료진의 검토와 함께 이루어져야 한다.
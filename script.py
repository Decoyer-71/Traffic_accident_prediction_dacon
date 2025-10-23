# script.py
import os, argparse, joblib
import numpy as np
import pandas as pd


# =======================
# 학습 때 사용한 전처리 유틸 (그대로)
# =======================


# 'Age'컬럼 시각화 및 데이터 활용을 위한 전처리
def convert_age_to_numeric(age_str):
    if isinstance(age_str, str) and len(age_str) > 1 :
        decade = int(age_str[:-1])
        group = age_str[-1]
        if group == 'a' : return decade + 2
        elif group == 'b' : return decade + 7
    try : return int(age_str)
    except (ValueError, TypeError) : return np.nan


# =======================
# 학습 때 사용한 A/B 검사 전처리 (그대로)
# =======================
def preprocess_A(train_A):
    df = train_A.copy()
    print("Step 1: 전처리 및 사용컬럼 선정...")
    df['Age'] = df['Age'].apply(convert_age_to_numeric)
    
    print("Step 2: feature 생성...")
    df['A1-3'] = df['A1-3'].apply(lambda x: x.split(',').count('1') if isinstance(x, str) else 0)
    df['A2-3'] = df['A2-3'].apply(lambda x: x.split(',').count('1') if isinstance(x, str) else 0)
    df['A3-6'] = df['A3-6'].apply(lambda x: x.split(',').count('1') if isinstance(x, str) else 0)
    df['A3-7'] = df['A3-7'].apply(lambda x: round(np.mean([float(num) * 0.001 for num in x.split(',')]), 2) if isinstance(x, str) else 0)
    df['A4-3'] = df['A4-3'].apply(lambda x: x.split(',').count('2') if isinstance(x, str) else 0)
    df['A4-5'] = df['A4-5'].apply(lambda x: round(np.mean([float(num) * 0.001 for num in x.split(',')]), 2) if isinstance(x, str) else 0)
    df['A5-2'] = df['A5-2'].apply(lambda x: x.split(',').count('2') if isinstance(x, str) else 0)

    # 모델 A 학습에 사용된 피처만 선택 (Test_id는 나중에 사용하기 위해 포함)
    feature_cols = ['Test_id', 'Age', 'A1-3', 'A2-3', 'A3-6', 'A3-7', 'A4-3', 'A4-5', 'A5-2', 'A6-1',
                    'A7-1', 'A8-1', 'A8-2', 'A9-1', 'A9-2', 'A9-3', 'A9-4', 'A9-5']
    # 일부 피처는 feature 생성 과정에서 사용되지 않았으므로, 원본 데이터에 해당 컬럼이 없을 수 있습니다.
    # 존재하는 컬럼만 선택하도록 수정합니다.
    return df[[col for col in feature_cols if col in df.columns]]


def preprocess_B(train_B):
    df = train_B.copy()
    print("Step 1: 전처리 및 사용컬럼 선정...")
    df['Age'] = df['Age'].apply(convert_age_to_numeric)

    print("Step 2: feature 생성...")
    df['B1-1'] = df['B1-1'].apply(lambda x: x.split(',').count('2') if isinstance(x, str) else 0)
    df['B1-3'] = df['B1-3'].apply(lambda x: len([int(i) for i in x.split(',') if int(i) in (2, 4)]) if isinstance(x, str) else 0)
    df['B2-1'] = df['B2-1'].apply(lambda x: x.split(',').count('2') if isinstance(x, str) else 0)
    df['B2-3'] = df['B2-3'].apply(lambda x: len([i for i in x.split(',') if int(i) in (2, 4)]) if isinstance(x, str) else 0)
    df['B3-1'] = df['B3-1'].apply(lambda x: x.split(',').count('2') if isinstance(x, str) else 0)
    df['B3-2'] = df['B3-2'].apply(lambda x: round(np.mean([float(num) for num in x.split(',')]), 2) if isinstance(x, str) else 0)
    df['B4-1'] = df['B4-1'].apply(lambda x: len([int(i) for i in x.split(',') if int(i) in (2, 4, 6)]) if isinstance(x, str) else 0)
    df['B4-2'] = df['B4-2'].apply(lambda x: round(np.mean([float(num) for num in x.split(',')]), 2) if isinstance(x, str) else 0)
    df['B5-1'] = df['B5-1'].apply(lambda x: x.split(',').count('2') if isinstance(x, str) else 0)
    df['B5-2'] = df['B5-2'].apply(lambda x: round(np.mean([float(num) for num in x.split(',')]), 2) if isinstance(x, str) else 0)
    df['B6'] = df['B6'].apply(lambda x: x.split(',').count('2') if isinstance(x, str) else 0)
    df['B7'] = df['B7'].apply(lambda x: x.split(',').count('2') if isinstance(x, str) else 0)
    df['B8'] = df['B8'].apply(lambda x: x.split(',').count('2') if isinstance(x, str) else 0)
    df['B10-6'] = df['B10-6'].apply(lambda x: 20 - int(x) if pd.notna(x) else 0)

    # 모델 B 학습에 사용된 피처만 선택 (Test_id는 나중에 사용하기 위해 포함)
    feature_cols = ['Test_id', 'Age', 'B1-1', 'B1-3', 'B2-1', 'B2-3', 'B3-1', 'B3-2', 'B4-1', 'B4-2', 'B5-1',
                    'B5-2', 'B6', 'B7', 'B8', 'B9-2', 'B9-3', 'B9-5', 'B10-2', 'B10-3', 'B10-5',
                    'B10-6']
    # 존재하는 컬럼만 선택하도록 수정합니다.
    return df[[col for col in feature_cols if col in df.columns]]



# =======================
# 정렬/보정 (모델이 학습 때 본 피처 순서로)
# =======================
DROP_COLS = ["Test_id","Test","PrimaryKey","Age","TestDate"]

def align_to_model(X_df, model):
    """
    모델이 학습 시 사용한 피처에 맞춰 입력 데이터프레임을 정렬하고 보정합니다.
    다양한 종류의 모델(scikit-learn, LightGBM 등)을 지원합니다.
    """
    # 1. 모델에 저장된 피처 이름 가져오기 (다양한 라이브러리 호환)
    feat_names = []
    if hasattr(model, 'feature_name_') and callable(model.feature_name_):
        feat_names = model.feature_name_()  # LightGBM
    elif hasattr(model, 'feature_names_in_'):
        feat_names = model.feature_names_in_  # Scikit-learn
    
    feat_names = list(feat_names)

    # 2. 피처 이름이 모델에 없는 경우 (e.g., NumPy 배열로 학습된 LogisticRegression)
    if not feat_names:
        print("Warning: Model does not contain feature names. Falling back to numeric columns.")
        X = X_df.drop(columns=['Test_id'], errors='ignore').select_dtypes(include=np.number).copy()
        
        expected_features = getattr(model, "n_features_in_", -1)
        if expected_features != -1 and X.shape[1] != expected_features:
            raise ValueError(f"Feature count mismatch! Model expects {expected_features} features, but got {X.shape[1]}.")
        return X.fillna(0.0)

    # 3. 피처 이름이 모델에 있는 경우
    X = X_df.copy()
    for c in feat_names:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_names]
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

# =======================
# main
# =======================
def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    TEST_DIR  = "./data"              # test.csv, A.csv, B.csv, sample_submission.csv 위치
    MODEL_DIR = "./model"             # 로지스틱 모델(base model) 저장 위치
    OUT_DIR   = "./output"
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    OUT_PATH  = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load models...")
    model_A = joblib.load(os.path.join(MODEL_DIR, "logistic_model_A.pkl"))
    model_B = joblib.load(os.path.join(MODEL_DIR, "logistic_model_B.pkl"))
    print(" OK.")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    meta = pd.read_csv(os.path.join(TEST_DIR, "test.csv"))
    Araw = pd.read_csv(os.path.join(TEST_DIR, "./test/A.csv"))
    Braw = pd.read_csv(os.path.join(TEST_DIR, "./test/B.csv"))
    Araw = pd.read_csv(os.path.join(TEST_DIR, "test/A.csv"))
    Braw = pd.read_csv(os.path.join(TEST_DIR, "test/B.csv"))
    print(f" meta={len(meta)}, Araw={len(Araw)}, Braw={len(Braw)}")
    
    # ---- 매핑 ----
    # Araw, Braw에 'Test' 컬럼이 존재하므로 ["Test_id", "Test"]를 기준으로 merge
    A_df = meta.loc[meta["Test"] == "A", ["Test_id", "Test"]].merge(Araw, on=["Test_id", "Test"], how="left")
    B_df = meta.loc[meta["Test"] == "B", ["Test_id", "Test"]].merge(Braw, on=["Test_id", "Test"], how="left")
    print(f" mapped: A={len(A_df)}, B={len(B_df)}")

    # ---- 전처리 ----
    A_df = preprocess_A(A_df)
    B_df = preprocess_B(B_df)

    # ---- 피처 정렬/보정 ----
    XA = align_to_model(A_df, model_A)
    XB = align_to_model(B_df, model_B)
    print(f" aligned: XA={XA.shape}, XB={XB.shape}")

    # ---- 예측 ----
    print("Inference Model...")
    predA = model_A.predict_proba(XA)[:,1] if len(XA) else np.array([])
    predB = model_B.predict_proba(XB)[:,1] if len(XB) else np.array([])

    # ---- Test_id와 합치기 ----
    subA = pd.DataFrame({"Test_id": A_df["Test_id"].values, "prob": predA})
    subB = pd.DataFrame({"Test_id": B_df["Test_id"].values, "prob": predB})
    probs = pd.concat([subA, subB], axis=0, ignore_index=True)

    # ---- sample_submission 기반 결과 생성 (Label 컬럼에 0~1 확률 채움) ----
    os.makedirs(OUT_DIR, exist_ok=True)
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    # sample의 Test_id 순서에 맞추어 prob 병합
    out = sample.merge(probs, on="Test_id", how="left")
    out["Label"] = out["prob"].astype(float).fillna(0.0)
    out = out.drop(columns=["prob"])

    out.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved: {OUT_PATH} (rows={len(out)})")

if __name__ == "__main__":
    main()
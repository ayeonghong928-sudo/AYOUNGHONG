import pandas as pd
import numpy as np
# 1) 데이터 로드
df = pd.read_csv("walmart.csv")
# 2) 날짜 데이터 전처리
# 문자열 형태의 날짜를 datetime으로 변환
# 시계열 분석, 월별/주별 패턴 분석을 위해 필수

df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
# 3) 비즈니스 핵심 변수 생성

# 매출 = 판매수량 × 단가
# 수요 분석 + 성과 분석에 동시에 사용

df["revenue"] = df["quantity_sold"] * df["unit_price"]
# 프로모션 / 결품 여부 컬럼을 0-1 형태로 통일

for col in ["promotion_applied", "stockout_indicator"]:
    if col in df.columns:
        df[col] = df[col].astype(int)
#4) 재고 관련 핵심 전처리

# 재고가 재주문점 이하인지 여부
# 재고 운영 리스크 분석에 사용

if "inventory_level" in df.columns and "reorder_point" in df.columns:
    df["needs_reorder"] = (df["inventory_level"] <= df["reorder_point"]).astype(int)

    # 재주문점과의 차이
    # 음수일수록 재고 부족 상태
    df["inventory_gap"] = df["inventory_level"] - df["reorder_point"]
# 5) 수요 예측 오차 변수

# 예측값과 실제값의 차이
# 예측 정확도 및 재고 비효율 분석 가능

if "forecasted_demand" in df.columns and "actual_demand" in df.columns:
    df["demand_error"] = df["actual_demand"] - df["forecasted_demand"]
    df["abs_demand_error"] = df["demand_error"].abs()
# 6) 결측치 처리

num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
# 7) 데이터 정렬
# 날짜 기준 정렬

df = df.sort_values("transaction_date")
out_path = "walmart_preprocessed_final.csv"
df.to_csv(out_path, index=False)

from google.colab import files
files.download(out_path)
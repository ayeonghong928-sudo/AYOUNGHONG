import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 0) 프로젝트 주제: 수요 예측과 프로모션 효과를 반영한 재고 운영 분석
# =========================================================
FILE_PATH = "walmart_preprocessed_final.csv"
TOP_N = 8
DPI = 120

COLORS = {
    "blue":   "#2E86AB",
    "orange": "#F18F01",
    "green":  "#06A77D",
    "red":    "#D00000",
    "purple": "#7A5195",
    "gray":   "#4B5563"
}

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": DPI,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

# =========================================================
# 1) 데이터 로드 + 컬럼 자동 인식 
# =========================================================
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"파일 없음: {FILE_PATH}")

df = pd.read_csv(FILE_PATH)

# 날짜/월 파생
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month

# 숫자 변환
for c in ["quantity_sold","inventory_level","abs_demand_error","promotion_applied","needs_reorder","month"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 제품 컬럼 자동 선택
product_candidates = ["product_name","product","Product","product_id","sku","SKU","item","Item"]
product_col = next((c for c in product_candidates if c in df.columns), None)
if product_col is None:
    raise ValueError("제품 컬럼을 찾지 못했습니다. product_name/product_id 등 컬럼 확인 필요")

# 필수 컬럼 체크
required = ["quantity_sold"]
for r in required:
    if r not in df.columns:
        raise ValueError(f"필수 컬럼 없음: {r}")

# =========================================================
# 2) 데이터 개요/품질 확인 
# =========================================================
overview = pd.DataFrame({
    "항목": ["Rows", "Cols", "Target", "Product Key", "Time Key", "Promo Key", "Inventory Key", "Error Key", "Reorder Key"],
    "내용": [
        f"{df.shape[0]:,}",
        f"{df.shape[1]}",
        "quantity_sold",
        product_col,
        "month" if "month" in df.columns else ("date" if "date" in df.columns else "N/A"),
        "promotion_applied" if "promotion_applied" in df.columns else "N/A",
        "inventory_level" if "inventory_level" in df.columns else "N/A",
        "abs_demand_error" if "abs_demand_error" in df.columns else "N/A",
        "needs_reorder" if "needs_reorder" in df.columns else "N/A"
    ]
})
display(overview)

quality = pd.DataFrame({
    "missing_rate(%)": (df.isna().mean() * 100).round(2),
    "n_unique": df.nunique(dropna=True),
    "dtype": df.dtypes.astype(str)
}).sort_values("missing_rate(%)", ascending=False)
display(quality.head(12))

summary_cols = [c for c in ["quantity_sold","inventory_level","abs_demand_error","promotion_applied","needs_reorder","month"] if c in df.columns]
desc = df[summary_cols].describe().T
desc = desc[["count","mean","std","min","25%","50%","75%","max"]].round(3)
display(desc)

# =========================================================
# EDA 0) 수요 비중 분석 
# 전체 수요가 소수의 핵심 제품에 집중되어 있는지 확인하기 위한 분석
# =========================================================
total_by_product = df.groupby(product_col)["quantity_sold"].sum().sort_values(ascending=False)
share = (total_by_product / total_by_product.sum() * 100).dropna()
top_share = share.head(TOP_N)

top_table = pd.DataFrame({
    "rank": np.arange(1, len(top_share)+1),
    "product": top_share.index.astype(str),
    "total_demand": total_by_product.loc[top_share.index].values.astype(float),
    "demand_share(%)": top_share.values.astype(float)
})
top_table["total_demand"] = top_table["total_demand"].round(0).astype(int)
top_table["demand_share(%)"] = top_table["demand_share(%)"].round(2)
display(top_table)

plt.figure(figsize=(6.4, 2.8))
vals = top_share.values[::-1]
names = top_share.index.astype(str)[::-1]
bars = plt.barh(names, vals, color=COLORS["blue"], height=0.42, edgecolor="white", linewidth=0.6)
for i, v in enumerate(vals):
    plt.text(v + 0.15, i, f"{v:.1f}%", va="center", color=COLORS["gray"])
plt.title("EDA 0) Demand Share by Top Products")
plt.xlabel("Demand Share (%)")
plt.tight_layout()
plt.show()
top_products = total_by_product.head(TOP_N)

# =========================================================
# EDA 1) 수요 분포 분석
## 수요 분포의 형태와 변동성을 파악하여 평균 기반 재고 운영의 한계를 확인하기 위한 분석
# =========================================================
plt.figure(figsize=(6.4, 2.8))
sns.histplot(df["quantity_sold"].dropna(), bins=20, kde=True, color=COLORS["blue"], edgecolor="white", alpha=0.55)
mu = df["quantity_sold"].mean()
plt.axvline(mu, color=COLORS["red"], linestyle="--", linewidth=1.6, label=f"mean={mu:.2f}")
plt.title("EDA 1) Demand Distribution (Hist + KDE)")
plt.xlabel("Quantity Sold")
plt.ylabel("Count")
plt.legend(framealpha=0.95)
plt.tight_layout()
plt.show()

# =========================================================
# EDA 2) 프로모션 효과 분석
# 프로모션 적용 여부에 따라 수요 분포가 실제로 유의미하게 변화하는지 확인하기 위한 분석

# =========================================================
if "promotion_applied" in df.columns:
    tmp = df[["quantity_sold","promotion_applied"]].dropna().copy()
    tmp["promotion_applied"] = (tmp["promotion_applied"] > 0).astype(int)

    promo_stats = tmp.groupby("promotion_applied")["quantity_sold"].agg(["count","mean","std"]).reset_index()
    promo_stats["promotion_applied"] = promo_stats["promotion_applied"].map({0:"No Promo", 1:"Promo"})
    promo_stats = promo_stats.rename(columns={"count":"n","mean":"avg_demand","std":"std_demand"})
    promo_stats[["avg_demand","std_demand"]] = promo_stats[["avg_demand","std_demand"]].round(3)
    display(promo_stats)

    plt.figure(figsize=(6.4, 2.8))
    sns.kdeplot(tmp[tmp["promotion_applied"]==0]["quantity_sold"], label="No Promo", color=COLORS["orange"], linewidth=2.2)
    sns.kdeplot(tmp[tmp["promotion_applied"]==1]["quantity_sold"], label="Promo", color=COLORS["green"], linewidth=2.2)
    plt.title("EDA 2) Promotion Effect on Demand (KDE)")
    plt.xlabel("Quantity Sold")
    plt.ylabel("Density")
    plt.legend(framealpha=0.95)
    plt.tight_layout()
    plt.show()


# =========================================================
# EDA 3. 재고 수준 vs 품절 위험 
# 재고 수준이 낮아질수록 재주문 신호 발생 확률이 증가하는지 검증하기 위한 분석
# =========================================================


df["inventory_bin"] = pd.qcut(df["inventory_level"], 5, labels=["L1","L2","L3","L4","L5"])

risk = (
    df.groupby("inventory_bin")["needs_reorder"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(6.5,3))
plt.plot(risk["inventory_bin"], risk["needs_reorder"],
         marker="o", linewidth=2,
         color=COLORS["purple"])

plt.title("EDA 3) Stock Level vs Reorder Risk")
plt.xlabel("Inventory Level (Low → High)")
plt.ylabel("Reorder Probability")
plt.tight_layout()
plt.show()

# =========================================================
# EDA 4) 예측 오차 분포 분석
# 수요 예측 오차의 분포와 극단값을 확인하여 재고 품절/과잉의 원인을 파악하기 위한 분석
# =========================================================
if "abs_demand_error" in df.columns:
    err = df["abs_demand_error"].dropna()
    p90, p95, p99 = np.percentile(err, [90,95,99])
    err_table = pd.DataFrame({
        "metric": ["mean","median","p90","p95","p99","max"],
        "value": [err.mean(), np.median(err), p90, p95, p99, err.max()]
    }).round(2)
    display(err_table)

    plt.figure(figsize=(6.4, 2.8))
    sns.histplot(err, bins=30, kde=True, color=COLORS["red"], edgecolor="white", alpha=0.30)
    plt.axvline(err.mean(), color=COLORS["red"], linestyle="--", linewidth=1.6, label=f"mean={err.mean():.2f}")
    plt.axvline(p95, color=COLORS["orange"], linestyle="--", linewidth=1.6, label=f"p95={p95:.1f}")
    plt.title("EDA 4) Absolute Forecast Error (Hist + KDE)")
    plt.xlabel("Absolute Demand Error")
    plt.ylabel("Count")
    plt.legend(framealpha=0.95)
    plt.tight_layout()
    plt.show()

# =========================================================
# EDA 5. Seasonality Heatmap 시즌성 히트맵 
# 제품별 월별 평균 수요 패턴을 통해 시즌성 및 시기별 강약 여부를 확인하기 위한 분석
# =========================================================
pivot = (
    df[df[product_col].isin(top_products.index)]
    .pivot_table(
        index=product_col,
        columns="month",
        values="quantity_sold",
        aggfunc="mean"
    )
    .reindex(top_products.index)
)

plt.figure(figsize=(6.5,3.8))
sns.heatmap(
    pivot,
    cmap="YlOrRd",
    annot=True,
    fmt=".1f",
    linewidths=0.4,
    linecolor="white",
    cbar_kws={"label":"Avg Demand"}
)

plt.title("EDA 5) Seasonality Heatmap (Avg Demand)")
plt.xlabel("Month")
plt.ylabel("Product")
plt.tight_layout()
plt.show()

# =========================================================
# EDA 6. 재주문 신호 vs 평균 수요 분석
# 재주문 신호가 실제 수요 증가 시점을 잘 포착하는지 검증하기 위한 분석
# =========================================================
avg_reorder = (
    df.groupby("needs_reorder")["quantity_sold"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(5.5,3))
plt.bar(
    ["No","Yes"],
    avg_reorder["quantity_sold"],
    color=[COLORS["orange"], COLORS["green"]],
    width=0.35
)

for i, v in enumerate(avg_reorder["quantity_sold"]):
    plt.text(i, v+0.03, f"{v:.2f}", ha="center")

plt.title("EDA 6) Avg Demand by Reorder Signal")
plt.ylabel("Avg Quantity Sold")
plt.tight_layout()
plt.show()


# =========================================================
# EDA 7) 파레토 분석 (ABC) 
# 상위 소수 제품이 전체 수요의 대부분을 차지하는지 확인하여 재고 관리 우선순위를 설정하기 위한 분석
# =========================================================
top_totals = total_by_product.head(TOP_N)
cum_share = top_totals.cumsum() / top_totals.sum() * 100

abc_idx = int(np.argmax(cum_share.values >= 80)) + 1
abc_table = pd.DataFrame({
    "rank": np.arange(1, len(top_totals)+1),
    "product": top_totals.index.astype(str),
    "total_demand": top_totals.values.astype(int),
    "cum_share(%)": cum_share.values.round(2)
})
display(abc_table)

fig, ax1 = plt.subplots(figsize=(6.4, 2.9))
x = np.arange(len(top_totals))

ax1.bar(x, top_totals.values, color=COLORS["blue"], width=0.40, edgecolor="white", linewidth=0.7)
ax1.set_ylabel("Total Demand")
ax1.set_xlabel("Rank (by Total Demand)")

ax2 = ax1.twinx()
ax2.plot(x, cum_share.values, color=COLORS["red"], marker="o", linewidth=2.2, markersize=4)
ax2.axhline(80, color=COLORS["orange"], linestyle="--", linewidth=1.8)
ax2.set_ylabel("Cumulative Share (%)")

ax1.set_title(f"EDA 7) Pareto Chart (Top {TOP_N})")
ax1.set_xticks(x)
ax1.set_xticklabels([str(i+1) for i in range(len(top_totals))])

plt.tight_layout()
plt.show()

# =========================================================
# EDA 8. Product Lifecycle Cohort 제품 생애주기 코호트 분석 
# 제품 출시 이후 시간 경과에 따른 평균 수요 변화를 통해 생애주기 패턴을 확인하기 위한 분석
# =========================================================


cohort = df[[product_col,"month","quantity_sold"]].dropna()

first_month = cohort.groupby(product_col)["month"].min()
cohort["age"] = cohort["month"] - cohort[product_col].map(first_month)

cohort = cohort[(cohort["age"] >= 0) & (cohort["age"] <= 8)]

lifecycle = (
    cohort.groupby("age")["quantity_sold"]
           .mean()
)

plt.figure(figsize=(6.5,3))
plt.plot(
    lifecycle.index,
    lifecycle.values,
    marker="o",
    linewidth=2,
    color=COLORS["purple"]
)

plt.title("EDA 8) Product Lifecycle (Average)")
plt.xlabel("Months Since First Sale")
plt.ylabel("Avg Quantity Sold")
plt.tight_layout()
plt.show()

# =========================================================
# EDA 분석 요약
# =========================================================
# 본 탐색적 데이터 분석을 통해
# 1) 수요 집중도와 분포 특성,
# 2) 프로모션 적용에 따른 수요 변화,
# 3) 재고 수준 및 재주문 신호의 관계,
# 4) 수요 예측 오차의 분포와 위험 구간,
# 5) 시즌성 및 제품 생애주기 패턴을 확인하였다.
# =========================================================

# =========================================================

eda_summary = [
    "전체 수요는 일부 상위 제품에 집중되어 있어, 모든 상품을 동일 기준으로 관리하는 방식은 비효율적이다.",
    "수요 분포는 한쪽으로 치우쳐 있고 변동성이 커, 평균 수요 기반 운영은 품절/과잉 재고 위험이 있다.",
    "프로모션 적용 시 수요 변화가 발생하며, 재주문 신호 발생 비율도 함께 증가할 수 있다.",
    "예측오차가 큰 구간에서 재주문 신호가 더 자주 나타나, 예측 정확도가 재고 안정성에 영향을 준다.",
    "시즌성과 제품 생애주기 패턴이 존재하여 시기/제품 특성을 반영한 재고 전략이 필요하다."
]
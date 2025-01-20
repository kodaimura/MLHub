# 回帰（Regression）
- 連続値の予測やモデル化

## 1. 線形回帰系
- **最小二乗法（Ordinary Least Squares, OLS）**  
  - 回帰分析の基本で、シンプルな線形モデルで広く使用される。
- **リッジ回帰（Ridge Regression）**  
  - 多重共線性が問題となる場合に利用。
- **ラッソ回帰（Lasso Regression）**  
  - 特徴量選択が必要な場合に有効。
- **Elastic Net回帰（Elastic Net Regression）**  
  - リッジ回帰とラッソ回帰の特性を組み合わせた手法。

## 2. 一般化線形モデル（GLM）に基づく手法
- **ロジスティック回帰（Logistic Regression）**  
  - 二値分類問題で頻繁に使用される。
- **ポアソン回帰（Poisson Regression）**  
  - カウントデータ（例: 発生回数など）に適用。
- **負の二項回帰（Negative Binomial Regression）**  
  - カウントデータで分散が平均より大きい場合に使用。

## 3. 非線形回帰
- **多項式回帰（Polynomial Regression）**  
  - 非線形の関係を線形モデルで近似する場合に利用。
- **スプライン回帰（Spline Regression）**  
  - 柔軟なフィットが必要な非線形データに適用。

## 4. ベイズ的手法
- **ベイズ回帰（Bayesian Regression）**  
  - 不確実性の推定が重要な場合や、先行知識をモデルに組み込みたい場合に利用。

## 5. 機械学習ベース
- **サポートベクター回帰（Support Vector Regression, SVR）**  
  - 小規模データセットで強力な性能を発揮。
- **ランダムフォレスト回帰（Random Forest Regression）**  
  - 非線形データや複雑な相互作用を扱う際に利用。
- **勾配ブースティング回帰（Gradient Boosting Regression）**  
  - 高性能な予測が求められる場合に使用（例: XGBoost、LightGBM）。
- **ニューラルネットワーク回帰（Neural Network Regression）**  
  - 複雑な非線形パターンを扱う際に利用。

## 6. 時系列分析
- **自己回帰モデル（Autoregressive Model, AR）**  
  - 短期予測の基本モデル。
- **自己回帰移動平均モデル（Autoregressive Moving Average, ARMA）**  
  - トレンドや季節性を考慮した短期予測に利用。
- **自己回帰統合移動平均モデル（Autoregressive Integrated Moving Average, ARIMA）**  
  - トレンド除去や季節性調整が必要な時系列データに適用。

## 7. 特殊用途
- **分位点回帰（Quantile Regression）**  
  - データの特定分位点（例: 中央値や上位10%）をモデル化。
- **サバイバル回帰（Survival Regression）**  
  - サバイバル分析やイベント発生時間の解析に使用。

---

# 利用が特定される状況の例
- **リッジ回帰・ラッソ回帰**：高次元データや多重共線性がある場合。
- **ロジスティック回帰**：分類タスク全般。
- **ポアソン回帰**：カウントデータ（例: 顧客到着数、事故発生数）。
- **ARIMA**：経済指標や株価予測などの時系列データ。

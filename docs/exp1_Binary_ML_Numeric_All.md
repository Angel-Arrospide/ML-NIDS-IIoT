# Experiment 1 — Binary Classification: All Numeric Features

## Objective

Establish a performance baseline using the full numeric feature set available after the cleaning phase. No feature selection is applied beyond what the cleaning step already performs (dropping raw IPs, ports, redundant/constant columns). This experiment intentionally includes features that may encode environment-specific patterns in order to measure the maximum achievable performance under the current dataset conditions.

## Feature Set

All numeric columns produced by `02_Cleaning.ipynb` after one-hot encoding of categorical fields. Feature selection: `df.select_dtypes(include=['number'])` on the processed pickle, dropping only the target column `Attack_label`.

**Total features: 50**

| Group | Features |
|---|---|
| ICMP | `icmp.seq_le` |
| HTTP | `http.content_length`, `http.response`, `http.tls_port`, `http.request.method_get`, `http.request.method_post`, `http.request.method_options`, `http.request.method_trace` |
| TCP | `tcp.ack`, `tcp.len`, `tcp.seq`, `tcp.nullchecksum`, `tcp.flag.res`, `tcp.flag.ns`, `tcp.flag.cwr`, `tcp.flag.ece`, `tcp.flag.urg`, `tcp.flag.ack`, `tcp.flag.psh`, `tcp.flag.rst`, `tcp.flag.syn`, `tcp.flag.fin` |
| UDP | `udp.stream`, `udp.time_delta` |
| DNS | `dns.retransmission`, `dns.retransmit_request`, `dns.retransmit_request_in` |
| MQTT | `mqtt.conack.flags`, `mqtt.conflag.cleansess`, `mqtt.len`, `mqtt.msgtype`, `mqtt.msgtype_connect`, `mqtt.msgtype_connack`, `mqtt.msgtype_publish`, `mqtt.msgtype_disconnect` |
| MBTCP | `mbtcp.len`, `mbtcp.trans_id`, `mbtcp.unit_id` |
| Frame | `frame.time.order`, `frame.time.delta` |
| IP (encoded) | `ip.src_category_Malformed`, `ip.src_category_Private`, `ip.src_category_Public`, `ip.src_category_Reserved`, `ip.dst_category_Malformed`, `ip.dst_category_Private`, `ip.dst_category_Public`, `ip.dst_category_Reserved` |
| ARP (encoded) | `arp.opcode_request`, `arp.opcode_reply` |

## Models

First we use LazyPredict to get a quick overview of the best models. Then we select the top 3 models and train them with hyperparameter tuning.

| Notebook | Model | Purpose |
|---|---|---|
| `exp1_Binary_1_ML_Lazy.ipynb` | LazyPredict (22 models) | Rapid benchmark to identify top-performing model families |
| `exp1_Binary_2_XGBClassifier.ipynb` | XGBoost | Gradient boosting with hyperparameter tuning |
| `exp1_Binary_3_RandomForest.ipynb` | Random Forest | Ensemble of decision trees with hyperparameter tuning |
| `exp1_Binary_4_LGBM.ipynb` | LightGBM | Fast gradient boosting with hyperparameter tuning |

## Preprocessing Pipeline

1. Load `data/processed/edge_iiot/ML-EdgeIIoT-dataset.pkl`
2. Select numeric columns: `df.select_dtypes(include=['number'])`
3. Separate target: `X = df_numeric.drop('Attack_label')`, `y = df_numeric['Attack_label']`
4. Extract attack type string for stratification: `Y_str = df['Attack_type']`
5. Encode any remaining categorical columns: `pd.get_dummies()`
6. Train/test split: 80/20, `stratify=Y_str`, `random_state=42`
7. Standardize: `StandardScaler` fit on train, applied to both sets

## Hyperparameter Tuning

- Method: `RandomizedSearchCV`, `n_iter=10`, `cv=3`
- Scoring: `f1_weighted` (appropriate for imbalanced binary classification)

## Evaluation Metrics

- Accuracy
- ROC AUC Score
- Classification report (precision, recall, F1 per class)
- Confusion matrix
- Top-20 feature importances (built-in)
- SHAP: `TreeExplainer`, stratified 5,000-sample subset, global bar + beeswarm plots

## Saved Artifacts

- `models/binary/edge_iiot/exp1/model_xgb.pkl` + `scaler_xgb.pkl`
- `models/binary/edge_iiot/exp1/model_rf.pkl` + `scaler_rf.pkl`
- `models/binary/edge_iiot/exp1/model_lgbm.pkl` + `scaler_lgbm.pkl`

## Notes

This experiment serves as the upper-bound reference for Experiment 2. Higher performance here vs. Experiment 2 indicates the model is relying on environment-specific features (`frame.time.order`, `frame.time.delta`, IP category distributions) that may not generalize to other deployments.

# Experiment 2 — Binary Classification: Environment-Agnostic Feature Selection

## Objective

Evaluate whether removing features that encode environment-specific information degrades classification performance, and if so, by how much. A model trained without these features is more likely to generalize to IIoT deployments with different network topologies, IP addressing schemes, and traffic timing patterns.

## Motivation

The Edge-IIoTset paper (Ferrag et al., 2022) explicitly discusses the challenge of generalizability in IIoT intrusion detection datasets. The dataset was captured in a specific laboratory environment with a fixed set of IoT sensors, a defined network topology, and controlled attack scenarios. Several features in the cleaned dataset reflect properties of this specific environment rather than universal attack signatures:

- **`ip.src_host` / `ip.dst_host`** — Raw IP addresses are fully environment-specific. The cleaning step in `02_Cleaning.ipynb` transforms them into categorical buckets (`Private`, `Public`, `Reserved`, `Malformed`). However, even after this transformation, the *distribution* of these categories is environment-specific: in the lab setup, normal IoT sensor traffic originates exclusively from `Private` addresses while many attacks arrive from `Public` addresses. A model trained on this pattern will learn an environmental shortcut rather than a true attack signature.

- **`frame.time.order`** — Packet processing order is a function of the capture setup, the Wireshark dissector configuration, and the capture machine's scheduling behavior. It is not reproducible across deployments.

- **`frame.time.delta`** — Inter-packet time deltas in the training data reflect the specific sensor sampling rates (e.g., a DHT11 temperature sensor publishing every 5 seconds) and the network topology of the lab. In a different IIoT environment with different sensor types or sampling intervals, these deltas will have entirely different statistical properties.

## Features Removed vs. Experiment 1

| Feature(s) | Reason |
|---|---|
| `frame.time.order` | Packet processing order — environment and capture-setup specific |
| `frame.time.delta` | Inter-packet timing — driven by local sensor sampling rates and network topology |
| `ip.src_category_Malformed` | Even after IP→category transformation, category distribution is environment-specific |
| `ip.src_category_Private` | Same reason |
| `ip.src_category_Public` | Same reason |
| `ip.src_category_Reserved` | Same reason |
| `ip.dst_category_Malformed` | Same reason |
| `ip.dst_category_Private` | Same reason |
| `ip.dst_category_Public` | Same reason |
| `ip.dst_category_Reserved` | Same reason |

**Result: 50 → 40 features** (10 features removed)

## Feature Set (40 features)

| Group | Features |
|---|---|
| ICMP | `icmp.seq_le` |
| HTTP | `http.content_length`, `http.response`, `http.tls_port`, `http.request.method_get`, `http.request.method_post`, `http.request.method_options`, `http.request.method_trace` |
| TCP | `tcp.ack`, `tcp.len`, `tcp.seq`, `tcp.nullchecksum`, `tcp.flag.res`, `tcp.flag.ns`, `tcp.flag.cwr`, `tcp.flag.ece`, `tcp.flag.urg`, `tcp.flag.ack`, `tcp.flag.psh`, `tcp.flag.rst`, `tcp.flag.syn`, `tcp.flag.fin` |
| UDP | `udp.stream`, `udp.time_delta` |
| DNS | `dns.retransmission`, `dns.retransmit_request`, `dns.retransmit_request_in` |
| MQTT | `mqtt.conack.flags`, `mqtt.conflag.cleansess`, `mqtt.len`, `mqtt.msgtype`, `mqtt.msgtype_connect`, `mqtt.msgtype_connack`, `mqtt.msgtype_publish`, `mqtt.msgtype_disconnect` |
| MBTCP | `mbtcp.len`, `mbtcp.trans_id`, `mbtcp.unit_id` |
| ARP (encoded) | `arp.opcode_request`, `arp.opcode_reply` |

## Models

| Notebook | Model | Purpose |
|---|---|---|
| `exp2_Binary_1_ML_Lazy.ipynb` | LazyPredict (22 models) | Rapid benchmark on the reduced feature set |
| `exp2_Binary_2_XGBClassifier.ipynb` | XGBoost | Gradient boosting with hyperparameter tuning |
| `exp2_Binary_3_RandomForest.ipynb` | Random Forest | Ensemble of decision trees with hyperparameter tuning |
| `exp2_Binary_4_LGBM.ipynb` | LightGBM | Fast gradient boosting with hyperparameter tuning |

## Preprocessing Pipeline

Identical to Experiment 1 up to feature selection, then adds:

1. Load `data/processed/edge_iiot/ML-EdgeIIoT-dataset.pkl`
2. Select numeric columns: `df.select_dtypes(include=['number'])`
3. Separate target and stratification label
4. Encode remaining categorical columns: `pd.get_dummies()`
5. **Drop environment-specific columns:**
   ```python
   cols_to_drop = [
       'frame.time.delta', 'frame.time.order',
       *[c for c in X.columns if c.startswith('ip.src_category') or c.startswith('ip.dst_category')]
   ]
   X = X.drop(columns=cols_to_drop, errors='ignore')
   ```
6. Train/test split: 80/20, `stratify=Y_str`, `random_state=42`
7. Standardize: `StandardScaler` fit on train, applied to both sets

## Hyperparameter Tuning

- Method: `RandomizedSearchCV`, `n_iter=10`, `cv=3`
- Scoring: `f1_weighted` (appropriate for imbalanced binary classification)

## Evaluation Metrics

Same as Experiment 1:
- Accuracy, ROC AUC Score, Classification report, Confusion matrix
- Top-20 feature importances
- SHAP: `TreeExplainer`, stratified 5,000-sample subset, global bar + beeswarm plots

## Saved Artifacts

- `models/binary/edge_iiot/exp2/model_xgb.pkl` + `scaler_xgb.pkl`
- `models/binary/edge_iiot/exp2/model_rf.pkl` + `scaler_rf.pkl`
- `models/binary/edge_iiot/exp2/model_lgbm.pkl` + `scaler_lgbm.pkl`

## Interpretation

A drop in performance from Experiment 1 to Experiment 2 quantifies how much the models in Experiment 1 were relying on environment-specific shortcuts. A small or negligible drop suggests the protocol-level features (TCP flags, MQTT message types, DNS retransmissions, etc.) are sufficient for robust detection and the models generalize well. A large drop indicates that environment-specific features were providing significant discriminative power and further feature engineering would be needed for real-world deployment.

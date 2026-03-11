"""
Auto Dataset Analyzer — istalgan CSV/Excel faylni tahlil qiladi
Risk dateset ekanligini topib, ML model ishlatadi
"""
import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, f1_score, accuracy_score)
from sklearn.impute import SimpleImputer
import base64
import warnings
warnings.filterwarnings('ignore')

# ── Risk bilan bog'liq kalit so'zlar ────────────────────────
RISK_KEYWORDS = {
    'target': [
        'default','risk','status','failed','bad','churn','fraud',
        'default_holati','target','label','y','outcome','is_default',
        'loan_status','credit_status','result','class','approved','rejected'
    ],
    'credit_score': ['score','kredit','credit','cscore','fico','rating','ball'],
    'amount': ['amount','sum','miqdor','loan','credit_amount','principal','balance','debt'],
    'income': ['income','daromad','salary','revenue','earnings','maosh'],
    'age': ['age','yosh','years'],
    'rate': ['rate','stavka','interest','apr','yield'],
    'ltv': ['ltv','loan_to_value','collateral'],
    'tenure': ['tenure','duration','term','muddat','months','days'],
}

def smart_detect_columns(df: pd.DataFrame) -> dict:
    """Datasetdagi ustunlarni aqlli aniqlash"""
    detected = {}
    cols_lower = {c.lower().replace(' ','_').replace('-','_'): c for c in df.columns}

    for role, keywords in RISK_KEYWORDS.items():
        for kw in keywords:
            for col_lower, col_orig in cols_lower.items():
                if kw in col_lower:
                    # Target ustun uchun binary tekshirish
                    if role == 'target':
                        uvals = df[col_orig].dropna().unique()
                        if len(uvals) == 2:
                            detected[role] = col_orig
                            break
                    else:
                        if col_orig not in detected.values():
                            detected[role] = col_orig
                            break
            if role in detected:
                break

    # Agar target topilmasa — eng kam unique qiymatli binary ustun
    if 'target' not in detected:
        for col in df.columns:
            uv = df[col].dropna().unique()
            if len(uv) == 2 and col not in detected.values():
                # Numeric binary
                try:
                    vals = sorted([float(x) for x in uv])
                    if vals == [0.0, 1.0] or vals == [0, 1]:
                        detected['target'] = col
                        break
                except: pass

    return detected


def preprocess_auto(df: pd.DataFrame, target_col: str):
    """Avtomatik preprocessing: encoding, imputing, scaling"""
    df2 = df.copy()

    # Target
    y_raw = df2[target_col].dropna()
    uvals = y_raw.unique()

    # Binary encoding
    if len(uvals) == 2:
        if set(uvals) != {0, 1}:
            le_t = LabelEncoder()
            df2[target_col] = le_t.fit_transform(df2[target_col].astype(str))
        else:
            df2[target_col] = df2[target_col].astype(int)

    # Faqat target bor qatorlarni ol
    df2 = df2.dropna(subset=[target_col])
    y = df2[target_col].values

    # Feature ustunlar
    feature_cols = [c for c in df2.columns if c != target_col]

    # Kategorik → label encoding
    cat_cols = df2[feature_cols].select_dtypes(include=['object','category']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col].astype(str))

    # Numeric feature'lar
    num_cols = df2[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    X = df2[num_cols].values

    # Imputing (NaN → median)
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Scaling
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    return X_sc, y, num_cols

def generate_prepared_data(df: pd.DataFrame, detected: dict) -> str:
    """Datasetni tahlilga tayyorlab, CSV string ko'rinishida qaytaradi"""
    df_prep = df.copy()
    
    # 1. Keraksiz (numeric bo'lmagan va unique qiymati juda ko'p) ustunlarni o'chirish
    # Masalan: ID, Ism va h.k.
    for col in df_prep.columns:
        if df_prep[col].dtype == 'object' and df_prep[col].nunique() > 100:
            df_prep.drop(columns=[col], inplace=True)

    # 2. Null qiymatlarni to'ldirish
    for col in df_prep.select_dtypes(include=[np.number]).columns:
        if df_prep[col].isnull().all():
            df_prep[col] = 0.0
        else:
            df_prep[col] = df_prep[col].fillna(df_prep[col].median())
    
    for col in df_prep.select_dtypes(exclude=[np.number]).columns:
        if df_prep[col].isnull().all():
            df_prep[col] = "Unknown"
        else:
            df_prep[col] = df_prep[col].fillna(df_prep[col].mode()[0] if not df_prep[col].mode().empty else "Unknown")

    # 3. Label Encoding for categories
    cat_cols = df_prep.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df_prep[col] = le.fit_transform(df_prep[col].astype(str))

    # 4. Standartizatsiya (Scaling) — optional, lekin ko'pincha kerak
    # Biz shunchaki cleaned CSV ni qaytaramiz
    
    return df_prep.to_csv(index=False)


def run_auto_analysis(file_bytes: bytes, filename: str) -> dict:
    """
    Asosiy funksiya: faylni o'qib, tahlil qiladi va natija qaytaradi
    """
    result = {}

    # ── 1. Faylni o'qish ────────────────────────────────────
    try:
        if filename.endswith('.csv'):
            # Turli delimiter'larni sinab ko'rish
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
                    if df.shape[1] > 1:
                        break
                except: pass
        elif filename.endswith(('.xlsx','.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return {'error': 'Faqat CSV yoki Excel fayllar qabul qilinади'}
    except Exception as e:
        return {'error': f'Fayl o\'qishda xato: {str(e)}'}

    if df.empty or df.shape[1] < 2:
        return {'error': 'Dataset bo\'sh yoki juda kam ustun'}

    result['basic'] = {
        'rows': len(df),
        'cols': df.shape[1],
        'columns': list(df.columns),
        'dtypes': {c: str(df[c].dtype) for c in df.columns},
        'missing_pct': {c: round(df[c].isna().mean()*100, 1) for c in df.columns},
    }

    # ── 2. Ustunlarni aniqlash ───────────────────────────────
    detected = smart_detect_columns(df)
    result['detected_columns'] = detected
    result['has_target'] = 'target' in detected

    # ── 3. Statistik tahlil ─────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe()
        result['statistics'] = {
            col: {
                'mean':  round(float(desc[col]['mean']), 4),
                'std':   round(float(desc[col]['std']), 4),
                'min':   round(float(desc[col]['min']), 4),
                'max':   round(float(desc[col]['max']), 4),
                'median':round(float(df[col].median()), 4),
            } for col in num_cols[:20]
        }

    # ── 4. Target aniqlanganda risk tahlil ──────────────────
    if 'target' in detected:
        target_col = detected['target']
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=[target_col])

        # Target encoding
        uv = df_clean[target_col].unique()
        if set(uv) != {0, 1}:
            le_t = LabelEncoder()
            df_clean[target_col] = le_t.fit_transform(df_clean[target_col].astype(str))

        default_rate = float(df_clean[target_col].mean())
        result['risk'] = {
            'default_rate': round(default_rate*100, 2),
            'default_count': int(df_clean[target_col].sum()),
            'normal_count':  int((df_clean[target_col]==0).sum()),
        }

        # Kategorik ustunlar bo'yicha default
        cat_cols = df_clean.select_dtypes(include=['object','category']).columns.tolist()
        cat_analysis = {}
        for col in cat_cols[:5]:
            try:
                gr = df_clean.groupby(col)[target_col].mean()
                cat_analysis[col] = {
                    str(k): round(float(v)*100, 2) for k,v in gr.items()
                }
            except: pass
        if cat_analysis:
            result['risk']['by_category'] = cat_analysis

        # ── 5. ML Model ─────────────────────────────────────
        try:
            X_sc, y, feat_names = preprocess_auto(df_clean, target_col)

            if len(X_sc) >= 50 and len(feat_names) >= 1:
                # Class imbalance tekshirish
                pos_share = y.mean()
                class_w = 'balanced' if pos_share < 0.3 or pos_share > 0.7 else None

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_sc, y, test_size=0.25, random_state=42,
                    stratify=y if len(np.unique(y))==2 else None
                )

                models_res = {}
                cv5 = StratifiedKFold(3, shuffle=True, random_state=42)

                # Logistic Regression
                lr = LogisticRegression(
                    max_iter=1000, class_weight=class_w, solver='saga', C=0.5
                )
                lr.fit(X_tr, y_tr)
                lr_prob = lr.predict_proba(X_te)[:,1]
                lr_auc = float(roc_auc_score(y_te, lr_prob))
                models_res['Logistic Regression'] = {'auc': round(lr_auc, 4)}

                # Random Forest
                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=6, class_weight=class_w,
                    random_state=42, n_jobs=-1
                )
                rf.fit(X_tr, y_tr)
                rf_prob = rf.predict_proba(X_te)[:,1]
                rf_auc = float(roc_auc_score(y_te, rf_prob))
                models_res['Random Forest'] = {'auc': round(rf_auc, 4)}

                # GBM
                gbm = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
                )
                gbm.fit(X_tr, y_tr)
                gbm_prob = gbm.predict_proba(X_te)[:,1]
                gbm_auc = float(roc_auc_score(y_te, gbm_prob))
                models_res['GBM'] = {'auc': round(gbm_auc, 4)}

                # Ensemble
                ens_prob = np.mean([lr_prob, rf_prob, gbm_prob], axis=0)
                ens_auc = float(roc_auc_score(y_te, ens_prob))
                models_res['Ensemble'] = {'auc': round(ens_auc, 4)}

                # Optimal threshold
                best_f1, best_thr = 0, 0.5
                for t in np.arange(0.1, 0.9, 0.05):
                    yp = (ens_prob >= t).astype(int)
                    f1 = f1_score(y_te, yp, zero_division=0)
                    if f1 > best_f1:
                        best_f1, best_thr = f1, t

                final_pred = (ens_prob >= best_thr).astype(int)
                cm = confusion_matrix(y_te, final_pred).tolist()

                # Feature importance (RF)
                importances = rf.feature_importances_
                fi_sorted = sorted(
                    zip(feat_names, importances.tolist()),
                    key=lambda x: x[1], reverse=True
                )[:15]

                # Risk segmentatsiya
                high_risk = (ens_prob >= 0.5).mean()
                medium_risk = ((ens_prob >= 0.3) & (ens_prob < 0.5)).mean()
                low_risk = (ens_prob < 0.3).mean()

                result['ml'] = {
                    'models': models_res,
                    'best_model': max(models_res, key=lambda k: models_res[k]['auc']),
                    'optimal_threshold': round(float(best_thr), 2),
                    'best_f1': round(float(best_f1), 4),
                    'confusion_matrix': cm,
                    'feature_importance': fi_sorted,
                    'train_size': len(X_tr),
                    'test_size': len(X_te),
                    'features_used': feat_names,
                    'risk_segments': {
                        'yuqori_risk': round(float(high_risk)*100, 1),
                        'orta_risk':   round(float(medium_risk)*100, 1),
                        'past_risk':   round(float(low_risk)*100, 1),
                    }
                }

                # ROC uchun ma'lumotlar
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_te, ens_prob)
                result['ml']['roc'] = {
                    'fpr': [round(x,4) for x in fpr[::max(1,len(fpr)//50)].tolist()],
                    'tpr': [round(x,4) for x in tpr[::max(1,len(tpr)//50)].tolist()],
                    'auc': round(ens_auc, 4),
                }

        except Exception as e:
            result['ml_error'] = str(e)

    # ── 6. Korrelyatsiya ────────────────────────────────────
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] >= 2:
        corr = num_df.corr()
        # Target bilan eng kuchli korrelyatsiyalar
        target_col = detected.get('target')
        if target_col and target_col in corr.columns:
            tc = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
            result['top_correlations'] = [
                {'feature': k, 'correlation': round(float(v), 4)}
                for k, v in tc.head(10).items()
            ]

    # ── 7. Umumiy risk xulosasi ──────────────────────────────
    if 'ml' in result:
        ens_auc_val = result['ml']['models']['Ensemble']['auc']
        def_rate = result['risk']['default_rate']
        if ens_auc_val >= 0.80:
            model_quality = "Ajoyib (AUC ≥ 0.80)"
        elif ens_auc_val >= 0.70:
            model_quality = "Yaxshi (AUC ≥ 0.70)"
        elif ens_auc_val >= 0.60:
            model_quality = "Qoniqarli (AUC ≥ 0.60)"
        else:
            model_quality = "Yaxshilanishi kerak (AUC < 0.60)"

        result['summary'] = {
            'model_quality': model_quality,
            'risk_level': 'Yuqori' if def_rate > 20 else ('O\'rta' if def_rate > 10 else 'Past'),
            'recommendation': (
                f"Dataset {result['basic']['rows']} qatordan iborat. "
                f"Default nisbati {def_rate}%. "
                f"Ensemble modeli AUC={ens_auc_val:.4f} — {model_quality}."
            )
        }

    # ── 8. Prepared Data ────────────────────────────────────
    try:
        prepared_csv = generate_prepared_data(df, detected)
        result['prepared_data_ready'] = True
        result['prepared_csv_sample'] = prepared_csv[:1000] # Faqat preview
        # To'liq datani saqlash uchun alohida endpoint kerak bo'ladi yoki base64
        result['prepared_data_b64'] = base64.b64encode(prepared_csv.encode()).decode()
    except Exception:
        result['prepared_data_ready'] = False

    return result

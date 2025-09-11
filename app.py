# Robust prediction cell - handles many label-encoder mismatch cases
import pickle
import pandas as pd
import numpy as np

# ---- load saved artifacts ----
with open("student_dropout_xgb.pkl", "rb") as f:
    artifacts = pickle.load(f)

xgb_model = artifacts["xgboost"]
label_encoders = artifacts["label_encoders"]   # can be sklearn LabelEncoder objects or dicts
feature_order = artifacts["features"]

# ---- helper: safe encoding function ----
def safe_label_encode(series, le):
    """
    Returns a numpy array of encoded values for `series` using `le`.
    Handles:
      - le as sklearn.preprocessing.LabelEncoder
      - le as a dict mapping original->encoded
      - unseen categories (mapped to a default)
    """
    s = series.copy()
    # case: stored mapping dict
    if isinstance(le, dict):
        mapping = le
        default = list(mapping.values())[0] if len(mapping)>0 else 0
        return s.map(mapping).fillna(default).astype(int).values

    # require LabelEncoder-like object
    if not hasattr(le, "classes_"):
        raise ValueError("Encoder is not a dict or sklearn LabelEncoder-like object.")

    classes = le.classes_
    # string-like classes (most common case)
    if classes.dtype.kind in ("U", "S", "O"):   # unicode, bytes, object
        s = s.astype(str)
        seen = set(map(str, classes))
        unseen = set(s.unique()) - seen
        if unseen:
            # fallback policy: map unseen categories to the encoder's first seen class
            default = classes[0]
            print(f"Warning: unseen categories {unseen} → mapping to default '{default}'")
            s = s.apply(lambda x: x if x in seen else default)
        return le.transform(s)

    # numeric classes (encoder was fit on numeric dtype)
    else:
        # try to coerce inputs to numeric
        try:
            numeric = pd.to_numeric(s)
            unseen = set(numeric.unique()) - set(classes.astype(numeric.dtype))
            if unseen:
                default = classes[0]
                print(f"Warning: unseen numeric categories {unseen} → mapping to default {default}")
                numeric = numeric.apply(lambda x: x if x in classes else default)
            return le.transform(numeric)
        except Exception:
            # cannot convert strings to numeric: map all to default numeric class and transform that
            default = classes[0]
            print(f"Warning: encoder expects numeric classes {list(classes)}, received strings {list(s.unique())}. Mapping all to default {default}")
            numeric = np.array([default] * len(s))
            return le.transform(numeric)

# ---- example new student (replace with your input) ----
new_student = {
    "School": "GP",
    "Gender": "M",
    "Age": 20,
    "Number_of_Failures": 3,
    "Travel_Time": 2,
    "Study_Time": 1,
    "Grade_1": 0,
    "Grade_2": 0,
    "Final_Grade": 10
}

# ---- prepare dataframe ----
new_df = pd.DataFrame([new_student])

# ---- encode categorical columns robustly ----
for col, le in label_encoders.items():
    if col in new_df.columns:
        try:
            new_df[col] = safe_label_encode(new_df[col], le)
        except Exception as exc:
            # last-resort fallback: map everything to the encoder's default class index
            print(f"Error encoding column '{col}': {exc}\nApplying fallback default encoding for '{col}'.")
            try:
                default_class = le.classes_[0]
                encoded_default = le.transform([default_class])[0]
            except Exception:
                encoded_default = 0
            new_df[col] = encoded_default

# ---- sanity checks ----
missing = [c for c in feature_order if c not in new_df.columns]
if missing:
    raise ValueError(f"Input is missing required features: {missing}")

# reorder and ensure numeric dtype
new_df = new_df[feature_order]
new_df = new_df.astype(float)

# ---- predict ----
pred = xgb_model.predict(new_df)[0]
prob = None
if hasattr(xgb_model, "predict_proba"):
    prob = xgb_model.predict_proba(new_df)[0][1]

print("Input row:")
print(new_df)
print("\nPrediction:", "Dropout" if prob*100 >= 65 else "Not Dropout")
if prob is not None:
    print(f"Dropout probability: {prob*100:.2f}%")

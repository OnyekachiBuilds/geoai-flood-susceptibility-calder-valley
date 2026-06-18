import geopandas as gpd
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("Loading training data...")

training_path = r"C:\Users\user\Documents\GeoAI-Based Flood Susceptibility Mapping in Calder Valley Using GIS, Remote Sensing, and Random Forest Machine Learning\FloodTrainingData.gpkg"

gdf = gpd.read_file(training_path)

print("Original rows:", len(gdf))

df = gdf[
    ["Flood_Class", "SAMPLE_1", "SAMPLE_2", "SAMPLE_3", "SAMPLE_4", "SAMPLE_5", "SAMPLE_6", "SAMPLE_7"]
].copy()

df = df.rename(columns={
    "SAMPLE_1": "DEM",
    "SAMPLE_2": "Slope",
    "SAMPLE_3": "NDVI",
    "SAMPLE_4": "LandCover",
    "SAMPLE_5": "Rainfall",
    "SAMPLE_6": "DistanceRiver",
    "SAMPLE_7": "FlowAccum"
})

df = df.replace([-9999, -3.40282e38], np.nan)
df = df.dropna()

df["Flood_Class"] = df["Flood_Class"].astype(int)

print("Rows after removing NoData:", len(df))
print(df["Flood_Class"].value_counts())

flooded = df[df["Flood_Class"] == 1]
non_flooded = df[df["Flood_Class"] == 0].sample(n=len(flooded), random_state=42)

balanced_df = pd.concat([flooded, non_flooded]).sample(frac=1, random_state=42)

X = balanced_df.drop(columns=["Flood_Class"])
y = balanced_df["Flood_Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

importance = pd.DataFrame({
    "Variable": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print(importance)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# 1. Confusion Matrix Figure
ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    display_labels=["Non-Flood", "Flood"],
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("Confusion_Matrix_RF.png", dpi=300)
plt.close()

print("Confusion matrix figure saved.")


# 2. ROC Curve Figure
RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.title("Random Forest ROC Curve")
plt.tight_layout()
plt.savefig("ROC_Curve_RF.png", dpi=300)
plt.close()

print("ROC curve figure saved.")

# 3. Feature Importance Chart
importance_sorted = importance.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(8, 5))
plt.barh(
    importance_sorted["Variable"],
    importance_sorted["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Predictor Variable")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("Feature_Importance_RF.png", dpi=300)
plt.close()

print("Feature importance figure saved.")
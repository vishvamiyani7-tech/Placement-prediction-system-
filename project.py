# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dtale
import sweetviz as sv
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from autoviz import AutoViz_Class
from ydata_profiling import ProfileReport












#information 
df = pd.read_csv("placementdata.csv")
print(df.head())
print(df.shape)
print(df.columns.tolist())
df.info()
print(df.describe())

#univariate variables
#numeric 
numeric_cols = df.select_dtypes(include=np.number).columns
print(df[numeric_cols].describe())
#histogram
df[numeric_cols].hist(figsize=(12, 8))
plt.tight_layout()
plt.show()
#catagorical
categorical_cols = df.select_dtypes(
    include=["object", "category", "bool"]).columns
#plot
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#bivariate variables
print(df[numeric_cols].corr())
if len(numeric_cols) >= 2:
    plt.figure(figsize=(6, 4))
#scatter plot
    sns.scatterplot(
        data=df,
        x=numeric_cols[0],
        y=numeric_cols[1])
    plt.title("Bivariate Analysis")
    plt.tight_layout()
    plt.show()
#comparison
if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=df,
        x=categorical_cols[0],
        y=numeric_cols[0])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#multivariate variables
# Correlation heatmap
if len(numeric_cols) >= 2:
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df[numeric_cols].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f")
    plt.title("Multivariate Analysis - Correlation Heatmap")
    plt.tight_layout()
    plt.show()
# Pair plot
if 2 <= len(numeric_cols) <= 5:
    sns.pairplot(df[numeric_cols].dropna())
    plt.show()

#EDA
print(df.isnull().sum())
missing_percentage = df.isnull().mean() * 100
print(missing_percentage)
df_clean = df.copy()
numeric_cols = df_clean.select_dtypes(
    include=np.number
).columns
categorical_cols = df_clean.select_dtypes(
    include=["object","string", "category", "bool"]
).columns

# Select numerical columns
numeric_cols = df_clean.select_dtypes(
    include=np.number
).columns
# Draw box plots
for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df_clean[col])
    plt.title(f"Outlier Detection: {col}")
    plt.show()

#spread of data
# Select quantitative columns
quantitative_cols = df_clean.select_dtypes(
    include=np.number
).columns
print("Quantitative Columns:")
print(quantitative_cols.tolist())
#count of certifications
if "Certifications" in df_clean.columns:
    print(df_clean["Certifications"].value_counts().sort_index())
# Example: distribution of CGPA
if "CGPA" in df_clean.columns:
    print(df_clean["CGPA"].describe())
    sns.histplot(data=df_clean, x="CGPA", kde=True)
    plt.title("CGPA Distribution")
    plt.show()

# Select categorical columns
qualitative_cols = df_clean.select_dtypes(
    include=["object", "string","category","bool"]
).columns
print("Qualitative Columns:")
print(qualitative_cols.tolist())
# Example: department categories
if "Department" in df_clean.columns:
    print(df_clean["Department"].value_counts())

    sns.countplot(data=df_clean, x="Department")
    plt.xticks(rotation=45)
    plt.show()
# Example: ordered skill levels
if "SkillLevel" in df_clean.columns:
    skill_order = ["Low", "Medium", "High"]
    print(
        df_clean["SkillLevel"]
        .value_counts()
        .reindex(skill_order, fill_value=0)
    )
# Example: check distribution of CGPA
if "CGPA" in df_clean.columns:
    sns.histplot(
        data=df_clean,
        x="CGPA",
        kde=True
    )
    plt.title("Distribution of CGPA")
    plt.show()

# Plot distributions of numerical columns
for col in quantitative_cols:
    sns.histplot(data=df_clean, x=col, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()
# Calculate skewness
print(df_clean[quantitative_cols].skew())
# Calculate kurtosis
print(df_clean[quantitative_cols].kurtosis())
# Numerical correlation
print("\nCorrelation:")
print(df.select_dtypes(include="number").corr())

#D TALE
import dtale
d = dtale.show(
    df,
    host="127.0.0.1",
    port=40000,
    reaper_on=False,
    open_browser=False
)
print("D-Tale URL:", d._main_url)
input("Press Enter to stop the server...")

#data profling
df = pd.read_csv("placementdata.csv")
profile = ProfileReport(
    df,
    title="Student Placement Dataset Report")
profile.to_file("placement_eda_report.html")
print("EDA report created successfully!")

#sweetviz
import sweetviz as sv
# Load dataset
df = pd.read_csv("placementdata.csv")
# Generate EDA report
report = sv.analyze(df)
# Save report
report.show_html("sweetviz_report.html")

#autoviz
AV = AutoViz_Class()
# Generate visualizations
AV.AutoViz(
    filename="",
    dfte=df,
    depVar="",
    verbose=1
)

#unit 3
#feature
feature_cols = [
    "CGPA",
    "Internships",
    "Projects",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "PlacementTraining"
]
#target
target_col = "PlacementStatus"

# Keep required columns and remove rows with missing target
data = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
X = data[feature_cols]
y = data[target_col].astype(str).str.strip()
# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

# Numeric data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
# Categorical data
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
# Process each type separately
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])
# Build classification model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model.fit(X_train, y_train)
# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
# Evaluation
print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Testing Accuracy:", accuracy_score(y_test, test_pred))
print("\nClassification Report:")
print(classification_report(y_test, test_pred))

#OVERFITTING AND UNDERFITTING
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
#check 
if train_accuracy - test_accuracy > 0.10:
    print("Possible Overfitting")
elif train_accuracy < 0.65 and test_accuracy < 0.65:
    print("Possible Underfitting")
else:
    print("No strong warning from accuracy alone")


# Unit 4: Cross-Validation and Model Evaluation
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
# Cross-Validation
scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)
print("Cross-Validation Scores:", scores)
print("Mean CV Accuracy:", scores.mean())
# Train the final model
model.fit(X_train, y_train)
# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
# Evaluation
print("Training Accuracy:",
      accuracy_score(y_train, train_pred))

print("Testing Accuracy:",
      accuracy_score(y_test, test_pred))

print("\nClassification Report:")
print(classification_report(y_test, test_pred))


# New Student Placement Prediction

print("\n--- New Student Placement Prediction ---")

feature_cols = [
    "CGPA",
    "Internships",
    "Projects",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "PlacementTraining"
]

numeric_cols = [
    "CGPA",
    "Internships",
    "Projects",
    "AptitudeTestScore",
    "SoftSkillsRating"
]

new_student = {}

print("\nEnter New Student Details:")

for col in feature_cols:
    value = input(f"Enter {col}: ")

    if col in numeric_cols:
        new_student[col] = float(value)
    else:
        new_student[col] = value.strip().title()

# Convert input into DataFrame
new_student_df = pd.DataFrame(
    [new_student],
    columns=feature_cols
)

# Predict placement
prediction = model.predict(new_student_df)[0]

print("\n--- Prediction Result ---")
print("Predicted Placement Status:", prediction)

# Show prediction probability
probability = model.predict_proba(new_student_df)[0]

print("\nPrediction Probability:")

for status, prob in zip(model.classes_, probability):
    print(f"{status}: {prob * 100:.2f}%")

print("\nStudent Details:")
print(new_student_df)




















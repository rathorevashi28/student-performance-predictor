# ============================================================
#  Student Performance Analysis & Score Predictor
#  Author  : Vashistha Rathore
#  GitHub  : github.com/rathorevashi28
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

print("=" * 55)
print("  Student Performance Analysis & Score Predictor")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# STEP 1 — Load Data
# ─────────────────────────────────────────────────────────
print("\n[1] Loading dataset...")
df = pd.read_csv('data/student_data.csv')
print(f"    Dataset loaded: {df.shape[0]} students, {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────
# STEP 2 — Data Cleaning
# ─────────────────────────────────────────────────────────
print("\n[2] Cleaning data...")
print(f"    Missing values:\n{df.isnull().sum()}")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"    Rows after cleaning: {len(df)}")

# ─────────────────────────────────────────────────────────
# STEP 3 — Basic Statistics using NumPy
# ─────────────────────────────────────────────────────────
print("\n[3] Basic Statistics (NumPy)")
numeric_cols = ['study_time', 'attendance', 'assignment_score',
                'midterm_score', 'final_score']

for col in numeric_cols:
    arr = df[col].values
    print(f"\n    {col}:")
    print(f"      Mean    : {np.mean(arr):.2f}")
    print(f"      Median  : {np.median(arr):.2f}")
    print(f"      Std Dev : {np.std(arr):.2f}")
    print(f"      Min     : {np.min(arr):.2f}  |  Max: {np.max(arr):.2f}")

pass_count  = (df['result'] == 'Pass').sum()
fail_count  = (df['result'] == 'Fail').sum()
print(f"\n    Pass: {pass_count} students  |  Fail: {fail_count} students")
print(f"    Pass Rate: {pass_count / len(df) * 100:.1f}%")

# ─────────────────────────────────────────────────────────
# STEP 4 — EDA Charts
# ─────────────────────────────────────────────────────────
print("\n[4] Generating EDA charts...")

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10})

# Chart 1 — Pass vs Fail bar chart
fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#2ecc71', '#e74c3c']
ax.bar(['Pass', 'Fail'], [pass_count, fail_count], color=colors, width=0.4)
ax.set_title('Pass vs Fail Count', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Students')
for i, v in enumerate([pass_count, fail_count]):
    ax.text(i, v + 0.3, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/01_pass_fail_count.png', dpi=150)
plt.close()
print("    Saved: outputs/01_pass_fail_count.png")

# Chart 2 — Study Time vs Final Score scatter
fig, ax = plt.subplots(figsize=(7, 5))
colors_map = df['result'].map({'Pass': '#2ecc71', 'Fail': '#e74c3c'})
ax.scatter(df['study_time'], df['final_score'], c=colors_map, alpha=0.7, s=60)
ax.set_xlabel('Study Time (hours/day)')
ax.set_ylabel('Final Score')
ax.set_title('Study Time vs Final Score', fontsize=14, fontweight='bold')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='Pass'),
                   Patch(facecolor='#e74c3c', label='Fail')]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig('outputs/02_studytime_vs_score.png', dpi=150)
plt.close()
print("    Saved: outputs/02_studytime_vs_score.png")

# Chart 3 — Attendance vs Final Score scatter
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df['attendance'], df['final_score'], c=colors_map, alpha=0.7, s=60)
ax.set_xlabel('Attendance (%)')
ax.set_ylabel('Final Score')
ax.set_title('Attendance vs Final Score', fontsize=14, fontweight='bold')
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig('outputs/03_attendance_vs_score.png', dpi=150)
plt.close()
print("    Saved: outputs/03_attendance_vs_score.png")

# Chart 4 — Correlation Heatmap
fig, ax = plt.subplots(figsize=(7, 5))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            linewidths=0.5, square=True)
ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/04_correlation_heatmap.png', dpi=150)
plt.close()
print("    Saved: outputs/04_correlation_heatmap.png")

# Chart 5 — Distribution of Final Scores by result
fig, ax = plt.subplots(figsize=(7, 4))
df[df['result'] == 'Pass']['final_score'].plot(kind='hist', bins=10,
    alpha=0.6, color='#2ecc71', label='Pass', ax=ax)
df[df['result'] == 'Fail']['final_score'].plot(kind='hist', bins=10,
    alpha=0.6, color='#e74c3c', label='Fail', ax=ax)
ax.set_xlabel('Final Score')
ax.set_ylabel('Number of Students')
ax.set_title('Final Score Distribution — Pass vs Fail',
             fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/05_score_distribution.png', dpi=150)
plt.close()
print("    Saved: outputs/05_score_distribution.png")

# Chart 6 — Average scores by result (bar chart)
fig, ax = plt.subplots(figsize=(8, 5))
avg_by_result = df.groupby('result')[numeric_cols].mean()
avg_by_result.T.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'],
                     alpha=0.8, width=0.6)
ax.set_title('Average Scores: Pass vs Fail', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Score / Value')
ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
ax.legend(title='Result')
plt.tight_layout()
plt.savefig('outputs/06_avg_scores_by_result.png', dpi=150)
plt.close()
print("    Saved: outputs/06_avg_scores_by_result.png")

# ─────────────────────────────────────────────────────────
# STEP 5 — Machine Learning: Logistic Regression
# ─────────────────────────────────────────────────────────
print("\n[5] Training Logistic Regression Model...")

# Encode target: Pass=1, Fail=0
le = LabelEncoder()
df['result_encoded'] = le.fit_transform(df['result'])   # Fail=0, Pass=1

# Features and target
X = df[['study_time', 'attendance', 'assignment_score',
         'midterm_score', 'final_score']]
y = df['result_encoded']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"    Training samples : {len(X_train)}")
print(f"    Testing  samples : {len(X_test)}")

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)

print(f"\n    Model Results:")
print(f"      Accuracy  : {accuracy * 100:.1f}%")
print(f"      Precision : {precision * 100:.1f}%")
print(f"      Recall    : {recall * 100:.1f}%")
print(f"\n    Full Report:\n")
print(classification_report(y_test, y_pred,
      target_names=['Fail', 'Pass']))

# Chart 7 — Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Fail', 'Pass'],
            yticklabels=['Fail', 'Pass'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/07_confusion_matrix.png', dpi=150)
plt.close()
print("    Saved: outputs/07_confusion_matrix.png")

# ─────────────────────────────────────────────────────────
# STEP 6 — Predict for a New Student
# ─────────────────────────────────────────────────────────
print("\n[6] Predicting for a New Student...")
print("    Input: study_time=4, attendance=70, assignment=60,"
      " midterm=55, final=58")

new_student = pd.DataFrame({
    'study_time':       [4],
    'attendance':       [70],
    'assignment_score': [60],
    'midterm_score':    [55],
    'final_score':      [58]
})

prediction = model.predict(new_student)[0]
probability = model.predict_proba(new_student)[0]
result_label = le.inverse_transform([prediction])[0]

print(f"\n    Predicted Result : {result_label}")
print(f"    Confidence       : {max(probability) * 100:.1f}%")
print(f"    Pass Probability : {probability[1] * 100:.1f}%")
print(f"    Fail Probability : {probability[0] * 100:.1f}%")

# ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  All done! Check the outputs/ folder for all charts.")
print("=" * 55)

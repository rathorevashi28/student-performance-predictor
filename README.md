# Student Performance Analysis & Score Predictor

A Python project that analyses student academic data to find patterns and predict whether a student will **Pass or Fail** using Machine Learning.

I made this project to learn data analysis and machine learning. This was my first time using Pandas and Scikit-learn properly.
---

## What This Project Does

- Loads and cleans a student dataset using **Pandas**
- Calculates statistics like mean, median, and standard deviation using **NumPy**
- Creates 7 charts and graphs using **Matplotlib** and **Seaborn**
- Trains a **Logistic Regression** model using Scikit-learn to predict pass/fail
- Evaluates the model with accuracy, precision, and recall
- Predicts the result for any new student input

---

## Project Structure

```
student_performance/
│
├── data/
│   └── student_data.csv        # Dataset (50 students)
│
├── outputs/                    # All charts saved here after running
│   ├── 01_pass_fail_count.png
│   ├── 02_studytime_vs_score.png
│   ├── 03_attendance_vs_score.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_score_distribution.png
│   ├── 06_avg_scores_by_result.png
│   └── 07_confusion_matrix.png
│
├── main.py                     # Main script 
├── requirements.txt            # All libraries needed
└── README.md
```

---

## Charts Generated

| Chart | What It Shows |
|---|---|
| Pass vs Fail Count | How many students passed vs failed |
| Study Time vs Final Score | Does more study = better score? |
| Attendance vs Final Score | Does attendance affect result? |
| Correlation Heatmap | Which features are most related |
| Score Distribution | How scores are spread for Pass/Fail |
| Average Scores by Result | Comparison of all scores |
| Confusion Matrix | How accurate the ML model is |

---

## How to Run

**Step 1 — Clone the repository**
```bash
git clone https://github.com/rathorevashi28/student-performance-predictor
cd student-performance-predictor
```

**Step 2 — Install libraries**
```bash
pip install -r requirements.txt
```

**Step 3 — Run the project**
```bash
python main.py
```

All 7 charts will be saved in the `outputs/` folder automatically.

---

## Tech Stack

| Tool | Use |
|---|---|
| Python | Main language |
| Pandas | Loading and cleaning data |
| NumPy | Statistical calculations |
| Matplotlib | Plotting graphs |
| Seaborn | Heatmap and styled charts |
| Scikit-learn | Logistic Regression model |

---

## Dataset

The dataset (`student_data.csv`) contains 50 students with these columns:

| Column | Description |
|---|---|
| name | Student name |
| study_time | Hours studied per day (1–7) |
| attendance | Attendance percentage (35–97%) |
| assignment_score | Assignment marks (out of 100) |
| midterm_score | Mid-term exam marks |
| final_score | Final exam marks |
| result | Pass or Fail |

---

## Model Results

- **Algorithm:** Logistic Regression
- **Train/Test Split:** 80% train, 20% test
- **Features used:** study_time, attendance, assignment_score, midterm_score, final_score
- **Target:** result (Pass / Fail)
Note: Accuracy is high because the dataset is small and clean. On real-world messy data it would be lower.
---

## Author

**Vashistha Rathore**
- GitHub: [github.com/rathorevashi28](https://github.com/rathorevashi28)
- LinkedIn: [linkedin.com/in/vashistha-rathore-715050289](https://www.linkedin.com/in/vashistha-rathore-715050289/)

# Hotel Booking Demand --- Machine Learning Capstone Project

An end-to-end machine learning project using the **Hotel Booking
Demand** dataset. The project demonstrates exploratory data analysis
(EDA), data cleaning and preprocessing, regression, classification, and
unsupervised clustering.

## Project Overview

The project investigates three machine learning tasks using hotel
booking information:

1.  **Regression:** predict the Average Daily Rate (`adr`) using Linear
    Regression.
2.  **Classification:** predict whether a booking is canceled
    (`is_canceled`) using Logistic Regression.
3.  **Clustering:** discover groups of similar bookings using K-Means
    without using the cancellation target.

The workflow also documents data-quality issues and uses visualizations
to compare data before and after selected preprocessing operations.

## Dataset

-   **Name:** Hotel Booking Demand
-   **Rows:** 119,390
-   **Columns:** 32
-   **Source:** [Hotel Booking Demand ---
    Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
-   **Research paper:** [Hotel booking demand datasets --- Data in Brief
    (2019)](https://doi.org/10.1016/j.dib.2018.11.126)

Download `hotel_bookings.csv` from Kaggle and place it in the `data/`
folder when running the project locally.

## Repository Structure

``` text
ML-Capstone-Project/
├── data/
│   └── hotel_bookings.csv
├── notebooks/
│   └── Hotel_Booking.ipynb
├── results/
│   └── project_report.pdf
├── README.md
└── project_report.pdf
```

> Keep the executed notebook in `notebooks/` and the final PDF report in
> the repository root (or in `results/`). Update the paths above if your
> final GitHub folder structure is different.

## Technologies and Libraries

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Jupyter Notebook / Google Colab

## How to Run

### Option 1 --- Google Colab

Open the executed notebook using the link below:

**[Open Hotel Booking Notebook in Google
Colab](https://colab.research.google.com/drive/1GZVaOLJd5r8-bfddFfuZo5UdGGhXli-K?usp=sharing)**

Make sure the notebook is shared with **Anyone with the link ---
Viewer**. If you want to rerun it, upload `hotel_bookings.csv` when
prompted, or update the CSV path to match your Colab session.

### Option 2 --- Run Locally

1.  Clone this repository:

    ``` bash
    git clone <YOUR_PUBLIC_GITHUB_REPOSITORY_URL>
    cd ML-Capstone-Project
    ```

2.  Create and activate a virtual environment:

    **Windows**

    ``` bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

    **macOS / Linux**

    ``` bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the required libraries:

    ``` bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```

4.  Place `hotel_bookings.csv` in the `data/` folder.

5.  Open the notebook:

    ``` bash
    jupyter notebook
    ```

6.  In the notebook's data-loading cell, use the local path:

    ``` python
    file_path = "data/hotel_bookings.csv"
    df_raw = pd.read_csv(file_path)
    df = df_raw.copy()
    ```

Run the notebook cells from top to bottom.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

-   Inspect dataset dimensions, data types, and descriptive statistics.
-   Identify missing values and exact duplicate rows.
-   Review numerical distributions and potential outliers.
-   Explore booking cancellation distribution and cancellation rate by
    hotel type.
-   Visualize important findings.

### 2. Data Cleaning and Preprocessing

-   Retain an untouched raw-data copy for comparison.
-   Remove exact duplicate rows from the working dataset.
-   Convert negative ADR values to missing values.
-   Apply missing-value imputation inside the model pipelines.
-   Encode categorical features using One-Hot Encoding.
-   Scale numerical features using StandardScaler.
-   Compare selected data conditions before and after preprocessing.

### 3. Regression --- Linear Regression

-   **Target:** `adr`
-   **Evaluation metrics:** MSE, RMSE, and R².
-   **Visualizations:** Actual vs. Predicted plot and residual plot.

### 4. Classification --- Logistic Regression

-   **Target:** `is_canceled`
-   **Evaluation metrics:** Accuracy, Precision, Recall, Confusion
    Matrix, and ROC-AUC.
-   **Visualizations:** Confusion Matrix and ROC Curve.

### 5. Clustering --- K-Means

-   Uses selected booking-behaviour features without the cancellation
    label.
-   Uses the Elbow Method to inspect candidate cluster counts.
-   Uses the Silhouette Score to evaluate cluster separation.
-   Uses PCA to visualize clusters in two dimensions.

## Results Summary

The following baseline results were obtained from the executed notebook:

  Task             Algorithm                                 Result
  ---------------- --------------------- --------------------------
  Regression       Linear Regression                   MSE = 998.01
  Regression       Linear Regression                   RMSE = 31.59
  Regression       Linear Regression                     R² = 0.639
  Classification   Logistic Regression            Accuracy = 79.22%
  Classification   Logistic Regression           Precision = 67.33%
  Classification   Logistic Regression              Recall = 47.43%
  Classification   Logistic Regression              ROC-AUC = 0.842
  Clustering       K-Means (K = 4)         Silhouette Score = 0.227

These are baseline results from the current notebook run. The
classification recall indicates that the model misses a proportion of
canceled bookings at the default decision threshold. The Silhouette
Score indicates modest separation among the discovered clusters.

## Questions and Answers

The notebook includes ten data-driven questions with answers calculated
from the dataset, covering:

-   Dataset size and missing values
-   Duplicate records
-   Cancellation rate
-   Cancellation rate by hotel type
-   Median lead time and ADR
-   Most frequent market segment and customer type
-   Average number of special requests

## Notes and Limitations

-   Exact duplicate rows were removed based on exact row equality. A
    production booking system should ideally validate duplicates using a
    unique booking identifier.
-   The high positive ADR value was retained; no additional outlier
    capping or removal was performed.
-   Missing values are imputed within model pipelines, fitted on
    training data, to reduce data leakage.
-   `reservation_status` and `reservation_status_date` are excluded from
    cancellation prediction to avoid target leakage.
-   The models are baseline implementations and were not extensively
    tuned.
-   K-Means clusters are exploratory groups; the Silhouette Score
    suggests only modest separation.

## Submission Checklist

-   [ ] Public GitHub repository created.
-   [ ] Dataset added to `data/` (if permitted by the dataset license
    and repository size constraints).
-   [ ] Executed notebook added to `notebooks/`.
-   [ ] Final `project_report.pdf` added to the repository.
-   [ ] Google Colab link included in the PDF report.
-   [ ] Colab sharing set to **Anyone with the link --- Viewer**.
-   [ ] README paths updated to match the final repository structure.

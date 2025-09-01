# Breast Cancer Detection using Machine Learning

## 1. Overview

This project is a classic machine learning classification problem focused on detecting breast cancer based on diagnostic medical measurements. Using the Wisconsin Breast Cancer dataset, a K-Nearest Neighbors (KNN) model is trained and evaluated to classify tumors as either malignant (cancerous) or benign (non-cancerous).

The primary goal is to demonstrate a complete machine learning workflow, from data cleaning and exploratory analysis to model training, evaluation, and refinement using feature scaling.

## 2. Dataset

The dataset used is the **Wisconsin Breast Cancer dataset**. It contains 569 instances and 30 numeric features, which are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

- **Source:** The data file is included in the `/data` directory of this repository.
- **Features:** Include measurements like mean radius, mean texture, mean perimeter, mean area, etc.
- **Target Variable:** `diagnosis` (M = malignant, B = benign).

## 3. Project Workflow

The project is structured in a single Jupyter Notebook (`01_Data_Exploration.ipynb`) and follows these key steps:

1.  **Data Loading & Initial Inspection:** The dataset is loaded using Pandas and a health check is performed with `.info()` and `.head()`.
2.  **Data Cleaning:** An empty, irrelevant column (`Unnamed: 32`) is identified and removed. The categorical `diagnosis` column ('M'/'B') is encoded into numerical format (1/0).

3.  **Exploratory Data Analysis (EDA):** Visualizations like count plots, histograms, and correlation heatmaps are used to understand feature distributions and their relationships with the target variable.

4.  **Data Preparation:** The dataset is split into features (X) and target (y), and then further divided into training (80%) and testing (20%) sets.
5.  **Model Training & Evaluation:** A K-Nearest Neighbors (KNN) classifier is trained on the training data.
6.  **Model Improvement:** Feature scaling is applied using `StandardScaler` to normalize the feature ranges, and the model is re-trained and re-evaluated to demonstrate performance improvement.

## 4. Results & Key Findings

The model's performance was evaluated before and after feature scaling, with significant improvements observed after scaling.

-   **Initial Model Accuracy:** ~94.7%
-   **Final Model Accuracy (with Feature Scaling):** **96.5%**

Most importantly, the scaled model achieved a **perfect recall of 1.00 for the malignant class**, meaning it correctly identified all actual malignant tumors in the test set. This highlights the critical importance of preprocessing steps like feature scaling in machine learning.

## 5. Technologies Used

-   **Python**
-   **Pandas:** For data manipulation and analysis.
-   **NumPy:** For numerical operations.
-   **Matplotlib & Seaborn:** For data visualization.
-   **Scikit-learn:** For model building, preprocessing, and evaluation.
-   **Jupyter Notebook:** For interactive development.

## 6. How to Run This Project

1.  Clone this repository to your local machine.
2.  Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate   # On Windows
    ```
3.  Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
4.  Open the Jupyter Notebook `01_Data_Exploration.ipynb` and run the cells sequentially.
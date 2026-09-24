# Impact of AI & Social Media on Students

### A Data Analytics Project on Digital Behaviour, Mental Health, and Student Wellness

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Overview

This project explores the relationship between Artificial Intelligence (AI) tool usage, social media consumption, and student well-being.

Using a dataset of **16,000 students aged 13–25**, the project applies data cleaning, exploratory data analysis (EDA), feature engineering, correlation analysis, and data visualization to identify patterns in digital behaviour, sleep, physical activity, mental health, and overall wellness.

The objective is to understand how different forms of digital engagement relate to student health and derive data-driven insights that can support student wellness initiatives.

## Objectives

* Analyse daily social media and AI tool usage among students.
* Investigate the relationship between screen time, sleep, and mental health.
* Compare digital behaviour and health outcomes across gender and education levels.
* Develop meaningful features to measure digital balance and overall wellness.
* Present actionable insights through statistical analysis and visualizations.

## Dataset

| Attribute         | Details                          |
| ----------------- | -------------------------------- |
| Total records     | 16,000 students                  |
| Age range         | 13–25 years                      |
| Education levels  | High School, College, University |
| Missing values    | 0                                |
| Duplicate records | 0                                |
| Features          | Raw and engineered variables     |

### Key Features

* Demographics: Age, gender, education level.
* Digital behaviour: Daily social media hours, AI tool usage hours.
* Lifestyle: Sleep duration and physical activity.
* Health metrics: Mental health and physical health scores (0–100).

### Feature Engineering

Three additional features were developed:

| Feature               | Description                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| Total Screen Time     | Combined daily social media and AI tool usage hours.                                                         |
| Digital Balance Ratio | Proportion of total screen time spent using AI tools.                                                        |
| Wellness Index        | Equal-weighted composite of normalized sleep, physical activity, mental health, and physical health metrics. |

## Tech Stack

| Technology       | Purpose                                        |
| ---------------- | ---------------------------------------------- |
| Python           | Data analysis and processing                   |
| Pandas           | Data cleaning, manipulation, and aggregation   |
| NumPy            | Numerical computations and feature engineering |
| Matplotlib       | Statistical charts and visualizations          |
| Jupyter Notebook | Exploratory analysis and experimentation       |

## Key Findings

The analysis revealed several notable patterns:

| Metric                     | Finding          |
| -------------------------- | ---------------- |
| Average social media usage | 4.54 hours/day   |
| Average AI tool usage      | 2.59 hours/day   |
| Average total screen time  | 7.13 hours/day   |
| Average sleep              | 6.55 hours/night |
| Average Wellness Index     | 62.73/100        |

### 1. Physical Activity and Wellness

Physical activity showed the strongest positive correlation with the Wellness Index (r = +0.785), followed by sleep duration (r = +0.541).

### 2. Social Media vs AI Tool Usage

Social media usage had a negative correlation with mental health scores (r = −0.404), while AI tool usage showed a weaker negative correlation (r = −0.111).

This highlights differences in the observed associations of different types of digital engagement.

### 3. Screen Time and Sleep

Total screen time was negatively correlated with sleep duration (r = −0.324), suggesting an association between higher digital engagement and reduced sleep.

### 4. Student Wellness Across Demographics

Health and wellness metrics were broadly similar across education levels and age groups. The dataset also showed small differences in average wellness and health scores across gender groups.

*Note: Correlation indicates association, not causation. Findings describe patterns within the analysed dataset and should not be interpreted as proof of direct causal effects.*

## Data Visualizations

The project includes eight visualizations designed to communicate the analysis and answer specific research questions.

| No. | Visualization                      | Purpose                                                              |
| --- | ---------------------------------- | -------------------------------------------------------------------- |
| 01  | Correlation Heatmap                | Examine relationships between digital usage and health metrics.      |
| 02  | Screen Time vs Health Scores       | Explore screen time associations with mental and physical health.    |
| 03  | Digital Usage vs Mental Health     | Compare social media and AI usage by gender.                         |
| 04  | AI Usage & Sleep by Education      | Compare average AI usage and sleep duration across education levels. |
| 05  | Health Scores by Gender            | Visualize mental and physical health score differences.              |
| 06  | Wellness Index Boxplot             | Examine wellness distributions and potential outliers.               |
| 07  | Engineered Feature Histograms      | Analyse distributions of screen time, digital balance, and wellness. |
| 08  | Screen Time & Mental Health by Age | Examine trends across ages 13–25.                                    |

## Project Structure

The repository can be organized as follows:

```text
Impact-of-AI-and-Social-Media-on-Students/
│
├── AI_SocialMedia_Student_Dataset.csv
├── notebooks/
│   └── student_wellness_analysis.ipynb
│
├── charts/
│   ├── 01_correlation_heatmap.png
│   ├── 02_scatter_screentime_vs_health.png
│   ├── 03_scatter_digital_vs_mental_health.png
│   ├── 04_bar_ai_sleep_by_education.png
│   ├── 05_bar_health_by_gender.png
│   ├── 06_boxplot_wellness_by_education.png
│   ├── 07_histograms_engineered_features.png
│   └── 08_line_screentime_mentalhealth_by_age.png
│
├── README.md
└── requirements.txt
```

*The structure above is a suggested organization. Adjust the paths and filenames to match the actual repository.*

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the repository URL with your GitHub repository link.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install the main libraries directly:

```bash
pip install pandas numpy matplotlib jupyter
```

### 3. Run the Analysis

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the analysis notebook and execute the cells to explore the dataset, perform analysis, and generate visualizations.

## Key Takeaways

* Physical activity and sleep show substantial positive associations with student wellness.
* Social media usage has a stronger negative association with mental health than AI tool usage in this dataset.
* Screen time, sleep, and wellness are interconnected dimensions of student lifestyle.
* Demographic comparisons provide additional context for understanding student digital behaviour.

## Future Scope

* Develop an interactive dashboard using Power BI, Tableau, or Streamlit.
* Apply machine learning models to investigate wellness patterns and identify potentially at-risk student groups.
* Expand the analysis using additional datasets and longitudinal student data.
* Incorporate predictive analytics and interactive filters for demographic comparisons.

## Conclusion

This project demonstrates how data analytics can be used to investigate the relationship between technology usage and student wellness.

By combining exploratory data analysis, feature engineering, correlation analysis, and visual storytelling, the project transforms student behavioural data into interpretable insights that can inform further research and wellness initiatives.

---

**Author:** Sneha Mittra
**Project Domain:** Data Analytics | Exploratory Data Analysis | Student Wellness
**Tools:** Python, Pandas, NumPy, Matplotlib

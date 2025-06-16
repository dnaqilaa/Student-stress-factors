
# Student Stress Factors Analysis

This project analyzes a dataset of student stress factors using Python. It explores various dimensions contributing to student stress—such as psychological, physiological, environmental, academic, and social factors—through descriptive statistics, visualizations, and machine learning.

## 📁 Dataset
The dataset is sourced from Kaggle:  
**Student Stress Factors - A Comprehensive Analysis**  
The CSV file analyzed is: `StressLevelDataset.csv`

## 📊 Key Features

- Descriptive statistics to summarize student conditions.
- Categorized factor analysis:
  - Psychological
  - Physiological
  - Environmental
  - Academic
  - Social
- Data visualizations:
  - Bar charts
  - Pair plots
  - Correlation heatmaps
  - Box plots
- Machine learning (Random Forest Regressor) to determine feature importance for stress level prediction.

## 🧪 Main Technologies Used

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- KaggleHub (for dataset download)

## 📂 File Structure

- `student_stress_factors.py`: Main script performing data import, analysis, visualizations, and feature importance extraction.

## 🧬 Analysis Sections

1. **Descriptive Statistics**  
   - Total number of students  
   - Average anxiety levels  
   - History of mental health issues

2. **Factor-Level Analysis**  
   - Negative experiences grouped into five factor categories  
   - Key metrics like depression rate, sleep quality, academic concerns

3. **Comparative Correlation**  
   - Relationships between anxiety, academic performance, depression, and sleep

4. **Feature Importance (ML)**  
   - Random Forest used to identify which features most strongly predict stress

5. **Visualizations**  
   - Heatmap of correlations  
   - Pair plots for key features  
   - Box plots and bar charts for distributions and importance

## 📈 How to Run

1. Clone or download this repository.
2. Make sure you have the required Python libraries installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn kagglehub
   ```
3. Run the script:
   ```bash
   python student_stress_factors.py
   ```

> 💡 Note: This script was originally developed and run in a Kaggle or Colab environment. You may need to adjust dataset paths if running locally.

## 📌 Author

- Aqila Afifah
- Dataset credit: Kaggle user `rxnach`

## 📝 License

This project is for educational and non-commercial research use only.

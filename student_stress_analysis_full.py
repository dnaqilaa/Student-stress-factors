import os
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

def download_and_load_data():
    try:
        dataset_path = kagglehub.dataset_download('rxnach/student-stress-factors-a-comprehensive-analysis')
        csv_file_path = os.path.join(dataset_path, "StressLevelDataset.csv")
        df = pd.read_csv(csv_file_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")

def descriptive_statistics(df):
    return {
        "num_students": len(df),
        "average_anxiety_level": df['anxiety_level'].mean(),
        "num_mental_health_history": len(df[df['mental_health_history'] == 1])
    }

def psychological_factors(df):
    avg_esteem = df['self_esteem'].mean()
    df['depression_binary'] = (df['depression'] > 10).astype(int)
    return {
        "low_self_esteem_count": len(df[df['self_esteem'] < avg_esteem]),
        "depression_percent": (df['depression_binary'].sum() / len(df)) * 100
    }

def physiological_factors(df):
    return {
        "frequent_headaches": len(df[df['headache'] >= 4]),
        "avg_blood_pressure": df['blood_pressure'].mean(),
        "poor_sleep_quality": len(df[df['sleep_quality'] <= 2])
    }

def environmental_factors(df):
    df['feeling_unsafe'] = (df['safety'] <= 2).astype(int)
    return {
        "high_noise": len(df[df['noise_level'] > 3]),
        "feeling_unsafe_percent": (df['feeling_unsafe'].sum() / len(df)) * 100,
        "basic_needs_unmet": len(df[df['basic_needs'] == 0])
    }

def academic_factors(df):
    return {
        "low_academic_perf": len(df[df['academic_performance'] <= 2]),
        "avg_study_load": df['study_load'].mean(),
        "career_concerns": len(df[df['future_career_concerns'] == 1])
    }

def social_factors(df):
    df['bullying_binary'] = (df['bullying'] > 3).astype(int)
    return {
        "strong_social_support": len(df[df['social_support'] >= 4]),
        "bullying_percent": (df['bullying_binary'].sum() / len(df)) * 100,
        "extracurricular": len(df[df['extracurricular_activities'] == 1])
    }

def comparative_analysis(df):
    bullying_threshold = 3
    bullied = df[df['bullying'] > bullying_threshold]
    if len(bullied) == 0:
        bullied_mh_percent = None
    else:
        bullied_with_history = bullied[bullied['mental_health_history'] == 1]
        bullied_mh_percent = (len(bullied_with_history) / len(bullied)) * 100

    return {
        "anxiety_academic_corr": df['anxiety_level'].corr(df['academic_performance']),
        "sleep_depression_corr": df['sleep_quality'].corr(df['depression']),
        "bullied_mh_percent": bullied_mh_percent
    }

def generate_visualizations(df):
    corr_matrix = df[['anxiety_level', 'self_esteem', 'depression', 'sleep_quality', 'academic_performance', 'stress_level']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap of Key Factors")
    plt.savefig('factors_correlation_heatmap.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df[['anxiety_level', 'self_esteem', 'depression', 'sleep_quality', 'academic_performance']], orient='h')
    plt.title("Box Plots of Key Factors")
    plt.xlabel("Factor Values")
    plt.savefig('factors_boxplots.png', bbox_inches='tight')
    plt.close()

def feature_importance_analysis(df):
    if 'stress_level' not in df.columns:
        return None

    factors_defs = {
        'Psychological': ['anxiety_level', 'self_esteem', 'mental_health_history', 'depression'],
        'Physiological': ['headache', 'blood_pressure', 'sleep_quality', 'breathing_problem'],
        'Environmental': ['noise_level', 'living_conditions', 'safety', 'basic_needs'],
        'Academic': ['academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns'],
        'Social': ['social_support', 'peer_pressure', 'extracurricular_activities', 'bullying']
    }

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    result = {}

    for factor, features in factors_defs.items():
        X = df[features]
        y = df['stress_level']
        model.fit(X, y)
        importances = model.feature_importances_
        feature_importance = dict(zip(features, importances))
        result[factor] = feature_importance

        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(feature_importance.values()), y=list(feature_importance.keys()), hue=list(feature_importance.keys()), dodge=False, legend=False, palette='viridis')
        plt.title(f"Feature Importance: {factor}")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.savefig(f"feature_importance_{factor}.png", bbox_inches='tight')
        plt.close()

    return result

def main():
    print("Downloading and loading data...")
    df = download_and_load_data()

    print("\n--- Descriptive Statistics ---")
    stats = descriptive_statistics(df)
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print("\n--- Psychological Factors ---")
    psych = psychological_factors(df)
    for key, value in psych.items():
        print(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")

    print("\n--- Physiological Factors ---")
    phys = physiological_factors(df)
    for key, value in phys.items():
        print(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")

    print("\n--- Environmental Factors ---")
    env = environmental_factors(df)
    for key, value in env.items():
        print(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")

    print("\n--- Academic Factors ---")
    acad = academic_factors(df)
    for key, value in acad.items():
        print(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")

    print("\n--- Social Factors ---")
    soc = social_factors(df)
    for key, value in soc.items():
        print(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")

    print("\n--- Comparative Analysis ---")
    comp = comparative_analysis(df)
    for key, value in comp.items():
        if value is None:
            print(f"{key.replace('_', ' ').title()}: No data")
        else:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")

    print("\n--- Generating Visualizations ---")
    generate_visualizations(df)
    print("Visualizations saved as PNG files.")

    print("\n--- Feature Importance Analysis ---")
    importance = feature_importance_analysis(df)
    if importance:
        for factor, feats in importance.items():
            print(f"\n{factor} Factor Importance:")
            for feature, score in sorted(feats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {score:.4f}")
    else:
        print("Skipped: 'stress_level' column not found.")

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()
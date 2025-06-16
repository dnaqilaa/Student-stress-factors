import unittest
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Sample functions extracted and simplified from the main script
def calculate_average_anxiety(df):
    return df['anxiety_level'].mean()

def count_students_with_mental_health_history(df):
    return len(df[df['mental_health_history'] == 1])

def calculate_feature_importance(df, features, target='stress_level'):
    X = df[features]
    y = df[target]
    model = RandomForestRegressor()
    model.fit(X, y)
    return model.feature_importances_

class TestStudentStressFactors(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'anxiety_level': [3, 4, 5],
            'mental_health_history': [0, 1, 1],
            'stress_level': [10, 20, 30],
            'self_esteem': [2, 3, 4]
        })

    def test_average_anxiety(self):
        self.assertAlmostEqual(calculate_average_anxiety(self.df), 4.0)

    def test_mental_health_history_count(self):
        self.assertEqual(count_students_with_mental_health_history(self.df), 2)

    def test_feature_importance_length(self):
        features = ['anxiety_level', 'self_esteem']
        importances = calculate_feature_importance(self.df, features)
        self.assertEqual(len(importances), 2)

if __name__ == '__main__':
    unittest.main()
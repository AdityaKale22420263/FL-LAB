"""
Generate Synthetic Student Performance Dataset
This creates a realistic student performance dataset for federated learning
"""

import pandas as pd
import numpy as np

def generate_student_performance_dataset(num_students=500, seed=42):
    """
    Generate a synthetic student performance dataset
    
    Parameters:
    -----------
    num_students : int
        Number of student records to generate
    seed : int
        Random seed for reproducibility
    """
    np.random.seed(seed)
    
    print("Generating synthetic student performance dataset...")
    
    # Student demographics
    student_ids = [f'S{str(i).zfill(4)}' for i in range(1, num_students + 1)]
    genders = np.random.choice(['Male', 'Female'], num_students)
    ages = np.random.randint(18, 25, num_students)
    
    # Previous education
    previous_grades = np.random.choice(['A', 'B', 'C', 'D'], num_students, p=[0.15, 0.35, 0.35, 0.15])
    
    # Study habits and behavior
    study_hours = np.random.gamma(2, 2, num_students)  # Gamma distribution for study hours
    study_hours = np.clip(study_hours, 0, 15)  # Clip to 0-15 hours per day
    
    attendance = np.random.beta(8, 2, num_students) * 100  # Beta distribution for attendance
    attendance = np.clip(attendance, 50, 100)  # Clip to 50-100%
    
    participation = np.random.beta(5, 3, num_students) * 100
    participation = np.clip(participation, 30, 100)
    
    # Assignment and quiz scores
    assignment_scores = np.random.beta(6, 2, num_students) * 100
    quiz_scores = np.random.beta(6, 2.5, num_students) * 100
    
    # Internal marks (midterm)
    internal_marks = np.random.beta(6, 2, num_students) * 50
    internal_marks = np.clip(internal_marks, 10, 50)
    
    # Lifestyle factors
    sleep_hours = np.random.normal(7, 1.5, num_students)
    sleep_hours = np.clip(sleep_hours, 4, 10)
    
    exercise_hours = np.random.exponential(1.5, num_students)
    exercise_hours = np.clip(exercise_hours, 0, 5)
    
    # Family and socioeconomic factors
    parental_education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                         num_students, p=[0.3, 0.4, 0.2, 0.1])
    family_support = np.random.choice(['Low', 'Medium', 'High'], 
                                     num_students, p=[0.2, 0.5, 0.3])
    
    internet_access = np.random.choice(['Yes', 'No'], num_students, p=[0.85, 0.15])
    
    # Extra curricular activities
    extracurricular = np.random.choice(['Yes', 'No'], num_students, p=[0.6, 0.4])
    
    # Part-time job
    part_time_job = np.random.choice(['Yes', 'No'], num_students, p=[0.4, 0.6])
    
    # Calculate final score based on various factors with realistic correlations
    base_score = (
        study_hours * 3.5 +
        attendance * 0.3 +
        participation * 0.15 +
        assignment_scores * 0.2 +
        quiz_scores * 0.15 +
        internal_marks * 0.4 +
        sleep_hours * 1.2 +
        exercise_hours * 0.8
    )
    
    # Add adjustments based on categorical variables
    grade_adjustment = {'A': 10, 'B': 5, 'C': 0, 'D': -5}
    base_score += np.array([grade_adjustment[g] for g in previous_grades])
    
    education_adjustment = {'High School': 0, 'Bachelor': 3, 'Master': 6, 'PhD': 9}
    base_score += np.array([education_adjustment[e] for e in parental_education])
    
    support_adjustment = {'Low': -3, 'Medium': 0, 'High': 3}
    base_score += np.array([support_adjustment[s] for s in family_support])
    
    # Adjustments for binary factors
    base_score += (internet_access == 'Yes') * 5
    base_score += (extracurricular == 'Yes') * 2
    base_score -= (part_time_job == 'Yes') * 3
    
    # Add some random noise
    noise = np.random.normal(0, 5, num_students)
    final_score = base_score + noise
    
    # Normalize to 0-100 range
    final_score = (final_score - final_score.min()) / (final_score.max() - final_score.min()) * 100
    final_score = np.clip(final_score, 0, 100)
    
    # Introduce some missing values randomly (5% of data)
    def add_missing_values(arr, missing_rate=0.05):
        arr_copy = arr.copy()
        mask = np.random.random(len(arr_copy)) < missing_rate
        arr_copy[mask] = np.nan
        return arr_copy
    
    # Create DataFrame
    df = pd.DataFrame({
        'student_id': student_ids,
        'gender': genders,
        'age': ages,
        'previous_grade': previous_grades,
        'study_hours': add_missing_values(study_hours),
        'attendance': add_missing_values(attendance),
        'participation': add_missing_values(participation),
        'assignment_scores': add_missing_values(assignment_scores),
        'quiz_scores': add_missing_values(quiz_scores),
        'internal_marks': add_missing_values(internal_marks),
        'sleep_hours': add_missing_values(sleep_hours),
        'exercise_hours': add_missing_values(exercise_hours),
        'parental_education': parental_education,
        'family_support': family_support,
        'internet_access': internet_access,
        'extracurricular': extracurricular,
        'part_time_job': part_time_job,
        'final_score': final_score
    })
    
    # Round numerical columns
    numeric_columns = ['study_hours', 'attendance', 'participation', 'assignment_scores', 
                      'quiz_scores', 'internal_marks', 'sleep_hours', 'exercise_hours', 'final_score']
    for col in numeric_columns:
        df[col] = df[col].round(2)
    
    return df


if __name__ == "__main__":
    # Generate dataset
    df = generate_student_performance_dataset(num_students=500, seed=42)
    
    # Save to CSV
    output_file = 'student_performance.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Dataset generated successfully!")
    print(f"  File: {output_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    print("\n" + "="*60)
    print("Dataset Preview:")
    print("="*60)
    print(df.head(10))
    
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(df.describe())
    
    print("\n" + "="*60)
    print("Missing Values:")
    print("="*60)
    print(df.isnull().sum())
    
    print("\n✓ Dataset ready for federated learning preprocessing!")
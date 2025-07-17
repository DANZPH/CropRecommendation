"""
Model Comparison Script
Run this to see which ensemble method works best for your data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def compare_models():
    # Load data
    print("Loading data...")
    data = pd.read_csv('Crop_recommendation.csv')
    
    # Prepare data
    le = LabelEncoder()
    data['label_encoded'] = le.fit_transform(data['label'])
    
    X = data.drop(['label', 'label_encoded'], axis=1)
    y = data['label_encoded']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Voting (Hard)': VotingClassifier(
            estimators=[
                ('knn', KNeighborsClassifier(n_neighbors=5)),
                ('dt', DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
            ],
            voting='hard'
        ),
        'Voting (Soft)': VotingClassifier(
            estimators=[
                ('knn', KNeighborsClassifier(n_neighbors=5)),
                ('dt', DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
            ],
            voting='soft'
        ),
        'Stacking': StackingClassifier(
            estimators=[
                ('knn', KNeighborsClassifier(n_neighbors=5)),
                ('dt', DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5))
            ],
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5
        )
    }
    
    print("\nTraining and evaluating models...")
    print("=" * 60)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Test accuracy
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'Test Accuracy': test_accuracy,
            'CV Mean': cv_mean,
            'CV Std': cv_std
        }
        
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  CV Accuracy: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)
    
    print("\n" + "=" * 60)
    print("FINAL COMPARISON:")
    print("=" * 60)
    print(comparison_df)
    
    # Find best model
    best_model = comparison_df['Test Accuracy'].idxmax()
    best_accuracy = comparison_df.loc[best_model, 'Test Accuracy']
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   Test Accuracy: {best_accuracy:.4f}")
    print(f"   CV Accuracy: {comparison_df.loc[best_model, 'CV Mean']:.4f}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    # Sort by test accuracy
    sorted_models = comparison_df.sort_values('Test Accuracy', ascending=False)
    
    print("Ranking by Test Accuracy:")
    for i, (model, row) in enumerate(sorted_models.iterrows(), 1):
        print(f"{i}. {model}: {row['Test Accuracy']:.4f}")
    
    print(f"\n💡 For your Streamlit app, use: '{best_model}'")
    
    return best_model, comparison_df

if __name__ == "__main__":
    best_model, results = compare_models()
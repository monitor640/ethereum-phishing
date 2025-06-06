import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
def load_and_prepare_data():
    embeddings = torch.load('./node2vec_embeddings_larger.pt', map_location=torch.device('cpu'))  # or wherever your embeddings are
    data = torch.load('../data_simple_oversampled.pt', map_location=torch.device('cpu'), weights_only=False)
    print(data)
    labels = data.y.squeeze()
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Class distribution: {torch.bincount(labels)}")
    
    X = embeddings.detach().cpu().numpy()  
    y = labels.detach().cpu().numpy()      
    
    print(f"Converted X shape: {X.shape}, dtype: {X.dtype}")
    print(f"Converted y shape: {y.shape}, dtype: {y.dtype}")
    
    return X, y




def compare_multiple_classifiers():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    print("\n=== Comparing Multiple Classifiers ===")
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {'accuracy': accuracy, 'f1': f1}
        
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Check if it predicts both classes
        pred_counts = np.bincount(y_pred)
        pred_0 = pred_counts[0] if len(pred_counts) > 0 else 0
        pred_1 = pred_counts[1] if len(pred_counts) > 1 else 0
        print(f"  Predictions - Normal: {pred_0}, Phishing: {pred_1}")

        print(f"\n=== Detailed Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Phishing']))
        
        # Confusion matrix
        print(f"\n=== Confusion Matrix ===")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print("[[TN, FP],")
        print(" [FN, TP]]")

        #plot roc curve with sns
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
        sns.lineplot(x=fpr, y=tpr)
        plt.show()

    
    # Find best classifier
    best_clf = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest classifier: {best_clf[0]} (F1: {best_clf[1]['f1']:.4f})")
    
    return results

if __name__ == '__main__':
    compare_multiple_classifiers()
   
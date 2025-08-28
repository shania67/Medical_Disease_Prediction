import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and print metrics"""
    y_pred = model.predict(X_test)
    
    print("✅ Model Evaluation Results")
    print("----------------------------")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    try:
        print(f"ROC-AUC:   {roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr'):.4f}")
    except:
        print("ROC-AUC:   Not available for this model")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

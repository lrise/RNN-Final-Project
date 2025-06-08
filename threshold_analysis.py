#!/usr/bin/env python3
"""
Threshold Optimization Script - Find the Optimal Toxicity Detection Threshold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import argparse

def analyze_thresholds(csv_path, output_path="threshold_analysis.png"):
    """
    Analyze performance under different thresholds
    """
    # Read results
    df = pd.read_csv(csv_path)
    
    print(f"Loaded data: {len(df)} samples")
    
    # Extract ground truth and prediction scores
    y_true = df['JailbreakSuccess'].values
    y_scores = df['DefenseToxicityScore'].values
    
    # Test different thresholds
    thresholds = np.arange(0.1, 0.9, 0.02)  # From 0.1 to 0.9, step 0.02
    
    results = []
    
    for threshold in thresholds:
        # Predict based on threshold
        y_pred = (y_scores >= threshold).astype(int)
        
        # Confusion matrix
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Metrics
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'fpr': fpr,
            'fnr': fnr,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    results_df = pd.DataFrame(results)
    
    # Find best thresholds
    best_f1_idx = results_df['f1'].idxmax()
    best_precision_idx = results_df['precision'].idxmax()
    best_balanced_idx = results_df.apply(
        lambda row: row['precision'] * 0.6 + row['recall'] * 0.4,  # Precision-weighted
        axis=1
    ).idxmax()
    
    print("\n🎯 Threshold Optimization Recommendation:")
    print("="*50)
    
    current_threshold = 0.4
    current_idx = results_df.iloc[(results_df['threshold'] - current_threshold).abs().argsort()[:1]].index[0]
    current_result = results_df.iloc[current_idx]
    
    print(f"Current Threshold {current_threshold:.2f}:")
    print(f"  Accuracy: {current_result['accuracy']:.4f}")
    print(f"  Precision: {current_result['precision']:.4f}")
    print(f"  Recall: {current_result['recall']:.4f}")
    print(f"  F1 Score: {current_result['f1']:.4f}")
    print(f"  False Positives: {current_result['fp']}")
    
    print(f"\nRecommended Threshold {results_df.iloc[best_balanced_idx]['threshold']:.2f} (Balanced Precision and Recall):")
    best_result = results_df.iloc[best_balanced_idx]
    print(f"  Accuracy: {best_result['accuracy']:.4f} ({'↑' if best_result['accuracy'] > current_result['accuracy'] else '↓'}{abs(best_result['accuracy'] - current_result['accuracy']):.4f})")
    print(f"  Precision: {best_result['precision']:.4f} ({'↑' if best_result['precision'] > current_result['precision'] else '↓'}{abs(best_result['precision'] - current_result['precision']):.4f})")
    print(f"  Recall: {best_result['recall']:.4f} ({'↑' if best_result['recall'] > current_result['recall'] else '↓'}{abs(best_result['recall'] - current_result['recall']):.4f})")
    print(f"  F1 Score: {best_result['f1']:.4f} ({'↑' if best_result['f1'] > current_result['f1'] else '↓'}{abs(best_result['f1'] - current_result['f1']):.4f})")
    print(f"  False Positives: {best_result['fp']} (reduced by {current_result['fp'] - best_result['fp']})")
    
    print(f"\nHigh Precision Threshold {results_df.iloc[best_precision_idx]['threshold']:.2f} (Minimize False Positives):")
    precision_result = results_df.iloc[best_precision_idx]
    print(f"  Accuracy: {precision_result['accuracy']:.4f}")
    print(f"  Precision: {precision_result['precision']:.4f}")
    print(f"  Recall: {precision_result['recall']:.4f}")
    print(f"  F1 Score: {precision_result['f1']:.4f}")
    print(f"  False Positives: {precision_result['fp']} (reduced by {current_result['fp'] - precision_result['fp']})")
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(results_df['threshold'], results_df['accuracy'], 'b-', label='Accuracy', linewidth=2)
    plt.plot(results_df['threshold'], results_df['precision'], 'r-', label='Precision', linewidth=2)
    plt.plot(results_df['threshold'], results_df['recall'], 'g-', label='Recall', linewidth=2)
    plt.plot(results_df['threshold'], results_df['f1'], 'm-', label='F1 Score', linewidth=2)
    plt.axvline(x=results_df.iloc[best_balanced_idx]['threshold'], color='orange', linestyle='--', alpha=0.7, label='Recommended Threshold')
    plt.axvline(x=current_threshold, color='red', linestyle=':', alpha=0.7, label='Current Threshold')
    plt.xlabel('Toxicity Score Threshold')
    plt.ylabel('Score')
    plt.title('Performance vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(results_df['threshold'], results_df['fp'], 'r-', label='False Positives (FP)', linewidth=2)
    plt.plot(results_df['threshold'], results_df['fn'], 'orange', label='False Negatives (FN)', linewidth=2)
    plt.axvline(x=results_df.iloc[best_balanced_idx]['threshold'], color='orange', linestyle='--', alpha=0.7)
    plt.xlabel('Toxicity Score Threshold')
    plt.ylabel('Count')
    plt.title('False Positives vs False Negatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    fpr_roc, tpr_roc, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr_roc, tpr_roc)
    plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_curve, precision_curve)
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.plot(results_df['threshold'], results_df['f1'], 'purple', linewidth=3)
    max_f1_threshold = results_df.iloc[results_df['f1'].idxmax()]['threshold']
    max_f1_value = results_df['f1'].max()
    plt.scatter([max_f1_threshold], [max_f1_value], color='red', s=100, zorder=5)
    plt.axvline(x=max_f1_threshold, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Toxicity Score Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'Best F1 Point (Threshold = {max_f1_threshold:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    balance_scores = results_df['precision'] * 0.6 + results_df['recall'] * 0.4
    plt.plot(results_df['threshold'], balance_scores, 'green', linewidth=2, label='Balance Score (0.6P + 0.4R)')
    best_balance_threshold = results_df.iloc[balance_scores.idxmax()]['threshold']
    plt.axvline(x=best_balance_threshold, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('Toxicity Score Threshold')
    plt.ylabel('Balance Score')
    plt.title('Precision-Oriented Balance Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Analysis chart saved to: {output_path}")
    
    results_df.to_csv("threshold_optimization_results.csv", index=False)
    print("📋 Detailed results saved to: threshold_optimization_results.csv")
    
    return results_df.iloc[best_balanced_idx]['threshold']

def generate_optimized_config(optimal_threshold):
    """
    Generate optimized configuration
    """
    config_code = f'''
# Optimized Configuration
config = {{
    "toxicity_threshold": {optimal_threshold:.3f},  # Optimized threshold
    "harmfulness_threshold": 0.25,  # Slightly reduced
    "enable_ollama_replacement": True,
    "replacement_threshold": "medium", 
    "enable_sanitization": True,
    "strict_mode": False,  # Disable strict mode to reduce false positives
    "risk_weights": {{
        "toxicity": 0.4,        # Increase toxicity weight
        "harmfulness": 0.2,     # Decrease harmfulness weight
        "pattern_risk": 0.2,
        "semantic_risk": 0.1,   # Decrease semantic weight
        "personal_info": 0.1
    }}
}}
'''
    
    with open("optimized_config.py", "w", encoding="utf-8") as f:
        f.write(config_code)
    
    print(f"⚙️ Optimized configuration saved to: optimized_config.py")
    print("📝 Usage:")
    print("   Copy this config to your main script, or import it directly")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Threshold Optimization Tool')
    parser.add_argument('--csv', required=True, help='CSV file from defense system output')
    parser.add_argument('--output', default='threshold_analysis.png', help='Output chart file name')
    
    args = parser.parse_args()
    
    print("🔍 Starting threshold optimization analysis...")
    optimal_threshold = analyze_thresholds(args.csv, args.output)
    
    print(f"\n🎯 Suggested threshold: {optimal_threshold:.3f}")
    generate_optimized_config(optimal_threshold)
    
    print("\n✅ Analysis complete! Next steps:")
    print("1. Check the generated analysis chart")
    print("2. Use the suggested threshold to re-run your defense system")
    print("3. Compare performance before and after optimization")

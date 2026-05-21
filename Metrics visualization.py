import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# =====================================================
# CONFUSION MATRIX HEATMAP

plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['NORMAL', 'PNEUMONIA'],
    yticklabels=['NORMAL', 'PNEUMONIA']
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# =====================================================
# ROC CURVE

fpr, tpr, thresholds = roc_curve(y_true,y_pred_probs)
plt.figure(figsize=(7,6))
plt.plot(fpr,tpr,label=f"AUC = {auc_score:.4f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()
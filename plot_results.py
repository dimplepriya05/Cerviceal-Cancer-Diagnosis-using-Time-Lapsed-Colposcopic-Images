# plot_results.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import joblib   # optional - for loading history if saved with joblib/pickle

# ------------------ user configs ------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))   # current script folder
# adjust these if your dataset is in another relative path
test_data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'test')

model_path = os.path.join(PROJECT_ROOT, 'resnet.h5')        # change if filename different
history_path = os.path.join(PROJECT_ROOT, 'history.pkl')   # optional: saved training history
out_dir = os.path.join(PROJECT_ROOT, 'FIGS')               # folder to save figures
os.makedirs(out_dir, exist_ok=True)

img_height, img_width = 224, 224
batch_size = 1   # using 1 to keep labels aligned; change if you handle steps properly
# ---------------------------------------------------

# 1) load trained model
print("Loading model from:", model_path)
model = load_model(model_path)

# 2) recreate test generator (shuffle=False to preserve order for labels)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    directory=test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # important!
)

# 3) make predictions on entire test set
steps = int(np.ceil(test_generator.samples / test_generator.batch_size))
print("Predicting on test set: steps =", steps, "samples =", test_generator.samples)
preds = model.predict(test_generator, steps=steps, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_generator.classes

# 4) Confusion matrix
if hasattr(test_generator, 'class_indices'):
    idx_to_class = {v: k for k, v in test_generator.class_indices.items()}
    class_labels = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
else:
    class_labels = None

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

plt.figure(figsize=(6,6))
disp.plot(cmap='Blues', ax=plt.gca(), values_format='d')
plt.title('Confusion Matrix')
cm_path = os.path.join(out_dir, 'confusion_matrix.png')
plt.savefig(cm_path, bbox_inches='tight', dpi=200)
plt.close()
print("Saved confusion matrix ->", cm_path)

# 5) ROC curve (binary) or multiclass micro-average
num_classes = preds.shape[1]
if num_classes == 2:
    # Binary ROC (use probability for class 1)
    y_score = preds[:,1]
    y_true_bin = label_binarize(y_true, classes=[0,1]).ravel()
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label='ROC (AUC = %0.3f)' % roc_auc)
    plt.plot([0,1], [0,1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    roc_path = os.path.join(out_dir, 'roc_curve.png')
    plt.savefig(roc_path, bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved ROC curve ->", roc_path)
else:
    # multiclass: compute micro-avg ROC
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), preds.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label='Micro-avg ROC (AUC = %0.3f)' % roc_auc)
    plt.plot([0,1], [0,1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (multiclass, micro-avg)')
    plt.legend(loc='lower right')
    roc_path = os.path.join(out_dir, 'roc_curve_multiclass.png')
    plt.savefig(roc_path, bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved ROC curve ->", roc_path)

# 6) Optional: plot training/validation accuracy & loss if history was saved during training
if os.path.exists(history_path):
    print("Loading training history from:", history_path)
    try:
        history = joblib.load(history_path)   # may be joblib or pickle
    except Exception:
        import pickle
        with open(history_path, 'rb') as f:
            history = pickle.load(f)

    # try keys
    acc_key = 'accuracy' if 'accuracy' in history else ('acc' if 'acc' in history else None)
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history else ('val_acc' if 'val_acc' in history else None)

    # Accuracy plot
    if acc_key and val_acc_key:
        plt.figure()
        plt.plot(history[acc_key], label='Training Accuracy')
        plt.plot(history[val_acc_key], label='Validation Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        acc_path = os.path.join(out_dir, 'accuracy_curve.png')
        plt.savefig(acc_path, bbox_inches='tight', dpi=200)
        plt.close()
        print("Saved accuracy curve ->", acc_path)

    # Loss plot
    if 'loss' in history and 'val_loss' in history:
        plt.figure()
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training vs Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        loss_path = os.path.join(out_dir, 'loss_curve.png')
        plt.savefig(loss_path, bbox_inches='tight', dpi=200)
        plt.close()
        print("Saved loss curve ->", loss_path)
else:
    print("No history file found at", history_path, "- skipping accuracy/loss plotting.")
    print("If you want accuracy/loss plots, save training history as a pickle (see instructions below).")

print("ALL DONE. Figures saved to:", out_dir)

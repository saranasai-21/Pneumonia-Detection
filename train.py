import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
from tensorflow.keras.losses import BinaryFocalCrossentropy

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, f1_score)

# MIXED PRECISION

mixed_precision.set_global_policy('mixed_float16')

# REPRODUCIBILITY

SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# CONFIG

IMG_SIZE = 256
BATCH_SIZE = 16
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 4
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# DATASET PATHS

TRAIN_DIR = r"Downloads\data\train"
VAL_DIR = r"Downloads\data\val"
TEST_DIR = r"Downloads\data\test"

# MODEL PATH

MODEL_PATH = "Downloads/pneumonia_densenet.keras"

# CLAHE PREPROCESSING

def apply_clahe(image):
    image = image.astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return enhanced

# CUSTOM PREPROCESS

def custom_preprocess(image):
    image = apply_clahe(image)
    image = preprocess_input(image)
    return image

# DATA GENERATORS

train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocess,
    rotation_range=7,
    zoom_range=0.08,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.9, 1.1],
    shear_range=0.05,
    horizontal_flip=False,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(preprocessing_function=custom_preprocess)

# LOAD DATA

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True, seed=SEED
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False
)

# CLASS WEIGHTS

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = dict(enumerate(class_weights))

print("\n===== CLASS WEIGHTS =====")
print(class_weights)

# LOAD DENSENET121

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# FREEZE BASE MODEL

base_model.trainable = False

# BUILD MODEL

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid', dtype='float32')(x)
model = Model(inputs=base_model.input, outputs=output)

# COSINE LEARNING RATE SCHEDULER

lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-4,
    first_decay_steps=1000,
    t_mul=2.0, m_mul=0.9, alpha=1e-6
)

# COMPILE MODEL

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=BinaryFocalCrossentropy(gamma=2, label_smoothing=0.05),
    metrics=['accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
model.summary()

# CALLBACKS

callbacks = [
    EarlyStopping(monitor='val_auc',mode='max',patience=6,restore_best_weights=True,verbose=1),
    ModelCheckpoint(MODEL_PATH,monitor='val_auc',mode='max',save_best_only=True,verbose=1)
]

# PHASE 1 TRAINING

print("\n===== PHASE 1: TRAINING TOP LAYERS =====")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# PHASE 2: FINE-TUNING

print("\n===== PHASE 2: FINE-TUNING =====")
base_model.trainable = True

# Freeze earlier layers
for layer in base_model.layers[:-110]:
    layer.trainable = False

# LOWER LEARNING RATE FOR FINE-TUNING

fine_tune_lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-5,
    first_decay_steps=1000,
    t_mul=2.0, m_mul=0.9, alpha=1e-7
)

# RECOMPILE

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
    loss=BinaryFocalCrossentropy(gamma=2,label_smoothing=0.05),
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=TOTAL_EPOCHS,
    initial_epoch=history.epoch[-1] + 1,
    class_weight=class_weights,
    callbacks=callbacks
)

# LOAD BEST MODEL

model.load_weights(MODEL_PATH)

# EVALUATION

print("\n===== EVALUATING MODEL =====")
results = model.evaluate(test_generator, verbose=1)
metrics = dict(zip(model.metrics_names, results))
print("\n===== TEST METRICS =====")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# TEST-TIME AUGMENTATION (TTA)

print("\n===== TEST TIME AUGMENTATION =====")
tta_steps = 5
predictions = []
for i in range(tta_steps):
    preds = model.predict(test_generator,verbose=1)
    predictions.append(preds)
y_pred_probs = np.mean(predictions,axis=0)

# CONVERT TO LABELS

y_pred = (y_pred_probs > 0.5).astype(int)
y_true = test_generator.classes

# CLASSIFICATION REPORT

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_true,y_pred,target_names=['NORMAL','PNEUMONIA']))

# F1 SCORE

f1 = f1_score(y_true, y_pred)
print(f"\nF1 Score: {f1:.4f}")

print("\n===== CONFUSION MATRIX =====")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# ROC AUC

auc_score = roc_auc_score(y_true, y_pred_probs)
print(f"\nROC-AUC Score: {auc_score:.4f}")

# SAVE FINAL MODEL

model.save(MODEL_PATH)
print("\n===== MODEL SAVED =====")
print(f"Saved at: {MODEL_PATH}")
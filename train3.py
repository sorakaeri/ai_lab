import os
import tensorflow as tf
from tensorflow.keras import layers, callbacks # [7단계] callbacks 임포트 추가
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Config 및 데이터 로드
# ==========================================
BASE_DIR = "./dataset"  # 데이터셋 폴더 경로

IMG_HEIGHT = 96
IMG_WIDTH = 96
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

SEED = 123
BATCH_SIZE = 32
EPOCHS = 20 # [7단계] EarlyStopping이 있으므로 에포크를 더 크게(예: 50~100) 주셔도 무방합니다.
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

class_names = sorted([
    name for name in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, name))
])
num_classes = len(class_names)
print("Class Names:", class_names)
print("Num Classes:", num_classes)

image_paths = []
labels = []

for label, class_name in enumerate(class_names):
    class_dir = os.path.join(BASE_DIR, class_name)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(file_path)
            labels.append(label)

image_paths = np.array(image_paths)
labels = np.array(labels)

# Train 분리
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels, test_size=(VAL_RATIO + TEST_RATIO), 
    random_state=SEED, stratify=labels
)

# Validation / Test 분리
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=SEED, stratify=temp_labels
)

print("Train:", len(train_paths))
print("Validation:", len(val_paths))
print("Test:", len(test_paths))

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(load_image)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_paths, train_labels, shuffle=True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

# ==========================================
# 2. 모델 구성 (1~5단계 개선사항 모두 포함)
# ==========================================
model = tf.keras.Sequential(name='CNN_Final_AdamW_Callbacks')

model.add(layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
 
# [1단계] 정규화 및 증강
model.add(layers.RandomFlip("horizontal"))
model.add(layers.Rescaling(1./255, name='Normalization'))

# Block 1
model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2)) 

# Block 2
model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# Block 3
model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4)) 

# 완전 연결층
model.add(layers.Flatten()) 
model.add(layers.Dense(256, kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5)) 

# 출력층
model.add(layers.Dense(units=num_classes, activation='softmax', name='Output'))

# ------------------------------------------
# [6단계 추가] AdamW 최적화 알고리즘
# ------------------------------------------
try:
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
except AttributeError:
    # 텐서플로우 구버전 호환용 (대부분 최신 버전에서는 윗줄이 실행됨)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------
# [7단계 추가] 콜백(Callbacks) 설정
# ------------------------------------------
# 1. 학습률 스케줄러: 검증 손실이 개선되지 않으면 학습률을 줄여 미세 조정
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.3,  
    patience=4,  
    min_lr=1e-6, 
    verbose=1
)

# 2. 조기 종료: 검증 손실이 10번 개선되지 않으면 종료 후 최고 가중치 복원
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True, 
    verbose=1
)

# 모델 학습 (콜백 적용)
history = model.fit(
    train_ds,               
    validation_data=val_ds, 
    epochs=EPOCHS,          
    verbose=2,
    callbacks=[lr_scheduler, early_stopping] # 콜백 리스트 추가
)

# ==========================================
# 3. 성능 평가 및 시각화
# ==========================================
val_loss, val_acc = model.evaluate(val_ds, verbose=0)
print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

y_true = []
y_pred = []

for images, labels_batch in test_ds:
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)
    y_true.extend(labels_batch.numpy())
    y_pred.extend(preds)

print("\nClassification Report")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

epochs_range = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy")
plt.legend()
plt.show()
import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Config

BASE_DIR = "./dataset"  # 데이터셋 폴더 경로

IMG_HEIGHT = 96 # 입력 이미지 높이
IMG_WIDTH = 96  # 입력 이미지 너비
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

SEED = 123  # 랜덤 시드값

BATCH_SIZE = 32 # 한 번에 학습할 이미지 개수
EPOCHS = 20 # 전체 학습 반복 횟수
TRAIN_RATIO = 0.7   # 학습 데이터 비율
VAL_RATIO = 0.15    # 검증 데이터 비율
TEST_RATIO = 0.15   # 테스트 데이터 비율


# 폴더 이름을 클래스 이름으로 사용
class_names = sorted([
    name for name in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, name))
])

# 클래스 개수 저장
num_classes = len(class_names)
# 클래스 이름 출력
print("Class Names:", class_names)
# 클래스 개수 출력
print("Num Classes:", num_classes)


# 이미지 파일 경로를 저장할 리스트
image_paths = []
# 이미지 라벨을 저장할 리스트
labels = []


for label, class_name in enumerate(class_names):
    class_dir = os.path.join(BASE_DIR, class_name)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(file_path)
            labels.append(label)


# 이미지 경로 리스트를 numpy 배열로 변환
image_paths = np.array(image_paths)
# 라벨 리스트를 numpy 배열로 변환
labels = np.array(labels)





# Train 
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,                         
    labels,                              
    test_size=(VAL_RATIO + TEST_RATIO),  
    random_state=SEED,
    stratify=labels
)

# Validation + Test
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    random_state=SEED,
    stratify=temp_labels
)


# 학습 데이터 개수 출력
print("Train:", len(train_paths))
# 검증 데이터 개수 출력
print("Validation:", len(val_paths))
# 테스트 데이터 개수 출력
print("Test:", len(test_paths))



# 이미지 경로와 라벨을 받아 실제 이미지 tensor로 변환
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image, label


# 이미지 경로와 라벨 배열을 TensorFlow Dataset으로 변환
def make_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED
        )
    ds = ds.map(load_image)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# 학습 데이터셋 생성
train_ds = make_dataset(train_paths, train_labels, shuffle=True)
# 검증 데이터셋 생성
val_ds = make_dataset(val_paths, val_labels)
# 테스트 데이터셋 생성
test_ds = make_dataset(test_paths, test_labels)

# Model
model = tf.keras.Sequential(name='SingleLayerPerceptron')

# 입력층
model.add(layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))

model.add(layers.Flatten())

# 출력층
model.add(layers.Dense(units=num_classes,activation='softmax',name='Output'))


# 모델 학습 방식
model.compile(
    # optimizer = "adam", 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 출력
model.summary()


# 모델 학습
history = model.fit(
    train_ds,               # 학습 데이터
    validation_data=val_ds, # 검증 데이터
    epochs=EPOCHS,          # 학습 반복 횟수
    verbose=2               # epoch별 결과만 출력
)


# 검증 데이터로 성능 평가
val_loss, val_acc = model.evaluate(
    val_ds,
    verbose=0
)

# 검증 정확도 출력
print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")
# 검증 손실 출력
print(f"Validation Loss: {val_loss:.4f}")


# 테스트 데이터로 최종 성능 평가
test_loss, test_acc = model.evaluate(
    test_ds,
    verbose=0
)

# 테스트 정확도 출력
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
# 테스트 손실 출력
print(f"Test Loss: {test_loss:.4f}")



# 실제 라벨 저장 리스트
y_true = []
# 예측 라벨 저장 리스트
y_pred = []

# 테스트 데이터셋 반복
for images, labels_batch in test_ds:
    # 모델 예측 수행
    preds = model.predict(images, verbose=0)
    # 가장 높은 확률을 가진 클래스 선택
    preds = np.argmax(preds, axis=1)
    # 실제 라벨 저장
    y_true.extend(labels_batch.numpy())
    # 예측 라벨 저장
    y_pred.extend(preds)


# 결과 출력
print("\nClassification Report")

print(classification_report(y_true,y_pred,target_names=class_names))

cm = confusion_matrix(y_true, y_pred)


# 히트맵
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 학습 과정
epochs_range = range(1, len(history.history['accuracy']) + 1)

# 학습 및 검증 정확도 그래프
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['loss'], label='Training Loss')
plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')

# 검증 정확도
plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy")
plt.legend()
plt.show()
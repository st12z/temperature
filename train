# Load data
import urllib.request
import zipfile
import os

# Đường dẫn thư mục đích
save_dir = r"C:\Users\admin\Desktop\IntSys"
os.makedirs(save_dir, exist_ok=True)

# URL file zip
url = "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = os.path.join(save_dir, "jena_climate_2009_2016.csv.zip")

# Tải file
print("Đang tải file...")
urllib.request.urlretrieve(url, zip_path)
print("Tải xong:", zip_path)

# Giải nén
print("Đang giải nén...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(save_dir)
print("Giải nén xong vào:", save_dir)
#------------------------------------------#

# Xử lý dữ liệu
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
fname = os.path.join("C:/Users/admin/Desktop/IntSys/jena_climate_2009_2016.csv")
from tensorflow.keras import layers
with open(fname) as f:
    data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]

print(header)
print(len(lines))


# Tạo mảng rỗng
temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))  # bỏ cột Date Time

for i, line in enumerate(lines):
    if not line.strip():   # bỏ qua dòng trống (nếu có)
        continue
    values = [float(x) for x in line.split(",")[1:]]  # bỏ cột đầu tiên (Date Time)
    temperature[i] = values[1]   # cột thứ 2 là T (degC)
    raw_data[i, :] = values
print("temperature: ",temperature)
print("raw_data: ",raw_data)

plt.plot(range(len(temperature)), temperature)


# 10 ngày đầu tiên (144 điểm mỗi ngày)
ten_days = 10 * 144

plt.figure(figsize=(15, 5))
plt.plot(range(ten_days), temperature[:ten_days])
plt.title("Nhiệt độ trong 10 ngày đầu tiên")
plt.xlabel("Timestep (10 phút mỗi điểm)")
plt.ylabel("Nhiệt độ (°C)")
plt.show()

# Chia tập dữ liệu train,test,valid
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)
# Chuẩn hóa dữ liệu
mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std
# Dùng hàm timeseries_dataset_from_array
sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256
# train_dataset
train_dataset = keras.utils.timeseries_dataset_from_array(
 raw_data[:-delay],
 targets=temperature[delay:],
 sampling_rate=sampling_rate,
 sequence_length=sequence_length,
 shuffle=True,
 batch_size=batch_size,
 start_index=0,
 end_index=num_train_samples)

# val_dataset
val_dataset = keras.utils.timeseries_dataset_from_array(
 raw_data[:-delay],
 targets=temperature[delay:],
 sampling_rate=sampling_rate,
 sequence_length=sequence_length,
 shuffle=True,
 batch_size=batch_size,
 start_index=num_train_samples,
 end_index=num_train_samples + num_val_samples)

# test_dataset
test_dataset = keras.utils.timeseries_dataset_from_array(
 raw_data[:-delay],
 targets=temperature[delay:],
 sampling_rate=sampling_rate,
 sequence_length=sequence_length,
 shuffle=True,
 batch_size=batch_size,
 start_index=num_train_samples + num_val_samples)
for samples, targets in train_dataset:
    print("samples shape:", samples.shape)
    print("targets shape:", targets.shape)
    break
# Common-sense Baseline
def evaluate_naive_method(dataset):
    total_abs_err = 0.
    samples_seen = 0
    for samples, targets in dataset:
        preds = samples[:, -1, 1] * std[1] + mean[1]  # lấy giá trị nhiệt độ hiện tại
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen
print(f"Validation MAE: {evaluate_naive_method(val_dataset):.2f}")
print(f"Test MAE: {evaluate_naive_method(test_dataset):.2f}")

# --------------------------------------------------------#

# Training model using dense
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Flatten()(inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
callbacks = [
 keras.callbacks.ModelCheckpoint("jena_dense.keras",
 save_best_only=True)
]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
 epochs=10,
 validation_data=val_dataset,
 callbacks=callbacks)
model = keras.models.load_model("jena_dense.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")\
#Plot
loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()


# Training using Conv1D
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Conv1D(8, 24, activation="relu")(inputs)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(8, 12, activation="relu")(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(8, 6, activation="relu")(x)
x = layers.GlobalAveragePooling1D()(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
callbacks = [
 keras.callbacks.ModelCheckpoint("jena_conv.keras",
 save_best_only=True)
]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
 epochs=10,
 validation_data=val_dataset,
 callbacks=callbacks)
model = keras.models.load_model("jena_conv.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")
#Plot
loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()

#----------------------------------------------#

# Training using LSTM
#LSTM
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(16)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
callbacks = [
 keras.callbacks.ModelCheckpoint("jena_lstm.keras",
 save_best_only=True)
]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
 epochs=10,
 validation_data=val_dataset,
 callbacks=callbacks)
model = keras.models.load_model("jena_lstm.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")
#Plot
loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()

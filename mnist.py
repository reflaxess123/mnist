import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# --- Загрузка данных ---
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = y.astype(int)
X = X / 255.0  # нормализация [0, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- One-hot encoding ---
def one_hot(y, num_classes=10):
    oh = np.zeros((y.shape[0], num_classes))
    oh[np.arange(y.shape[0]), y] = 1
    return oh

Y_train = one_hot(y_train)
Y_test = one_hot(y_test)

# --- Softmax ---
def softmax(z):
    # вычитаем max для численной стабильности
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# --- Параметры ---
n_features = 784
n_classes = 10
lr = 0.1
epochs = 200
batch_size = 256

W = np.zeros((n_features, n_classes))  # (784, 10)
b = np.zeros((1, n_classes))           # (1, 10)

# --- Обучение (mini-batch SGD) ---
n = X_train.shape[0]

for epoch in range(epochs):
    # перемешиваем данные каждую эпоху
    indices = np.random.permutation(n)
    X_shuf = X_train[indices]
    Y_shuf = Y_train[indices]

    for i in range(0, n, batch_size):
        X_batch = X_shuf[i:i+batch_size]
        Y_batch = Y_shuf[i:i+batch_size]
        m = X_batch.shape[0]

        # forward pass
        logits = X_batch @ W + b      # (m, 10)
        probs = softmax(logits)        # (m, 10)

        # градиенты (cross-entropy loss)
        error = probs - Y_batch        # (m, 10)
        dW = (X_batch.T @ error) / m   # (784, 10)
        db = np.mean(error, axis=0, keepdims=True)  # (1, 10)

        # обновление весов
        W -= lr * dW
        b -= lr * db

    # лог каждые 20 эпох
    if (epoch + 1) % 20 == 0:
        logits = X_train @ W + b
        probs = softmax(logits)
        loss = -np.mean(np.sum(Y_train * np.log(probs + 1e-8), axis=1))
        pred = np.argmax(probs, axis=1)
        acc = np.mean(pred == y_train)
        print(f"Epoch {epoch+1:3d} | Loss: {loss:.4f} | Train Acc: {acc:.4f}")

# --- Оценка на тесте ---
logits = X_test @ W + b
pred = np.argmax(softmax(logits), axis=1)
print(f"\nTest Accuracy: {np.mean(pred == y_test):.4f}")
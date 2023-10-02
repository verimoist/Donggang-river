import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 데이터 불러오기
data_path = "/home/seunghwan/바탕화면/edison2.xlsx"
data = pd.read_excel(data_path)
print(data.head())

# 날짜 및 다른 열 제거
columns_to_remove = ['date', 'averagewl', 'flow', '...14']  
data = data.drop(columns=columns_to_remove)
# 확인
print(data.head())

# 데이터 전처리 (정규화 및 시퀀스 형태로 변환)
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data.values)

# 훈련/검증 데이터 나누기
train, test = train_test_split(dataset, test_size=0.2, shuffle=False)

look_back = 7
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# LSTM 모델 구축
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 모델 학습
history = model.fit(X_train, y_train, epochs=100, batch_size=1, validation_data=(X_test, y_test))

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 원본 데이터 불러오기
df_original = pd.read_excel(data_path)

# averagewl 컬럼을 사용하여 MinMaxScaler 훈련시키기
scaler_y = MinMaxScaler()
scaler_y.fit(df_original['averagewl'].values.reshape(-1, 1))

# y_test와 test_predict를 역 스케일링
y_test_inv = scaler_y.inverse_transform(y_test)
test_predict_inv = scaler_y.inverse_transform(test_predict)

# 성능평가 MSE, RMSE, MAE, R2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 평가 지표 계산
mse = mean_squared_error(y_test_inv, test_predict_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inv, test_predict_inv)
r2 = r2_score(y_test_inv, test_predict_inv)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)

# 시각화
metrics = ['MSE', 'RMSE', 'MAE', 'R2 Score']
values = [mse, rmse, mae, r2]

plt.figure(figsize=(10, 5))
plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
plt.ylabel('Score')
plt.title('Regression Metrics')
plt.show()

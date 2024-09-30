import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    data_path = BASE_DIR + '\lab1.2\CO2 Emissions_Canada.csv'
    data_frame = pd.read_csv(data_path)

    num_only_data = data_frame[data_frame.columns[[3, 4, 7, 8, 9, 10, 11]]]

    print('Параметры выборки:\n1. Engine Size(L)\n2. Cylinders\n3. Fuel Consumption City (L/100 km)\n'
          '4.Fuel Consumption Hwy (L/100 km)\n5. Fuel Consumption Comb (L/100 km)\n6. Fuel Consumption Comb (mpg)'
          '\n7. CO2 Emissions(g/km)')
    x_number = int(input('Введите первый признак для отобржения (ось x): '))
    y_number = int(input('Введите второй признак для отобржения (ось y): '))

    plt.scatter(num_only_data[num_only_data.columns[x_number - 1]], num_only_data[num_only_data.columns[y_number - 1]])

    x_name = num_only_data.columns[x_number - 1]
    y_name = num_only_data.columns[y_number - 1]

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()
    plt.close()

    x_data = num_only_data.values[:, x_number - 1]
    y_data = num_only_data.values[:, y_number - 1]
    # print(x_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.2, random_state=42)

    lr = LinearRegression().fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
    regression_coef = lr.score(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
    print('Коэффициент регресси тренировочных данных: ', round(regression_coef, 4))

    y_pred = lr.predict(x_test.reshape(-1, 1))

    angular_coefficient = lr.coef_[0][0]
    interception_point = lr.intercept_[0]

    print('Угловой коэффициент: ', round(angular_coefficient, 4))
    print('Точка пересечения: ', round(interception_point, 4))

    mse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
    msa = round(mean_absolute_error(y_test, y_pred), 4)
    print('Среднеквадратичная ошибка: ', mse)
    print('Средняя абсолютная ошибка: ', msa)

    plt.scatter(x_train, y_train)
    plt.plot(x_train, angular_coefficient * x_train + interception_point, color='r')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()
    plt.close()

    x_value_for_pred = int(input(f'Введите значение первого признака ({x_name}) для выполнения предсказания: '))
    y_pred_by_x = lr.predict([[float(x_value_for_pred)]])
    print(f'Предсказанное значение {y_name}: ', round(y_pred_by_x[0][0], 4))


if __name__ == '__main__':
    main()

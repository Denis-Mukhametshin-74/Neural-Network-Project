"""
@file data_loader.py
@brief Загрузка и подготовка данных MNIST
@author Developer2
@version 1.0
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


class DataLoader:
    """
    @brief Класс для загрузки и обработки данных MNIST

    @details
    Загружает dataset MNIST, выполняет нормализацию и разделение на выборки
    Обрабатывает ошибки загрузки и предоставляет чистый интерфейс для данных
    """

    def __init__(self):
        """
        @brief Конструктор DataLoader
        """
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_mnist_data(self, test_size=0.2, random_state=42):
        """
        @brief Загрузка данных MNIST

        @param test_size: Доля тестовой выборки (по умолчанию 0.2)
        @param random_state: Seed для воспроизводимости (по умолчанию 42)
        @return: Кортеж с обучающими и тестовыми данными (X_train, X_test, y_train, y_test)

        @details
        Загружает MNIST dataset из sklearn и выполняет предобработку:
        - Нормализация пикселей к диапазону [0, 1]
        - One-hot encoding меток
        - Разделение на обучающую и тестовую выборки
        """
        try:
            print("Загрузка MNIST dataset...")

            # Загрузка данных
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            X, y = mnist.data, mnist.target

            # Нормализация пикселей к диапазону [0, 1]
            X = self.normalize_data(X)

            # Преобразование меток в one-hot encoding
            y_onehot = self.one_hot_encode(y)

            # Разделение на обучающую и тестовую выборки
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y_onehot, test_size=test_size, random_state=random_state
            )

            print(f"✅ Данные успешно загружены:")
            print(f"   Обучающая выборка: {self.X_train.shape[0]} примеров")
            print(f"   Тестовая выборка: {self.X_test.shape[0]} примеров")
            print(f"   Размерность признаков: {self.X_train.shape[1]}")
            print(f"   Количество классов: {self.y_train.shape[1]}")

            return self.X_train, self.X_test, self.y_train, self.y_test

        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            raise

    def normalize_data(self, X):
        """
        @brief Нормализация данных

        @param X: Входные данные
        @return: Нормализованные данные

        @details
        Преобразует значения пикселей из диапазона [0, 255] в [0.0, 1.0]
        """
        return X.astype(np.float32) / 255.0

    def one_hot_encode(self, y):
        """
        @brief One-hot encoding меток

        @param y: Исходные метки (0-9)
        @return: One-hot encoded метки

        @details
        Преобразует скалярные метки в one-hot encoding для 10 классов
        """
        lb = LabelBinarizer()
        return lb.fit_transform(y)
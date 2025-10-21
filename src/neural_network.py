"""
@file neural_network.py
@brief Основной класс нейросети для классификации цифр
@author Developer1
@version 1.0
"""

import numpy as np
import json


class NeuralNetwork:
    """
    @brief Класс нейросети с одним скрытым слоем

    @details
    Архитектура: input_size -> hidden_size -> output_size
    Использует сигмоиду для скрытого слоя и softmax для выходного
    Поддерживает forward propagation и базовые операции
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        @brief Конструктор нейросети

        @param input_size: Размер входного слоя
        @param hidden_size: Размер скрытого слоя
        @param output_size: Размер выходного слоя
        @param learning_rate: Скорость обучения (по умолчанию 0.01)

        @details
        Инициализирует веса случайными значениями из нормального распределения
        и устанавливает начальные смещения в ноль
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Инициализация весов случайными значениями
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * 0.01

        # Инициализация смещений нулями
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))

        # История для отладки
        self.loss_history = []

    def sigmoid(self, x):
        """
        @brief Сигмоидная функция активации

        @param x: Входные данные
        @return: Значение сигмоиды

        @details
        Формула: 1 / (1 + exp(-x))
        Используется для скрытого слоя
        Защита от переполнения через np.clip
        """
        # Защита от численного переполнения
        x_clipped = np.clip(x, -250, 250)
        return 1 / (1 + np.exp(-x_clipped))

    def softmax(self, x):
        """
        @brief Функция активации Softmax

        @param x: Входные данные
        @return: Вектор вероятностей

        @details
        Преобразует выходы в вероятности для многоклассовой классификации
        Стабильная реализация через вычитание максимума
        """
        # Стабильная реализация softmax
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        @brief Прямое распространение (forward propagation)

        @param X: Входные данные (batch_size x input_size)
        @return: Выходные вероятности

        @details
        Выполняет:
        1. hidden_input = X * weights1 + bias1
        2. hidden_output = sigmoid(hidden_input)
        3. output_input = hidden_output * weights2 + bias2
        4. output = softmax(output_input)
        """
        # Скрытый слой
        self.hidden_input = np.dot(X, self.weights1) + self.bias1
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Выходной слой
        self.output_input = np.dot(self.hidden_output, self.weights2) + self.bias2
        self.output = self.softmax(self.output_input)

        return self.output

    def compute_loss(self, y_true, y_pred):
        """
        @brief Вычисление функции потерь (кросс-энтропия)

        @param y_true: Истинные метки (one-hot encoded)
        @param y_pred: Предсказанные вероятности
        @return: Значение функции потерь

        @details
        Используется categorical cross-entropy для многоклассовой классификации
        Добавлен epsilon для избежания log(0)
        """
        m = y_true.shape[0]
        epsilon = 1e-8  # Защита от log(0)
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
        return loss

    def predict(self, X):
        """
        @brief Предсказание меток для новых данных

        @param X: Входные данные для предсказания
        @return: Предсказанные метки (индексы классов)

        @details
        Возвращает класс с максимальной вероятностью
        """
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def compute_accuracy(self, X, y):
        """
        @brief Вычисление точности предсказаний

        @param X: Входные данные
        @param y: Истинные метки (one-hot encoded)
        @return: Точность классификации

        @details
        Сравнивает предсказанные метки с истинными
        Возвращает долю правильных предсказаний
        """
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy
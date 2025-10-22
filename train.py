"""
@file train.py
@brief Основной скрипт для обучения и тестирования нейросети
@author Developer1 & Developer2
@version 1.0
"""

import numpy as np
from src.neural_network import NeuralNetwork
from src.data_loader import DataLoader


def main():
    """
    @brief Основная функция обучения

    @details
    Демонстрирует работу нейросети на MNIST данных
    Показывает базовую функциональность forward propagation
    """
    print("=== Neural Network Project ===")
    print("Демонстрация базовой функциональности US-1")

    try:
        # Загрузка данных
        loader = DataLoader()
        X_train, X_test, y_train, y_test = loader.load_mnist_data()

        # Создание нейросети
        input_size = X_train.shape[1]  # 784 features
        hidden_size = 128
        output_size = y_train.shape[1]  # 10 classes

        print(f"\nСоздание нейросети: {input_size}-{hidden_size}-{output_size}")
        nn = NeuralNetwork(input_size, hidden_size, output_size)

        # Тестирование forward propagation
        print("\nТестирование forward propagation...")
        sample_data = X_train[:5]  # Первые 5 примеров
        predictions = nn.forward(sample_data)

        print(f"Входные данные: {sample_data.shape}")
        print(f"Выходные вероятности: {predictions.shape}")
        print(f"Сумма вероятностей для первого примера: {predictions[0].sum():.6f}")

        # Проверка точности на случайных предсказаниях
        accuracy = nn.compute_accuracy(X_test, y_test)
        print(f"\nТочность на тестовой выборке: {accuracy:.4f}")

        print("\n✅ US-1 успешно завершена!")
        print("Нейросеть инициализирована, forward propagation работает корректно")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
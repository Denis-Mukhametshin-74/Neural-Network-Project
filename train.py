"""
@file train.py
@brief Основной скрипт для обучения и тестирования нейросети
@author Developer1 & Developer2
@version 2.0
"""

import numpy as np
import sys
import os
from src.neural_network import NeuralNetwork
from src.data_loader import DataLoader
from src.visualizer import NeuralNetworkVisualizer


def main():
    """
    @brief Основная функция обучения с визуализацией
    """
    print("=== Neural Network Project ===")
    print("🎨 Демонстрация с визуализацией US-1")

    # Создаем папку для графиков
    os.makedirs('docs/images', exist_ok=True)

    try:
        # Инициализируем визуализатор
        visualizer = NeuralNetworkVisualizer()

        # Загрузка данных
        print("\n📥 Загрузка данных MNIST...")
        loader = DataLoader()
        X_train, X_test, y_train, y_test = loader.load_mnist_data()

        # Визуализация примеров данных
        print("📊 Визуализация примеров данных...")
        visualizer.plot_mnist_samples(
            X_train, y_train,
            num_samples=10,
            save_path='docs/images/mnist_samples.png'
        )

        # Визуализация архитектуры сети
        print("🏗️ Визуализация архитектуры нейросети...")
        visualizer.plot_network_architecture(
            input_size=784, hidden_size=128, output_size=10,
            save_path='docs/images/network_architecture.png'
        )

        # Создание нейросети
        input_size = X_train.shape[1]  # 784 features
        hidden_size = 128
        output_size = y_train.shape[1]  # 10 classes

        print(f"\n🧠 Создание нейросети: {input_size}-{hidden_size}-{output_size}")
        nn = NeuralNetwork(input_size, hidden_size, output_size)

        # Тестирование forward propagation
        print("\n🔍 Тестирование forward propagation...")
        sample_data = X_train[:5]  # Первые 5 примеров
        predictions = nn.forward(sample_data)

        print(f"📐 Входные данные: {sample_data.shape}")
        print(f"📊 Выходные вероятности: {predictions.shape}")
        print(f"🧮 Сумма вероятностей для первого примера: {predictions[0].sum():.6f}")

        # Визуализация примеров предсказаний
        print("\n🎯 Визуализация примеров предсказаний...")
        visualizer.plot_prediction_example(
            nn, X_test, y_test,
            num_examples=3,
            save_path='docs/images/prediction_examples.png'
        )

        # Проверка точности на случайных предсказаниях
        accuracy = nn.compute_accuracy(X_test, y_test)
        print(f"\n📈 Точность на тестовой выборке: {accuracy:.4f}")

        # Демонстрация работы
        print(f"\n🔍 Пример предсказания:")
        print(f"📊 Вероятности по классам: {[f'{p:.3f}' for p in predictions[0]]}")
        print(f"🎯 Предсказанный класс: {np.argmax(predictions[0])}")

        print("\n" + "=" * 50)
        print("✅ US-1 успешно завершена!")
        print("🎨 Добавлена визуализация данных и архитектуры")
        print("📁 Графики сохранены в docs/images/")
        print("🧠 Нейросеть инициализирована, forward propagation работает корректно")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("\n🔧 Возможные решения:")
        print("1. Установите pandas: pip install pandas")
        print("2. Проверьте подключение к интернету")
        print("3. Убедитесь, что установлен matplotlib: pip install matplotlib")
        sys.exit(1)


if __name__ == "__main__":
    main()
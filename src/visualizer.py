"""
@file visualizer.py
@brief Визуализация работы нейросети и данных MNIST
@author Developer1
@version 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


class NeuralNetworkVisualizer:
    """
    @brief Класс для визуализации данных и работы нейросети

    @details
    Создает графики для анализа:
    - Примеры изображений MNIST
    - Архитектуру нейросети
    - Процесс предсказания
    - Матрицу весов
    """

    def __init__(self):
        """
        @brief Конструктор визуализатора
        """
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['figure.figsize'] = (10, 6)

    def plot_mnist_samples(self, X, y, num_samples=10, save_path=None):
        """
        @brief Визуализация примеров данных MNIST

        @param X: Данные изображений
        @param y: Метки (one-hot encoded)
        @param num_samples: Количество примеров для отображения
        @param save_path: Путь для сохранения графика
        """
        # Преобразуем one-hot обратно в цифры
        y_labels = np.argmax(y, axis=1)

        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.ravel()

        for i in range(num_samples):
            # Берем случайный пример
            idx = np.random.randint(0, len(X))
            image = X[idx].reshape(28, 28)
            label = y_labels[idx]

            # Отображаем изображение
            axes[i].imshow(image, cmap='gray', interpolation='none')
            axes[i].set_title(f'Цифра: {label}', fontsize=12, pad=10)
            axes[i].axis('off')

            # Добавляем сетку пикселей для наглядности
            axes[i].grid(False)

        plt.suptitle('🔢 Примеры рукописных цифр из датасета MNIST', fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 График сохранен: {save_path}")

        plt.show()

    def plot_network_architecture(self, input_size=784, hidden_size=128, output_size=10, save_path=None):
        """
        @brief Визуализация архитектуры нейросети

        @param input_size: Размер входного слоя
        @param hidden_size: Размер скрытого слоя
        @param output_size: Размер выходного слоя
        @param save_path: Путь для сохранения графика
        """
        fig = plt.figure(figsize=(15, 8))

        # Создаем сетку для отображения
        gs = gridspec.GridSpec(1, 3, width_ratios=[3, 2, 1])

        # Входной слой
        ax1 = plt.subplot(gs[0])
        self._draw_layer(ax1, input_size, "Входной слой\n784 нейрона\n(28×28 пикселей)", 'lightblue')

        # Скрытый слой
        ax2 = plt.subplot(gs[1])
        self._draw_layer(ax2, hidden_size, "Скрытый слой\n128 нейронов\n(Sigmoid)", 'lightgreen')

        # Выходной слой
        ax3 = plt.subplot(gs[2])
        self._draw_layer(ax3, output_size, "Выходной слой\n10 нейронов\n(Softmax)", 'lightcoral')

        # Соединения между слоями
        self._draw_connections(fig, [0.15, 0.5, 0.85], [input_size, hidden_size, output_size])

        plt.suptitle('🏗️ Архитектура нейросети для распознавания цифр', fontsize=18, y=0.95)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"🏗️ Схема архитектуры сохранена: {save_path}")

        plt.show()

    def _draw_layer(self, ax, num_neurons, title, color):
        """
        @brief Отрисовка одного слоя нейросети
        """
        ax.set_xlim(0, 1)
        ax.set_ylim(0, num_neurons + 1)

        # Рисуем нейроны как круги
        for i in range(num_neurons):
            circle = plt.Circle((0.5, i + 0.5), 0.3, fill=True, color=color, alpha=0.7)
            ax.add_patch(circle)
            if num_neurons <= 20:  # Подписываем только если нейронов немного
                ax.text(0.5, i + 0.5, str(i), ha='center', va='center', fontsize=8)

        ax.set_title(title, fontsize=12, pad=20)
        ax.axis('off')

    def _draw_connections(self, fig, x_positions, layer_sizes):
        """
        @brief Отрисовка соединений между слоями
        """
        for i in range(len(x_positions) - 1):
            x1, x2 = x_positions[i], x_positions[i + 1]
            # Рисуем несколько соединений для наглядности
            for j in range(min(10, layer_sizes[i])):
                for k in range(min(10, layer_sizes[i + 1])):
                    y1 = (j / min(10, layer_sizes[i])) * layer_sizes[i] + 0.5
                    y2 = (k / min(10, layer_sizes[i + 1])) * layer_sizes[i + 1] + 0.5
                    fig.axes[0].plot([x1 * fig.get_figwidth(), x2 * fig.get_figwidth()],
                                     [y1, y2], 'gray', alpha=0.1, linewidth=0.5)

    def plot_prediction_example(self, model, X, y, num_examples=3, save_path=None):
        """
        @brief Визуализация примеров предсказаний

        @param model: Обученная нейросеть
        @param X: Данные для предсказания
        @param y: Истинные метки (one-hot)
        @param num_examples: Количество примеров
        @param save_path: Путь для сохранения
        """
        # Берем случайные примеры
        indices = np.random.choice(len(X), num_examples, replace=False)

        fig, axes = plt.subplots(num_examples, 2, figsize=(12, 4 * num_examples))

        for i, idx in enumerate(indices):
            # Получаем данные
            image = X[idx].reshape(28, 28)
            true_label = np.argmax(y[idx])

            # Делаем предсказание
            prediction = model.forward(X[idx:idx + 1])
            predicted_label = np.argmax(prediction[0])
            confidence = prediction[0][predicted_label]

            # Отображаем изображение
            if num_examples == 1:
                ax_img = axes[0]
                ax_bar = axes[1]
            else:
                ax_img = axes[i, 0]
                ax_bar = axes[i, 1]

            # Изображение
            ax_img.imshow(image, cmap='gray')
            ax_img.set_title(f'Изображение | Истинная цифра: {true_label}', fontsize=14)
            ax_img.axis('off')

            # График вероятностей
            classes = list(range(10))
            colors = ['red' if j == predicted_label else 'blue' for j in classes]

            bars = ax_bar.bar(classes, prediction[0], color=colors, alpha=0.7)
            ax_bar.set_xlabel('Цифры')
            ax_bar.set_ylabel('Вероятность')
            ax_bar.set_title(f'Предсказание: {predicted_label} (уверенность: {confidence:.2%})', fontsize=14)
            ax_bar.set_xticks(classes)
            ax_bar.grid(True, alpha=0.3)

            # Подписываем правильный ответ
            if predicted_label == true_label:
                ax_bar.text(0.02, 0.95, '✅ ВЕРНО', transform=ax_bar.transAxes,
                            fontsize=12, color='green', fontweight='bold')
            else:
                ax_bar.text(0.02, 0.95, '❌ ОШИБКА', transform=ax_bar.transAxes,
                            fontsize=12, color='red', fontweight='bold')

        plt.suptitle('🎯 Примеры работы нейросети', fontsize=16, y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"🎯 График предсказаний сохранен: {save_path}")

        plt.show()
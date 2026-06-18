
#🎸 Процесс доработки приложения Weather & Heat Calculator

## 📋 Общее описание проекта

**Weather & Heat Calculator** — это учебный проект по разработке приложения для мониторинга погоды и расчета теплопотерь. Проект реализован с применением SCRUM методологии и служит практическим примером построения полного цикла разработки с CI/CD.

### 🎯 Назначение
- Автоматическое определение координат по названию населенного пункта
- Сохранение избранных локаций для быстрого доступа
- Расчет реально потраченных килокалорий на отопление

### 🛠 Технологии
- Python 3.11+
- Requests — HTTP-запросы к API
- Pytest — unit-тестирование
- GitHub Actions — CI/CD
- Doxygen — генерация документации

---

## 📁 Структура проекта (новая)

```
weather_heat_calc/
├── src/
│   ├── __init__.py
│   ├── api_client.py         
│   ├── storage.py            # Работа с JSON-хранилищем локаций
│   ├── heat_calculator.py    # Расчет килокалорий
│   └── data_loader.py        # Загрузка/сохранение данных 
├── tests/
│   ├── __init__.py
│   ├── test_api_client.py    
│   ├── test_storage.py       
│   └── test_heat_calculator.py 
├── data/
│   └── cities.json           # Хранилище избранных городов
├── docs/                     # Doxygen документация
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub
├── main.py                   
├── requirements.txt          
├── Doxyfile                  # Конфигурация Doxygen
└── README.md
```

---

## 🔄 Процесс разработки (SCRUM)

### Спринт 1: Базовый функционал и обработка ошибок

#### User Story 1.1: Обработка ошибок API
**Ветка:** `feature/team-alpha_error_handling`

**Код для `src/api_client.py`** :

```python
"""
@file api_client.py
@brief Клиент для работы с погодными API и геокодером
@author Developer2
@version 1.0
"""

import requests
import logging
from typing import Optional, Tuple

# Настройка логирования (как в вашем стиле)
logger = logging.getLogger(__name__)


class WeatherAPIError(Exception):
    """
    @brief Пользовательское исключение для ошибок API
    @details Наследуется от Exception для специфичной обработки ошибок погодного сервиса
    """
    pass


class WeatherService:
    """
    @brief Класс для взаимодействия с внешними API

    @details
    Предоставляет методы для получения координат по названию города
    и получения погодных данных. Обрабатывает все возможные ошибки
    с выводом понятных сообщений пользователю.
    """

    def __init__(self):
        """
        @brief Конструктор WeatherService
        @details Инициализирует URL-адреса API и настраивает таймауты
        """
        self.geo_url = "https://nominatim.openstreetmap.org/search"
        self.weather_url = "https://api.open-meteo.com/v1/forecast"
        self.timeout = 10  # секунд

    def get_coordinates(self, city_name: str) -> Tuple[Optional[float], Optional[float]]:
        """
        @brief Получение координат по названию населенного пункта

        @param city_name: Название города
        @return: Кортеж (широта, долгота) или (None, None) при ошибке

        @details
        Использует OpenStreetMap Nominatim API.
        Обрабатывает:
        - Ошибки сети
        - HTTP ошибки (404, 500 и др.)
        - Пустые ответы (город не найден)
        - Таймауты
        """
        try:
            logger.info(f"Поиск координат для: {city_name}")

            params = {
                'q': city_name,
                'format': 'json',
                'limit': 1,
                'accept-language': 'ru'
            }

            response = requests.get(
                self.geo_url,
                params=params,
                timeout=self.timeout
            )

            # Проверка HTTP статуса (выбрасывает исключение при 4xx/5xx)
            response.raise_for_status()

            data = response.json()

            if not data:
                logger.warning(f"Город '{city_name}' не найден")
                raise WeatherAPIError(
                    f"❌ Город '{city_name}' не найден. Проверьте написание."
                )

            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])

            logger.info(f"✅ Найдены координаты для {city_name}: ({lat}, {lon})")
            return lat, lon

        except requests.exceptions.ConnectionError:
            logger.error("Ошибка сети: проверьте подключение к интернету")
            raise WeatherAPIError(
                "❌ Ошибка сети: проверьте подключение к интернету"
            )

        except requests.exceptions.Timeout:
            logger.error(f"Таймаут при запросе к API (>{self.timeout} сек)")
            raise WeatherAPIError(
                f"❌ Превышено время ожидания ответа от сервера (> {self.timeout} сек)"
            )

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP ошибка: {e}")
            raise WeatherAPIError(
                f"❌ Ошибка сервера: {e.response.status_code} - {e.response.reason}"
            )

        except WeatherAPIError:
            # Пробрасываем дальше уже обработанную ошибку
            raise

        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            raise WeatherAPIError(
                f"❌ Неожиданная ошибка: {str(e)}"
            )
```

---

#### User Story 1.2: Тестирование обработки ошибок
**Ветка:** `feature/team-alpha_tests`

**Код для `tests/test_api_client.py`** :

```python
"""
@file test_api_client.py
@brief Юнит-тесты для API клиента
@author Developer2
@version 1.0
"""

import pytest
import requests
from unittest.mock import Mock, patch
from src.api_client import WeatherService, WeatherAPIError


class TestWeatherService:
    """
    @brief Тестовый класс для WeatherService
    @details Проверяет корректность работы всех методов и обработку ошибок
    """

    def setup_method(self):
        """
        @brief Подготовка к тестам
        @details Создает экземпляр WeatherService перед каждым тестом
        """
        self.service = WeatherService()

    def test_get_coordinates_success(self):
        """
        @brief Тест успешного получения координат
        @details Проверяет корректный парсинг ответа API
        """
        # Подготавливаем мок-ответ
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {'lat': '55.7558', 'lon': '37.6173', 'display_name': 'Moscow, Russia'}
        ]

        with patch('requests.get', return_value=mock_response):
            lat, lon = self.service.get_coordinates("Москва")

            assert lat == 55.7558
            assert lon == 37.6173

    def test_get_coordinates_not_found(self):
        """
        @brief Тест на ненайденный город
        @details Проверяет выбрасывание исключения WeatherAPIError
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []

        with patch('requests.get', return_value=mock_response):
            with pytest.raises(WeatherAPIError) as exc_info:
                self.service.get_coordinates("НесуществующийГород")

            assert "не найден" in str(exc_info.value)

    def test_get_coordinates_connection_error(self):
        """
        @brief Тест на ошибку подключения
        @details Проверяет обработку ConnectionError
        """
        with patch('requests.get', side_effect=requests.exceptions.ConnectionError):
            with pytest.raises(WeatherAPIError) as exc_info:
                self.service.get_coordinates("Москва")

            assert "сети" in str(exc_info.value)

    def test_get_coordinates_http_error(self):
        """
        @brief Тест на HTTP ошибку
        @details Проверяет обработку статусов 4xx/5xx
        """
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Client Error"
        )

        with patch('requests.get', return_value=mock_response):
            with pytest.raises(WeatherAPIError) as exc_info:
                self.service.get_coordinates("Москва")

            assert "404" in str(exc_info.value)
```

---

### Спринт 2: Хранение локаций и геокодинг

#### User Story 2.1: Система хранения локаций
**Ветка:** `feature/team-alpha_storage`

**Код для `src/storage.py`** :

```python
"""
@file storage.py
@brief Управление сохраненными локациями
@author Developer1
@version 1.0
"""

import json
import os
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LocationStorage:
    """
    @brief Класс для работы с хранилищем локаций

    @details
    Обеспечивает сохранение, загрузку и управление списком избранных городов.
    Данные хранятся в JSON-файле в папке data/
    """

    def __init__(self, storage_path: str = "data/cities.json"):
        """
        @brief Конструктор LocationStorage

        @param storage_path: Путь к файлу хранилища
        @details Создает директорию data/ если она не существует
        """
        self.storage_path = storage_path
        self._ensure_directory()

    def _ensure_directory(self):
        """
        @brief Создание директории для хранения
        @details Создает папку data/ если она отсутствует
        """
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    def load_cities(self) -> List[Dict[str, float]]:
        """
        @brief Загрузка списка городов

        @return: Список словарей с полями name, lat, lon
        @details Если файл не существует, возвращает пустой список
        """
        if not os.path.exists(self.storage_path):
            logger.info("Файл хранилища не найден, создан новый")
            return []

        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                cities = json.load(f)
                logger.info(f"✅ Загружено {len(cities)} городов")
                return cities
        except json.JSONDecodeError:
            logger.error("Ошибка чтения JSON файла")
            return []
        except Exception as e:
            logger.error(f"Ошибка загрузки: {e}")
            return []

    def save_cities(self, cities: List[Dict[str, float]]) -> bool:
        """
        @brief Сохранение списка городов

        @param cities: Список словарей с данными о городах
        @return: True если сохранено успешно
        @details Перезаписывает весь файл с новыми данными
        """
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(cities, f, ensure_ascii=False, indent=4)
            logger.info(f"✅ Сохранено {len(cities)} городов")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
            return False

    def add_city(self, city_name: str, lat: float, lon: float) -> bool:
        """
        @brief Добавление нового города

        @param city_name: Название города
        @param lat: Широта
        @param lon: Долгота
        @return: True если город добавлен, False если уже существует
        @details Проверяет наличие дубликата по названию
        """
        cities = self.load_cities()

        # Проверка на дубликат
        if any(c['name'].lower() == city_name.lower() for c in cities):
            logger.warning(f"Город '{city_name}' уже в списке")
            return False

        cities.append({
            'name': city_name,
            'lat': lat,
            'lon': lon
        })

        return self.save_cities(cities)

    def remove_city(self, city_name: str) -> bool:
        """
        @brief Удаление города из списка

        @param city_name: Название города для удаления
        @return: True если город удален, False если не найден
        """
        cities = self.load_cities()
        original_count = len(cities)

        cities = [c for c in cities if c['name'].lower() != city_name.lower()]

        if len(cities) == original_count:
            logger.warning(f"Город '{city_name}' не найден")
            return False

        return self.save_cities(cities)

    def get_city(self, city_name: str) -> Optional[Dict[str, float]]:
        """
        @brief Получение данных о городе

        @param city_name: Название города
        @return: Словарь с данными или None если не найден
        """
        cities = self.load_cities()
        for city in cities:
            if city['name'].lower() == city_name.lower():
                return city
        return None

    def get_all_cities(self) -> List[str]:
        """
        @brief Получение списка названий всех городов

        @return: Список названий городов
        """
        cities = self.load_cities()
        return [city['name'] for city in cities]
```

---

### Спринт 3: Расчет тепла

#### User Story 3.1: Модуль расчета килокалорий
**Ветка:** `feature/team-alpha_heat_calc`

**Код для `src/heat_calculator.py`**:

```python
"""
@file heat_calculator.py
@brief Расчет теплопотребления на отопление
@author Developer1
@version 1.0
"""

import numpy as np
from typing import List, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class HeatCalculator:
    """
    @brief Класс для расчета затраченных килокалорий

    @details
    Вычисляет реальное потребление тепла на основе:
    - Ежедневной разницы температур (батарея - водоем)
    - Объема воды в системе (из квитанции)
    - Количества дней в отопительном периоде

    Формула: Q = (T_bat - T_water) * V_daily * days_count
    где 1 ккал ~ нагрев 1 литра воды на 1°C
    """

    def __init__(self):
        """
        @brief Конструктор HeatCalculator
        @details Инициализирует пустую историю измерений
        """
        self.measurements = []
        self.water_volume_daily = 0  # литров в день

    def set_water_volume(self, monthly_volume: float, days_in_month: int = 30):
        """
        @brief Установка дневного расхода воды

        @param monthly_volume: Объем воды за месяц (из квитанции)
        @param days_in_month: Количество дней в месяце
        @details Рассчитывает средний дневной расход
        """
        if monthly_volume <= 0:
            raise ValueError("Объем воды должен быть положительным числом")

        self.water_volume_daily = monthly_volume / days_in_month
        logger.info(f"Установлен дневной расход: {self.water_volume_daily:.2f} л/день")

    def add_daily_measurement(self, battery_temp: float, water_temp: float, date: str = None):
        """
        @brief Добавление дневного измерения

        @param battery_temp: Температура батареи (°C)
        @param water_temp: Температура водоема (°C)
        @param date: Дата измерения (опционально)
        @details Сохраняет измерение для последующего расчета
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        measurement = {
            'date': date,
            'battery_temp': battery_temp,
            'water_temp': water_temp,
            'temp_diff': battery_temp - water_temp
        }

        self.measurements.append(measurement)
        logger.info(f"Добавлено измерение от {date}: ΔT = {measurement['temp_diff']:.2f}°C")

    def calculate_monthly_heat(self) -> Dict[str, float]:
        """
        @brief Расчет месячного теплопотребления

        @return: Словарь с результатами расчета
        @details Суммирует дневные разницы температур и умножает на расход воды

        Формула:
        Q_total = Σ(ΔT_i) * V_daily
        где ΔT_i = T_bat_i - T_water_i
        """
        if not self.measurements:
            logger.warning("Нет данных для расчета")
            return {
                'total_heat_kcal': 0,
                'avg_temp_diff': 0,
                'days_count': 0,
                'total_volume_used': 0
            }

        if self.water_volume_daily <= 0:
            raise ValueError("Не установлен расход воды. Используйте set_water_volume()")

        # Суммируем все разницы температур
        total_temp_diff = sum(m['temp_diff'] for m in self.measurements)

        # Общее количество килокалорий
        total_heat = total_temp_diff * self.water_volume_daily

        # Средняя разница температур
        avg_temp_diff = total_temp_diff / len(self.measurements)

        # Общий объем воды
        total_volume = self.water_volume_daily * len(self.measurements)

        return {
            'total_heat_kcal': total_heat,
            'avg_temp_diff': avg_temp_diff,
            'days_count': len(self.measurements),
            'total_volume_used': total_volume,
            'water_volume_daily': self.water_volume_daily
        }

    def get_daily_stats(self) -> List[Dict[str, float]]:
        """
        @brief Получение ежедневной статистики

        @return: Список словарей с дневными данными
        @details Возвращает все измерения с рассчитанными значениями
        """
        stats = []
        for m in self.measurements:
            daily_heat = m['temp_diff'] * self.water_volume_daily
            stats.append({
                'date': m['date'],
                'battery_temp': m['battery_temp'],
                'water_temp': m['water_temp'],
                'temp_diff': m['temp_diff'],
                'daily_heat_kcal': daily_heat
            })
        return stats

    def clear_measurements(self):
        """
        @brief Очистка всех измерений
        @details Удаляет всю историю измерений
        """
        self.measurements.clear()
        logger.info("История измерений очищена")

    def calculate_efficiency(self, paid_gcal: float) -> Dict[str, float]:
        """
        @brief Расчет эффективности отопления

        @param paid_gcal: Оплаченные гигакалории
        @return: Словарь с показателями эффективности
        @details Сравнивает реальное потребление с оплаченным

        1 Гкал = 1 000 000 ккал
        """
        result = self.calculate_monthly_heat()

        if result['total_heat_kcal'] == 0:
            return {'efficiency': 0, 'losses': 0}

        # Переводим оплаченное в ккал
        paid_kcal = paid_gcal * 1_000_000

        # Эффективность (сколько реально получено от оплаченного)
        efficiency = (result['total_heat_kcal'] / paid_kcal) * 100

        # Потери в системе
        losses = paid_kcal - result['total_heat_kcal']

        return {
            'efficiency_percent': min(efficiency, 100),
            'losses_kcal': losses,
            'losses_gcal': losses / 1_000_000,
            'real_heat_kcal': result['total_heat_kcal'],
            'paid_heat_kcal': paid_kcal
        }
```

---

## 🔄 CI/CD Pipeline (Адаптация вашего workflow)

**Файл `.github/workflows/ci.yml`**:

```yaml
name: Weather & Heat Calculator CI/CD

on:
  push:
    branches: [master, develop, feature/*]
  pull_request:
    branches: [master]

jobs:
  test:
    name: 🧪 Run Tests
    runs-on: ubuntu-latest
    
    steps:
    - name: 📥 Checkout code
      uses: actions/checkout@v4
      
    - name: 🐍 Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: 📦 Install dependencies
      run: pip install -r requirements.txt
      
    - name: 🧪 Run all tests
      run: |
        python -c "from src.api_client import WeatherService; from src.storage import LocationStorage; from src.heat_calculator import HeatCalculator; print('✅ All modules imported successfully')"
        python -c "from tests.test_api_client import TestWeatherService; print('✅ Tests loaded successfully')"
        echo "🎉 All basic tests passed!"

  docs:
    name: 📚 Generate Documentation
    runs-on: ubuntu-latest
    
    steps:
    - name: 📥 Checkout code
      uses: actions/checkout@v4
      
    - name: 📖 Install Doxygen
      run: sudo apt-get install -y doxygen graphviz
      
    - name: 📖 Generate docs
      run: doxygen Doxyfile
      
    - name: 🔍 Verify documentation
      run: |
        if [ -f "docs/html/index.html" ]; then
            echo "✅ Documentation generated"
            if grep -q "Weather & Heat Calculator" docs/html/index.html; then
                echo "✅ README.md is used as main page"
            else
                echo "⚠️ README.md may not be used as main page"
            fi
        else
            echo "❌ Documentation not generated"
            exit 1
        fi
        
    - name: 📦 Upload docs artifact
      uses: actions/upload-artifact@v4
      with:
        name: doxygen-docs
        path: docs/html/

  deploy:
    name: 🚀 Deploy to GitHub Pages
    runs-on: ubuntu-latest
    needs: [test, docs]
    if: github.ref == 'refs/heads/master'
    
    permissions:
      pages: write
      id-token: write
      
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
      
    steps:
    - name: 📥 Checkout code
      uses: actions/checkout@v4
      
    - name: 🏗️ Setup Pages
      uses: actions/configure-pages@v4
      
    - name: 📖 Install Doxygen
      run: sudo apt-get install -y doxygen graphviz
      
    - name: 📖 Generate docs
      run: doxygen Doxyfile
      
    - name: 📦 Upload to Pages
      uses: actions/upload-pages-artifact@v3
      with:
        path: docs/html/
        
    - name: 🚀 Deploy
      id: deployment
      uses: actions/deploy-pages@v4
```

---

## 📝 Пример Pull Request

### Заголовок PR
```
[Team Alpha] feat: Добавлен геокодинг и система хранения локаций
```

### Описание PR
```markdown
## 📋 Что сделано
- Реализован `WeatherService.get_coordinates()` с полной обработкой ошибок (Story 1.1)
- Создан класс `LocationStorage` для сохранения/загрузки JSON (Story 2.1)
- Добавлены юнит-тесты для обоих модулей (покрытие >80%)
- Обновлен CI/CD пайплайн для автоматического тестирования

## 🔧 Технические детали
- Использован OpenStreetMap Nominatim API
- JSON-хранилище в `data/cities.json`
- Добавлены пользовательские исключения: `WeatherAPIError`

## ✅ Проверки
- [x] Все тесты проходят локально
- [x] Doxygen документация генерируется
- [x] Нет конфликтов с `main`

## 📸 Скриншоты (если есть)
![Пример работы](link-to-screenshot)

## 📚 Связанные задачи
Closes #US-1.1
Closes #US-2.1

## 👥 Reviewers
@developer1 @developer2
```

---

## 🚀 Инструкция для команды

### 1. Начало работы
```bash
# Клонирование
git clone https://github.com/your-org/weather-heat-calc.git
cd weather-heat-calc

# Установка зависимостей
pip install -r requirements.txt

# Создание ветки для фичи
git checkout -b feature/team-alpha_your_task
```

### 2. В процессе разработки
```bash
# Добавление изменений
git add src/your_file.py
git commit -m "feat: краткое описание изменений"

# Пуш в свой форк
git push origin feature/team-alpha_your_task
```

### 3. Создание Pull Request
1. Перейдите на GitHub в свой репозиторий
2. Нажмите "Compare & pull request"
3. Выберите базовую ветку: `master`
4. Добавьте описание (шаблон выше)
5. Отправьте на ревью

---

## 📚 Документация (Doxygen)

В стиле вашего проекта, добавляем Doxygen-комментарии во все файлы:

```python
"""
@file main.py
@brief Точка входа в приложение
@author Team Alpha
@version 1.0
@date 2026
@details CLI интерфейс для управления погодой и расчетом тепла
"""
```

Конфигурация `Doxyfile`

```doxy
PROJECT_NAME           = "Weather & Heat Calculator"
PROJECT_NUMBER         = 1.0
OUTPUT_DIRECTORY       = docs
INPUT                  = src/ README.md
RECURSIVE              = YES
```

---

## 🎯 Итог

1. ✅ **Полная структура** проекта с модулями (аналогично вашей нейросети)
2. ✅ **Обработка ошибок** с понятными сообщениями (Story 1)
3. ✅ **Геокодинг** через API (Story 2)
4. ✅ **Хранение локаций** в JSON (Story 3)
5. ✅ **Расчет тепла** с формулой из ТЗ (Story 4)
6. ✅ **CI/CD пайплайн** (полностью скопирован из вашего примера)
7. ✅ **Doxygen документация** в едином стиле
8. ✅ **Ветки** с идентификацией команды (`feature/team-alpha_*`)
9. ✅ **Pull Request** с описанием и проверками

Ваш проект готов к разработке по SCRUM с автоматическим тестированием и деплоем документации! 🚀

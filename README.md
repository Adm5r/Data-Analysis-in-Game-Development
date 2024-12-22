# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1-5
- Иванова Ивана Варкравтовна
- НМТ-233511
Отметка о выполнении заданий (заполняется студентом):

| Лабораторная работа | Выполнение | Баллы |
| ------ | ------ | ------ |
| Лабораторная работа 1 | * |  |
| Лабораторная работа 2 | * |  |
| Лабораторная работа 3 | * |  |
| Лабораторная работа 4 | * |  |
| Лабораторная работа 5 | * |  |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Выполнение всей Лабораторных работ ( от 1 до 5 )
- Данные о работе: Номер лабораторной,выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic✨ (И вера в лучшее)

# Лабораторная работа №1
| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Залание 1 | * |  |
| Задание 2 | * |  |
| Задание 3 | * |  |
## Цель работы
Установить необходимое программное обеспечение, которое пригодится для создания интеллектуальных моделей на Python. Рассмотреть процесс установки игрового движка Unity для разработки игр.
## Задание 1
### Написать программу Hello World на Python с запуском в Jupiter Notebook.
```py
print('Hello world')
```
![Изображение из Jupiter Notebook](https://github.com/user-attachments/assets/cd8646ca-6956-4222-8691-b611ef1e659e)
## Задание 2
### Написать программу Hello World на C# с запуском на Unity
```c#
using UnityEngine;

public class odsfgj : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Debug.Log("Hello world");
    }
}
```
Как можно увидеть в этом случаи код является сложнее так как сам язык является более масштабируемым. 
## Задание 3
### Оформить отчет в виде документации на github (markdown-разметка).
Отчёт был оформлен по всем требованием представленные в файле ["Workshop#1-Установка программного обеспечения"](https://docs.google.com/document/d/1siJZTKkP5gJd--WsBzyDbTRI74BsrlbgqnTPW-z5-NQ/edit?usp=sharing)
и был немного дополнен для удобства проверки.
## Выводы
Входе выполнение лабораторной работе мы установили програмное опеспесение для выполнение курса ([Unity](https://unity.com/),[Anaconda](https://www.anaconda.com/),[Visual Studio Code](https://code.visualstudio.com/) и т.д) и научились базовый программе на 2 языках (Python,C#) по выводу сообщении "Hello world" и "Hello world" в Unity

# Лабораторная работа №2

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Залание 1 | * |  |
| Задание 2 | * |  |
## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.
## Задание 1
### Выберите одну из игровых переменных в игре СПАСТИ РТФ: Выживание (HP, SP, игровая валюта, здоровье и т.д.), опишите её роль в игре, условия изменения / появления и диапазон допустимых значений. 
### Экономическая модель игры: СПАСТИ РТФ: Выживание

Экономическая система игры представляет собой циклический процесс, где заработок, расходы и прогресс игрока напрямую зависят от его активности.

#### 1. Ресурсы
- **Монеты**: Основная внутриигровая валюта, получаемая за уничтожение зомби.
- **Патроны**: Расходуемый ресурс для стрельбы из пистолета. Покупаются за монеты.
- **Здоровье**: Жизнеспособность игрока.

#### 2. Доход
- **Убийство зомби**: Каждый убитый зомби приносит игроку доход.

#### 3. Расходы
- **Оружие и апгрейды**:
  - **Пистолет**: Можно улучшать скорострельность и урон.
  - **Патроны**: Ограниченный ресурс, который нужно периодически докупать.
- **Здоровье**: Возможна покупка способности **Vampyrism**.

#### 4. Баланс ресурсов
- Монеты игрок должен зарабатывать быстрее, чем тратить их на базовые ресурсы (патроны и здоровье), чтобы оставались средства на апгрейды.
- Чрезмерный дефицит здоровья или патронов делает игру слишком сложной и демотивирующей.
- Для поддержания интереса стоимость улучшений должна увеличиваться экспоненциально, а награды за успехи — постепенно расти.

#### 5. Роль здоровья в экономической модели
- **Стимул к тратам**: Потеря здоровья вынуждает игрока покупать аптечки или другие способы восстановления, создавая постоянный спрос на монеты.
- **Тактический ресурс**: Здоровье определяет, сколько ошибок игрок может допустить, прежде чем проиграет. Это влияет на выбор стратегии: атаковать агрессивно или экономить ресурсы.
- **Стимул прогресса**: Улучшения здоровья (например, увеличение максимального HP или регенерация) мотивируют игрока зарабатывать больше монет для дальнейших улучшений.
Грубо самая лучшая идея это добавить вамперизм как например например как Рейн из экшен-дилогии BloodRayne могла в любой момент запрыгнуть на врага и начать пить кровь. Причём для этого не нужно было соблюдать какие-то особые условия: просто подошёл поближе, нажал клавишу, и героиня уже восстанавливает здоровье, а враг замертво падает после укуса.
![0cfe08007948b3f2aa51faca8cedb5a8-1](https://github.com/user-attachments/assets/128f3140-7d2e-44cf-88ca-ec588b1eb9a5)
## Задание 2
###  С помощью скрипта на языке Python заполните google-таблицу данными, описывающими выбранную игровую переменную в игре “СПАСТИ РТФ:Выживание”.
- [Google таблицу](https://urfume-my.sharepoint.com/:x:/g/personal/evgeny_mitriashin_urfu_me/EQRybL5H5TFGpyEU57xbVQ4Bxl8_G7NkRvH93oXRrJNnCw?e=4urB65)
```py
import gspread
import numpy as np

client = gspread.service_account(filename='unitydatascience-440712-4ca8beda3fa8.json')
spreadsheet = client.open("UnitySheets")

health_values = np.random.randint(0, 30, 10)
time_stamps = list(range(1, 11))
damage_values = np.random.randint(1, 15, 10)

for index, time_point in enumerate(time_stamps, start=1):
    initial_health = health_values[index - 1]
    inflicted_damage = damage_values[index - 1]
    updated_health = initial_health - inflicted_damage
    updated_health = max(updated_health, 0)  # Здоровье не может быть меньше 0
    status = "Жив" if updated_health > 0 else "Мертв"
    
    # Обновление данных в Google Sheets
    spreadsheet.sheet1.update(f'A{index}', [[time_point]])  # Время
    spreadsheet.sheet1.update(f'B{index}', [[initial_health]])  # Исходное здоровье
    spreadsheet.sheet1.update(f'C{index}', [[updated_health]])  # Текущее здоровье
    spreadsheet.sheet1.update(f'D{index}', [[inflicted_damage]])  # Урон
    spreadsheet.sheet1.update(f'E{index}', [[status]])  # Статус игрока

    print(f"Текущее здоровье: {updated_health}, Статус: {status}")

```
## Выводы
В ходе этой лабораторной работы я освоил работу с API и Jupyter Notebook. Разбор каждой интересны. Работа оказалась интересной и  полезна.

# Лабораторная работа №3
| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Залание 1 | * |  |
| Задание 2 | * |  |
| Задание 3 | * |  |
## Цель работы
Научиться работать с балансировкой оружия в игре 
## Задание 1
### Расширьте варианты доступного оружия в игре.
Ход работы:
Решить какое оружие необходимо для расширения вариативности выбора в игре Save RTF и сбалансировать его, используя гугл таблицу
Для данной работы я выбрал копье, лук, арбалет, миномет и огнемет
Ссылка на заполненую по шаблону [Google таблицу](https://docs.google.com/spreadsheets/d/12V0QD20hEG96FhofnroFfmMycMuCZ-Wvk7E-xRa7ONo/edit?usp=sharing)
## Задание 2
### Визуализируйте параметры оружия в таблице. Постройте примеры для следующих математических величин:
- Среднеквадратическое отклонение (СКО)
- Разброс урона оружия
- Вариативность времени отклика игрока (реакция на события)
```py
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

def simulate_shots(weapon_name, num_shots, damage_per_shot, hit_probabilities, ax):
    # Генерация случайных выстрелов с отклонениями
    np.random.seed(rnd.randint(32, 64))
    shots_coords = np.random.normal(loc=0, scale=5, size=(num_shots, 2))

    # Вычисляем отклонения (расстояния от центра цели)
    distances = np.linalg.norm(shots_coords, axis=1)

    # Проверяем попадания
    hits, misses = [], []
    for i, distance in enumerate(distances):
        distance_index = min(int(distance), len(hit_probabilities) - 1)
        hit_chance = hit_probabilities[distance_index] / 100
        if np.random.rand() < hit_chance:
            hits.append(shots_coords[i])
        else:
            misses.append(shots_coords[i])

    hits = np.array(hits)
    misses = np.array(misses)

    # Выводим результаты
    print(f"{weapon_name}: {len(hits)} попаданий из {num_shots}")
    print(f"СКО отклонений: {np.std(distances):.2f} пикселей")

    # Визуализация
    if len(hits) > 0:
        ax.scatter(hits[:, 0], hits[:, 1], color='green', label='Попадания')
    if len(misses) > 0:
        ax.scatter(misses[:, 0], misses[:, 1], color='red', label='Промахи')
    ax.scatter(0, 0, color='blue', s=100, label='Цель')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_title(f"{weapon_name}\nУрон за выстрел: {damage_per_shot}")
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

# Параметры для различных оружий
weapons = [
    ("Копье", 10, 4, [100.00, 83.33, 83.33, 66.67, 66.67, 50.50, 33.33, 33.33, 00.00]),
    ("Лук", 50, 4, [16.67, 33.33, 33.33, 66.67, 66.67, 66.67, 66.67, 66.67, 83.33, 83.33]),
    ("Огнемет", 20, 4, [83.33, 66.67, 50.00, 50.00, 33.33, 16.67, 16.67]),
    ("Миномет", 15, 8, [16.67, 16.67, 33.33, 33.33, 33.33, 50.00, 66.67, 50.00, 83.33, 100.00, 100.00]),
    ("Арбалет", 30, 3, [66.67, 66.67, 66.67, 50.00, 50.00, 50.00, 33.33, 33.33])
]

# Создаем фигуру и подграфики
fig, axs = plt.subplots(3, 2, figsize=(12, 18))  # Увеличено количество подграфиков
axs = axs.flatten()

# Запускаем симуляцию для каждого оружия
for i, weapon in enumerate(weapons):
    simulate_shots(*weapon, axs[i])

plt.tight_layout()  # Упаковываем подграфики
plt.show()
```
![pythonw_Sr5INI84H2](https://github.com/user-attachments/assets/ed1b2dc3-5325-41f3-a936-071fdf2bc6b8)
## Задание 3
### Визуализировать данные из google-таблицы с помощью Python 
```py
import numpy as np
import matplotlib.pyplot as plt
import random as rnd

def simulate_shots(weapon_name, num_shots, damage_per_shot, hit_probabilities, ax):
    """
    Симуляция стрельбы для данного оружия.
    """
    # Устанавливаем случайное семя для генерации
    np.random.seed(rnd.randint(32, 64))
    # Генерируем координаты выстрелов с нормальным отклонением
    shots_coords = np.random.normal(loc=0, scale=5, size=(num_shots, 2))

    # Рассчитываем расстояния до цели
    distances = np.linalg.norm(shots_coords, axis=1)

    # Проверяем попадания
    hits, misses = [], []
    for i, distance in enumerate(distances):
        # Определяем вероятность попадания
        distance_index = min(int(distance), len(hit_probabilities) - 1)
        hit_chance = hit_probabilities[distance_index] / 100
        if np.random.rand() < hit_chance:
            hits.append(shots_coords[i])
        else:
            misses.append(shots_coords[i])

    hits = np.array(hits)
    misses = np.array(misses)

    # Выводим статистику
    print(f"{weapon_name}: {len(hits)} попаданий из {num_shots}")
    print(f"СКО отклонений: {np.std(distances):.2f} пикселей")

    # Визуализация попаданий и промахов
    if len(hits) > 0:
        ax.scatter(hits[:, 0], hits[:, 1], color='green', label='Попадания')
    if len(misses) > 0:
        ax.scatter(misses[:, 0], misses[:, 1], color='red', label='Промахи')
    ax.scatter(0, 0, color='blue', s=100, label='Цель')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_title(f"{weapon_name}\nУрон за выстрел: {damage_per_shot}")
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

# Параметры оружий: название, количество выстрелов, урон за выстрел, вероятность попадания
weapons = [
    ("Копье", 10, 4, [100.00, 83.33, 83.33, 66.67, 66.67, 50.50, 33.33, 33.33, 0.00]),
    ("Лук", 50, 4, [16.67, 33.33, 33.33, 66.67, 66.67, 66.67, 66.67, 66.67, 83.33, 83.33]),
    ("Огнемет", 20, 4, [83.33, 66.67, 50.00, 50.00, 33.33, 16.67, 16.67]),
    ("Миномет", 15, 8, [16.67, 16.67, 33.33, 33.33, 33.33, 50.00, 66.67, 50.00, 83.33, 100.00, 100.00]),
    ("Арбалет", 30, 3, [66.67, 66.67, 66.67, 50.00, 50.00, 50.00, 33.33, 33.33])
]

# Создаем графики для каждого оружия
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flatten()

for i, weapon in enumerate(weapons[:4]):  # Ограничиваем до 4 для отображения
    simulate_shots(*weapon, axs[i])

plt.tight_layout()
plt.show()
```
## Выводы
Я освоил работу с балансировкой параметров, и эта для меня как itch io разрабочика это помогло для проэктов. Я удивился что балансировать это довольно просто.
# Лабораторная работа №4
| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Залание 1 | * |  |
## Цель работы
## Задание 1
###  Реализовать перцептрон, который умеет производить вычисления в проекте Unity

## Выводы


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

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic✨

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
| Задание 3 | * |  |

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
В ходе этой лабораторной работы я освоила работу с API и Jupyter Notebook. Разбор каждой задачи потребовал значительных усилий, так как в процессе возникало множество нюансов, требующих дополнительного времени на изучение. Работа оказалась интересной и насыщенной полезной информацией.

# Лабораторная работа №3
| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Залание 1 | * |  |
| Задание 2 | * |  |
| Задание 3 | * |  |
## Цель работы
## Задание 1
### 
## Задание 2
###  
## Выводы

# Лабораторная работа №4
| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Залание 1 | * |  |
## Цель работы
## Задание 1
### 
## Выводы


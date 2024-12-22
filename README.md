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
| Лабораторная работа 5 | # |  |

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
| Залание 1 | * | 20 |
| Задание 2 | * | 60 |
| Задание 3 | * | 60 |
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
| Залание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |
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
| Залание 1 | * | 60 |
| Задание 2 | * | 60 |
| Задание 3 | * | 20 |
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
| Залание 1 | * | 20 |
| Залание 2 | * | 60 |
| Залание 3 | * | 20 |
## Цель работы
## Задание 1
###  Реализовать перцептрон, который умеет производить вычисления в проекте Unity
Перцептрон, вычисляющий OR. Он работает корректно, так как TotalError = 0
![image](https://github.com/user-attachments/assets/4171ceca-0b6b-4057-97c5-6fbdc6f12871)
![image](https://github.com/user-attachments/assets/86eb9606-dad1-4fa4-8505-eab4ffcb44ee)
## AND 
Перцептрон, вычисляющий AND.Он так же работает корректно, так как TotalError = 0, как и в прошлом примере
![image](https://github.com/user-attachments/assets/523f39d3-c55f-4887-b902-4a3e6a45e461)
![image](https://github.com/user-attachments/assets/c57596a5-164d-49cc-a0ea-ec34ed2362b3)
## NAND 
Перцептрон, вычисляющий NAND, аналогично TotalError = 0. Он работает без ошибок.
![image](https://github.com/user-attachments/assets/591aea0c-ba03-405a-9f96-8da34cba8b22)
![image](https://github.com/user-attachments/assets/e11194e6-bf08-42b5-a2c2-e4f4f489d754)
## XOR 
 TotalError = 0. Перцептрон работает корректно. Код для работы этой функции:
 ```py
 using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class XORNetwork : MonoBehaviour
{
    [System.Serializable]
    public class DataSet
    {
        public double[] inputs;
        public double target;
    }

    [SerializeField] private string modelName;
    [SerializeField] private int iterations = 20; // Увеличил количество итераций для обучения

    [SerializeField] private GameObject inputObj1;
    [SerializeField] private GameObject inputObj2;
    [SerializeField] private GameObject outputObj;

    private Color outputColor;

    public DataSet[] data;

    private double[,] inputToHiddenWeights;
    private double[] hiddenToOutputWeights;
    private double[] hiddenLayer;
    private double[] hiddenBiases;
    private double outputBias;
    private double learningRate = 0.1;

    private StreamWriter fileWriter = new StreamWriter("trainingResults.csv");

    void InitializeNetwork()
    {
        inputToHiddenWeights = new double[2, 2];
        hiddenToOutputWeights = new double[2];
        hiddenBiases = new double[2];

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                inputToHiddenWeights[i, j] = Random.Range(-1.0f, 1.0f);
            }
            hiddenToOutputWeights[i] = Random.Range(-1.0f, 1.0f);
            hiddenBiases[i] = Random.Range(-1.0f, 1.0f);
        }

        outputBias = Random.Range(-1.0f, 1.0f);
        hiddenLayer = new double[2];
    }

    double ApplyActivationFunction(double value)
    {
        return 1.0 / (1.0 + Mathf.Exp((float)-value));
    }

    double DerivativeOfActivation(double value)
    {
        return value * (1 - value);
    }

    void TrainNetwork(int epochs)
    {
        InitializeNetwork();

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            double cumulativeError = 0;

            foreach (var example in data)
            {
                // Propagate through hidden layer
                for (int i = 0; i < 2; i++)
                {
                    hiddenLayer[i] = 0;
                    for (int j = 0; j < 2; j++)
                    {
                        hiddenLayer[i] += example.inputs[j] * inputToHiddenWeights[j, i];
                    }
                    hiddenLayer[i] += hiddenBiases[i];
                    hiddenLayer[i] = ApplyActivationFunction(hiddenLayer[i]);
                }

                // Calculate output
                double output = 0;
                for (int i = 0; i < 2; i++)
                {
                    output += hiddenLayer[i] * hiddenToOutputWeights[i];
                }
                output += outputBias;
                output = ApplyActivationFunction(output);

                double error = example.target - output;
                double outputErrorGradient = error * DerivativeOfActivation(output);

                // Update weights for the hidden layer and output layer
                for (int i = 0; i < 2; i++)
                {
                    double hiddenError = outputErrorGradient * hiddenToOutputWeights[i];
                    double hiddenErrorGradient = hiddenError * DerivativeOfActivation(hiddenLayer[i]);

                    for (int j = 0; j < 2; j++)
                    {
                        inputToHiddenWeights[j, i] += learningRate * hiddenErrorGradient * example.inputs[j];
                    }

                    hiddenBiases[i] += learningRate * hiddenErrorGradient;
                    hiddenToOutputWeights[i] += learningRate * outputErrorGradient * hiddenLayer[i];
                }

                outputBias += learningRate * outputErrorGradient;

                cumulativeError += Mathf.Abs((float)error);
            }
        }
    }

    double PredictOutput(double[] inputs)
    {
        // Forward pass
        for (int i = 0; i < 2; i++)
        {
            hiddenLayer[i] = 0;
            for (int j = 0; j < 2; j++)
            {
                hiddenLayer[i] += inputs[j] * inputToHiddenWeights[j, i];
            }
            hiddenLayer[i] += hiddenBiases[i];
            hiddenLayer[i] = ApplyActivationFunction(hiddenLayer[i]);
        }

        double result = 0;
        for (int i = 0; i < 2; i++)
        {
            result += hiddenLayer[i] * hiddenToOutputWeights[i];
        }
        result += outputBias;
        return ApplyActivationFunction(result) > 0.5 ? 1 : 0;
    }

    void Start()
    {
        TrainNetwork(iterations);
        
        // Получаем данные от объектов
        float input1 = Mathf.Round(inputObj1.GetComponent<Renderer>().material.color.r);
        float input2 = Mathf.Round(inputObj2.GetComponent<Renderer>().material.color.r);

        double[] inputValues = { input1, input2 };
        float result = (float)PredictOutput(inputValues);

        outputColor = new Color(result, result, result);
        outputObj.GetComponent<Renderer>().material.color = outputColor;
    }

    [System.Serializable]
    private class DataCollection
    {
        public string[][] values;
    }
}
```
![image](https://github.com/user-attachments/assets/019e248f-5711-4188-8169-2123e2e2382f)
![image](https://github.com/user-attachments/assets/c69a13ec-0657-4412-84d7-57c5c94c8908)
## Задача 2 Необходимо построить графики, показывающие, как изменяется ошибка обучения в зависимости от количества эпох. При этом важно указать, от каких факторов зависит требуемое количество эпох для достижения оптимальных результатов в обу
В простых задачах, таких как AND , OR и NAND , которые являются линейно разделимыми , однослойный перцептрон способен эффективно решать задачу. В таких случаях количество эпох обучения — то есть число полных проходов по обучающим данным — может быть относительно небольшим. Этого достаточно, чтобы ошибка обучения снизилась до минимального уровня.
На старте обучения ошибка обычно достаточно высока. Это связано с тем, что веса нейронной сети инициализируются случайным образом, и, следовательно, выходы сети на первых этапах сильно отличаются от ожидаемых. Однако по мере того как процесс обучения продолжается, алгоритм постепенно корректирует веса, основываясь на полученных ошибках, что приводит к снижению ошибки и улучшению качества предсказаний.
Но при решении более сложных задач, таких как XOR , ситуация значительно меняется. Для этой задачи данные нелинейно разделимы , что делает задачу невозможной для решения с использованием однослойного перцептрона, даже если количество эпох будет очень большим. Это ограничение обусловлено архитектурой однослойного перцептрона, который не способен выявлять нелинейные зависимости в данных.
Для решения таких задач необходимы многослойные перцептроны (нейронные сети с несколькими скрытыми слоями). Многослойная структура позволяет модели выявлять более сложные зависимости, что делает возможным решение задач, таких как XOR, которые не могут быть решены с помощью однослойного перцептрона.
## Задание 3: Построить визуальную модель работы перцептрона на сцене Unity.
Функции и зависимости перекидываем на объекты
![firefox_ez2fxv74zn](https://github.com/user-attachments/assets/898e937f-b971-4c2d-be10-3bbb882ec5a5)
![41hTeih6gI](https://github.com/user-attachments/assets/0b9ac4f6-32d8-49e4-861e-ac1d606f6acf)
## Выводы
Лабораторная работа позволила углубиться в принципы работы перцептрона, изучить его эффективность на различных задачах, а также исследовать влияние архитектуры сети и числа эпох на качество обучения. Мы продемонстрировали, как многослойные модели решают задачи, которые не под силу однослойным сетям. 
## ✨Magic✨
### Анекдот
– В детстве я мечтал проходить сквозь стены. Но только в университете я научился этому.
– Магия и чародейство?
– Ярость и гипсокартон…

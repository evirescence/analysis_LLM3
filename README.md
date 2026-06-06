# Анализатор данных с ИИ

Веб-приложение на **Streamlit**: загрузите CSV/Excel, LLM-агент сам пишет и выполняет Python-код, возвращает метрики, графики и инсайты.

---

## Быстрый старт

### 1. Установить зависимости

```bash
pip install -r requirements.txt
```

### 2. Запустить приложение

```bash
streamlit run app.py
```

Откроется браузер на `http://localhost:8501`

### 3. Использование

1. В боковой панели введите **Anthropic API Key** (`sk-ant-...`)  
   (можно получить на https://console.anthropic.com)
2. Загрузите **CSV или Excel** файл
3. Введите вопрос или выберите пример из боковой панели
4. Нажмите **«Запустить анализ»**

---

## Структура проекта

```
datasage/
├── lab2.py            
├── requirements.txt 
└── README.md
```

---

## Что умеет агент

| Запрос | Что делает агент |
|--------|-----------------|
| "Полный EDA" | describe(), value_counts(), корреляции, гистограммы |
| "Найди выбросы" | IQR / Z-score, boxplot |
| "Корреляционная матрица" | seaborn heatmap |
| "Топ-10 по метрике X" | groupby + bar chart |
| "Распределения признаков" | histplot / kdeplot для всех числовых колонок |
| Любой свободный вопрос | агент сам решает какой код написать |

---

## Пример входных данных

`mymoviedb.csv`:
```
Title,Genre,Vote_Average,Release_Date,Overview
Spider-Man: No Way Home,"Action, Adventure",8.3,2021-12-15,"Peter Parker..."
The Batman,"Crime, Mystery",8.1,2022-03-01,"In his second year..."
```

## Пример выходных данных

**Графики** (автоматически): корреляционный heatmap, гистограммы рейтингов, bar chart по жанрам.

**Инсайты (markdown)**:
```
## Ключевые находки

- Средний рейтинг датасета: **7.1 / 10** (медиана: 7.3)
- Топ-жанр: **Action** — 34% фильмов
- Сильная корреляция Vote_Average ↔ Popularity: r = 0.71
- Выбросы: 3 фильма с рейтингом < 4.0

##  Рекомендации
- Рассмотреть удаление 3 выбросов перед ML-моделированием
- Признак Release_Date можно преобразовать в год/сезон для анализа трендов
```

---

## Технологии

| Компонент | Технология |
|-----------|-----------|
| UI | Streamlit |
| LLM | Anthropic claude-opus-4-5 |
| Агентность | Anthropic tool_use (function calling) |
| Code interpreter | `exec()` в изолированном namespace |
| Графики | matplotlib / seaborn / plotly |
| Данные | pandas |

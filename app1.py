import streamlit as st
import sqlite3
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import pandas as pd
import numpy as np

DB_PATH = r"C:\Users\user\Desktop\PythonProject\weather.db"

st.set_page_config(page_title="WeatherInsight", layout="wide")
st.title("🌦️ WeatherInsight: Погодные тренды")

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pl.read_database("SELECT * FROM weather ORDER BY date", conn)
    conn.close()
    return df


try:
    df = load_data()
except Exception as e:
    st.error("❌ Не удалось загрузить данные. Убедитесь, что база данных существует и доступна.")
    st.stop()

df = df.with_columns([
    pl.col("avg_temp").fill_null(df["avg_temp"].mean()),
    pl.col("total_precip").fill_null(0),
    pl.col("avg_wind").fill_null(df["avg_wind"].mean()),
    pl.col("is_rainy").fill_null(False)
])

df = df.with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))

st.subheader("📊 Общая статистика")
total_records = len(df)
unique_cities = df["city"].n_unique()
st.write(f"Всего записей: {total_records}")
st.write(f"Уникальных городов: {unique_cities}")

st.subheader("🗓️ Выбор диапазона дат")
min_date = df["date"].min()
max_date = df["date"].max()

start_date, end_date = st.date_input(
    "Выберите диапазон дат",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

cities = sorted(df["city"].unique().to_list())
selected_cities = st.multiselect("Выберите города для сравнения", cities, default=cities[:1])

if not selected_cities:
    st.warning("Пожалуйста, выберите хотя бы один город.")
    st.stop()

filtered_data = df.filter(
    (pl.col("city").is_in(selected_cities)) &
    (pl.col("date") >= start_date) &
    (pl.col("date") <= end_date)
)

def categorize_comfort(row):
    temp = row["avg_temp"]
    wind = row["avg_wind"]
    if 15 <= temp <= 25 and wind < 5:
        return "комфортно"
    elif 10 <= temp <= 30 and wind < 10:
        return "умеренно комфортно"
    else:
        return "некомфортно"

filtered_data = filtered_data.with_columns(
    pl.struct(["avg_temp", "avg_wind"])
    .map_elements(categorize_comfort)
    .alias("comfort_level")
)

st.subheader("📋 Исходные данные")
show_raw_data = st.checkbox("Показать исходные данные", value=False)
if show_raw_data:
    sort_column = st.selectbox("Столбец для сортировки", filtered_data.columns)
    sort_order = st.radio("Порядок сортировки", ["По возрастанию", "По убыванию"])
    ascending = sort_order == "По возрастанию"

    page_size = st.slider("Количество строк на странице", 10, 100, 20)
    total_pages = len(filtered_data) // page_size + 1
    page_num = st.number_input("Страница", 1, total_pages, 1)

    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size

    sorted_data = filtered_data.sort(sort_column, descending=not ascending)
    page_data = sorted_data[start_idx:end_idx]

    st.dataframe(page_data.to_pandas())
    st.write(f"Страница {page_num} из {total_pages} (всего записей: {len(filtered_data)})")

st.subheader("🔎 Разведочный анализ данных")

col1, col2, col3 = st.columns(3)

with col1:
    fig_hist_temp = px.histogram(
        filtered_data.to_pandas(),
        x="avg_temp",
        title="Распределение температуры",
        labels={"avg_temp": "Температура (°C)"}
    )
    st.plotly_chart(fig_hist_temp, use_container_width=True)

with col2:
    fig_hist_precip = px.histogram(
    filtered_data.to_pandas(),
    x = "total_precip",
    title = "Распределение осадков",
    labels = {"total_precip": "Осадки (мм)"}
)
st.plotly_chart(fig_hist_precip, use_container_width=True)

with col3:
    fig_hist_wind = px.histogram(
filtered_data.to_pandas(),
x = "avg_wind",
title = "Распределение скорости ветра",
labels = {"avg_wind": "Скорость ветра (м/с)"}
)
st.plotly_chart(fig_hist_wind, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig_box_temp = px.box(
filtered_data.to_pandas(),
x = "city",
y = "avg_temp",
title = "Температура по городам (boxplot)",
labels = {"avg_temp": "Температура (°C)", "city": "Город"}
)
st.plotly_chart(fig_box_temp, use_container_width=True)

with col2:
    fig_box_precip = px.box(
filtered_data.to_pandas(),
x = "city",
y = "total_precip",
title = "Осадки по городам (boxplot)",
labels = {"total_precip": "Осадки (мм)", "city": "Город"}
)
st.plotly_chart(fig_box_precip, use_container_width=True)

# Сравнение погодных показателей между городами
st.subheader("📈 Сравнение городов")

comparison_metric = st.selectbox(
    "Метрика для сравнения",
    ["avg_temp", "total_precip", "avg_wind"],
    format_func=lambda x: {
        "avg_temp": "Средняя температура",
        "total_precip": "Общее количество осадков",
        "avg_wind": "Средняя скорость ветра"
    }[x]
)

if comparison_metric:

    comparison_data = filtered_data.group_by("city").agg([
        pl.col(comparison_metric).mean().alias("mean_value")
    ]).to_pandas()

fig_comparison = px.bar(
    comparison_data,
    x="city",
    y="mean_value",
    title=f"Сравнение {comparison_metric.replace('_', ' ')} по городам",
    labels={"city": "Город", "mean_value": f"{comparison_metric} (среднее)"}
)
st.plotly_chart(fig_comparison, use_container_width=True)

st.subheader("📅 Временные ряды и прогноз")

time_series_metric = st.selectbox(
    "Выберите метрику для временного ряда",
    ["avg_temp", "total_precip", "avg_wind"],
    key="time_series"
)

if selected_cities and time_series_metric:
    for city in selected_cities:
        city_data = filtered_data.filter(pl.col("city") == city)
        if not city_data.is_empty():

            city_df = city_data.to_pandas()

            city_df['rolling_mean'] = city_df[time_series_metric].rolling(window=7).mean()

            fig_time_series = go.Figure()

            fig_time_series.add_trace(go.Scatter(
                x=city_df['date'],
                y=city_df[time_series_metric],
                mode='lines',
                name=f'{city} (реальные)',
                opacity=0.7
            ))

            fig_time_series.add_trace(go.Scatter(
                x=city_df['date'],
                y=city_df['rolling_mean'],
                mode='lines',
                name=f'{city} (скользящее среднее, 7 дней)',
                line=dict(dash='dash')
            ))

            fig_time_series.update_layout(
                title=f'Динамика {time_series_metric.replace("_", " ")} в {city}',
                xaxis_title="Дата",
                yaxis_title=time_series_metric.replace("_", " "),
                hovermode='x unified'
            )

            st.plotly_chart(fig_time_series, use_container_width=True)

st.subheader("📊 Статистика по категориям")

col1, col2, col3 = st.columns(3)

with col1:
    if "temp_category" in filtered_data.columns:
        temp_categories = filtered_data.group_by("temp_category").agg([
            pl.col("temp_category").count().alias("count")
        ]).to_pandas()
        fig_temp_cat = px.pie(
            temp_categories,
            names="temp_category",
            values="count",
            title="Распределение температур"
        )
        st.plotly_chart(fig_temp_cat, use_container_width=True)
    else:
        st.warning("Столбец 'temp_category' недоступен")


with col2:
    if "precip_category" in filtered_data.columns:
        precip_categories = filtered_data.group_by("precip_category").agg([
            pl.col("precip_category").count().alias("count")
        ]).to_pandas()
        fig_precip_cat = px.pie(
            precip_categories,
            names="precip_category",
            values="count",
            title="Распределение осадков"
        )
        st.plotly_chart(fig_precip_cat, use_container_width=True)
    else:
        st.warning("Столбец 'precip_category' недоступен")


with col3:
    if "comfort_level" in filtered_data.columns:
        comfort_categories = filtered_data.group_by("comfort_level").agg([
            pl.col("comfort_level").count().alias("count")
        ]).to_pandas()
        fig_comfort_cat = px.pie(
            comfort_categories,
            names="comfort_level",
            values="count",
            title="Уровень комфортности"
        )
        st.plotly_chart(fig_comfort_cat, use_container_width=True)
    else:
        st.warning("Столбец 'comfort_level' недоступен")

st.subheader("🔎 Дополнительная аналитика")

st.write("Корреляция между показателями:")
corr_data = filtered_data.select(["avg_temp", "total_precip", "avg_wind"]).to_pandas().corr()
fig_corr = px.imshow(
    corr_data,
    text_auto=True,
    title="Корреляционная матрица"
)
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("⚠️ Аномалии")
anomaly_type = st.radio(
    "Тип аномалий для поиска",
    ["Экстремальная температура", "Сильные осадки", "Сильный ветер"]
)

if anomaly_type == "Экстремальная температура":
    anomalies = filtered_data.filter(
        (pl.col("avg_temp") < filtered_data["avg_temp"].quantile(0.1)) |
        (pl.col("avg_temp") > filtered_data["avg_temp"].quantile(0.9))
    )
elif anomaly_type == "Сильные осадки":
    anomalies = filtered_data.filter(pl.col("total_precip") > 20)
else:
    anomalies = filtered_data.filter(pl.col("avg_wind") > 15)

if not anomalies.is_empty():
    st.write(f"Найдено {len(anomalies)} аномалий ({anomaly_type}):")
    st.dataframe(anomalies.to_pandas())
else:
    st.info(f"Аномалии ({anomaly_type}) не обнаружены.")

# Информация о приложении
st.sidebar.header("🤖 О приложении")
st.sidebar.write("""WeatherInsight" — интерактивное приложение для анализа исторических метеорологических данных.

**Возможности:**
- Выбор диапазона дат
- Фильтрация по городам
- Визуализация временных рядов
- Прогнозирование (скользящее среднее)
- Разведочный анализ данных (EDA)
- Поиск аномалий
- Статистика по категориям погоды

**Данные:** исторические метеорологические данные с разбивкой по городам.""")

st.sidebar.header("�� Настройки визуализации")
show_histograms = st.sidebar.checkbox("Показать гистограммы", True)
show_boxplots = st.sidebar.checkbox("Показать boxplots", True)
show_comparison = st.sidebar.checkbox("Показать сравнение городов", True)


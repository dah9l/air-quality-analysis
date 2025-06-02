import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# проверяем есть ли файл
file_name = 'Air_Quality.csv'
if not Path(file_name).exists():
    print(f"Не найден файл: {file_name}")
    exit()

try:
    # читаем данные
    df = pd.read_csv(file_name, parse_dates=['Date'])
    print(f"Загружено {len(df)} строк данных")
    
    # какие колонки у нас есть для анализа
    numeric_columns = ['CO', 'CO2', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']
    available_cols = []
    for col in numeric_columns:
        if col in df.columns:
            available_cols.append(col)
    
    print(f"Найдены колонки: {available_cols}")
    
    # 2. Обработка пропусков (ИСПРАВЛЕНО - убрана цепочка операций)
    numeric_cols = ['CO', 'CO2', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10', 'AQI']
    available_cols = [col for col in numeric_cols if col in df.columns]

    print(f"Найдены колонки: {available_cols}")

    # считаем базовую статистику
    print("\n--- Основная статистика ---")
    summary_stats = {
        'mean_values': df[available_cols].mean(),
        'median_vals': df[available_cols].median(),
        'mode_vals': df[available_cols].mode().iloc[0] if len(df) > 0 else None
    }
    
    stats_table = pd.DataFrame(summary_stats)
    print(stats_table.round(2))

    # строим временные ряды (долго грузит с большим объемом данных)
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, col in enumerate(available_cols):
        color = colors[i % len(colors)]  # циклически используем цвета
        ax.plot(df['Date'], df[col], label=col, color=color, linewidth=1.2)
    
    ax.legend(loc='upper right')
    ax.set_title('Динамика загрязнения воздуха по времени', fontsize=14, fontweight='bold')
    ax.set_xlabel('Время')
    ax.set_ylabel('Концентрация')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('timeseries_pollution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n--- Первые 5 записей ---")
    first_rows = df[['Date'] + available_cols].head()
    print(first_rows.to_string(index=False))

    # гистограммы по каждому показателю
    print("\n--- Анализ распределений ---")
    
    for idx, col in enumerate(available_cols):
        # создаем отдельный график для каждой переменной
        fig = plt.figure(figsize=(8, 5))
        
        n_bins = min(25, int(len(df) / 10))  # адаптивное количество bins
        plt.hist(df[col].dropna(), bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Распределение {col}', fontweight='bold')
        plt.xlabel(f'Значения {col}')
        plt.ylabel('Количество наблюдений')
        
        # добавляем вертикальную линию среднего
        mean_value = df[col].mean()
        plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Среднее: {mean_value:.2f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'distribution_{col.replace(".", "_")}.png', dpi=120)
        plt.close()
        
        # выводим статистики для каждой переменной
        min_val, max_val = df[col].min(), df[col].max()
        avg_val = df[col].mean()
        med_val = df[col].median()
        
        print(f"\n{col}: диапазон [{min_val:.3f} - {max_val:.3f}]")
        print(f"  Среднее значение: {avg_val:.3f}")
        print(f"  Медиана: {med_val:.3f}")

    # boxplot - смотрим на выбросы
    if len(available_cols) > 1:
        plt.figure(figsize=(13, 7))
        
        # используем seaborn для красивого boxplot
        box_data = df[available_cols]
        sns.boxplot(data=box_data, palette="Set2")
        plt.title('Распределение показателей качества воздуха (Box Plot)', fontsize=13)
        plt.xticks(rotation=30)
        plt.ylabel('Значения')
        plt.tight_layout()
        plt.savefig('boxplots_comparison.png', dpi=130)
        plt.close()
        
        print("\n--- Квартили и выбросы ---")
        quantiles = df[available_cols].describe().loc[['min', '25%', '50%', '75%', 'max']]
        print(quantiles.round(3).T)

    # корреляционный анализ
    if len(available_cols) >= 2:
        correlation_matrix = df[available_cols].corr()
        
        plt.figure(figsize=(10, 8))
        # используем другую цветовую схему
        heatmap = sns.heatmap(correlation_matrix, 
                            annot=True, 
                            cmap='RdBu_r',  # красно-синяя схема
                            center=0,
                            square=True,
                            fmt='.2f',
                            cbar_kws={"shrink": .8})
        
        plt.title('Матрица корреляций между показателями', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig('correlations_heatmap.png', dpi=140)
        plt.close()
        
        print("\n--- Корреляции ---")
        print(correlation_matrix.round(3))

    # итоговая сводка
    print("\n" + "="*50)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("="*50)
    final_summary = pd.DataFrame({
        'Среднее': df[available_cols].mean(),
        'Медиана': df[available_cols].median(),
        'Ст.отклонение': df[available_cols].std()
    })
    print(final_summary.round(3))
    
    print(f"\nОбработано {len(df)} записей по {len(available_cols)} показателям")
    print("Графики сохранены в текущей папке")

except FileNotFoundError as fnf_error:
    print(f"Файл не найден: {fnf_error}")
except pd.errors.EmptyDataError:
    print("Файл пустой или поврежден")
except Exception as general_error:
    print(f"Произошла ошибка: {general_error}")

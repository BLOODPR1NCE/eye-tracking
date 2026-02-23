# data_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import json

def load_data(data_dir='simple_data'):
    files = ['y_train', 'y_val', 'y_test', 'left_train', 'right_train']
    data = {}
    for f in files:
        data[f] = np.load(f'{data_dir}/{f}.npy')
    
    y_all = np.vstack([data['y_train'], data['y_val'], data['y_test']])
    print(f"Загружено: {len(y_all)} samples")
    return data, y_all

def extract_features(left, right, y, n=1000):
    n = min(n, len(left))
    left_flat = left[:n].reshape(n, -1)
    right_flat = right[:n].reshape(n, -1)
    
    features = pd.DataFrame({
        'left_mean': left_flat.mean(axis=1),
        'left_std': left_flat.std(axis=1),
        'right_mean': right_flat.mean(axis=1),
        'right_std': right_flat.std(axis=1),
        'left_skew': [stats.skew(left_flat[i]) for i in range(n)],
        'right_skew': [stats.skew(right_flat[i]) for i in range(n)],
        'left_kurtosis': [stats.kurtosis(left_flat[i]) for i in range(n)],
        'right_kurtosis': [stats.kurtosis(right_flat[i]) for i in range(n)],
        'gaze_x': y[:n, 0],
        'gaze_y': y[:n, 1]
    })
    return features

def plot_heatmap(corr_matrix):
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                annot_kws={'size': 8}, fmt='.3f')
    plt.title('Тепловая карта корреляции признаков', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150)
    plt.show()

def plot_8graphs(df_coords, df_features, corr_matrix):
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    ax = axes.flatten()
    
    # 1-2. Гистограммы
    for i, col in enumerate(['gaze_x', 'gaze_y']):
        ax[i].hist(df_coords[col], bins=50, alpha=0.7)
        ax[i].set(xlabel=f'{col} координата', ylabel='Частота', title=f'Распределение {col.upper()}')
    
    # 3. Scatter plot
    sample = df_coords.sample(min(5000, len(df_coords)), random_state=42)
    ax[2].scatter(sample['gaze_x'], sample['gaze_y'], alpha=0.3, s=1, c='blue')
    ax[2].set(xlabel='X', ylabel='Y', title='Зависимость X от Y')
    
    # 4. 2D гистограмма
    h = ax[3].hist2d(sample['gaze_x'], sample['gaze_y'], bins=50, cmap='viridis')
    ax[3].set(xlabel='X', ylabel='Y', title='2D гистограмма')
    
    # 5. Box plot
    pd.melt(df_coords, value_vars=['gaze_x', 'gaze_y']).pipe(
        lambda x: sns.boxplot(x='variable', y='value', data=x, ax=ax[4],
                             palette=['skyblue', 'lightcoral']))
    ax[4].set(xlabel='Координата', ylabel='Значение', title='Box plot')
    
    # 6-7. Корреляции признаков глаз
    for j, (feat, color) in enumerate(zip(['mean', 'std'], ['green', 'orange'])):
        ax[5+j].scatter(df_features[f'left_{feat}'], df_features[f'right_{feat}'], 
                       alpha=0.5, s=10, c=color)
        ax[5+j].set(xlabel=f'Левый глаз ({feat})', ylabel=f'Правый глаз ({feat})',
                   title=f'Корреляция {feat} значений')
    
    # 8. Гистограмма корреляций
    corr_vals = [corr_matrix.iloc[i,j] for i in range(len(corr_matrix)) 
                 for j in range(i+1, len(corr_matrix))]
    ax[7].hist(corr_vals, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax[7].set(xlabel='Корреляция', ylabel='Частота', title='Распределение корреляций')
    
    plt.suptitle('АНАЛИЗ КОРРЕЛИРУЮЩИХ ПЕРЕМЕННЫХ (8 ГРАФИКОВ)', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('correlation_analysis_8graphs.png', dpi=150, bbox_inches='tight')
    plt.show()

    
def analyze_gaze_data():
    print("="*60 + "\nАНАЛИЗ ДАННЫХ MPIIGAZE\n" + "="*60)
    
    if not os.path.exists('simple_data'):
        print("Папка simple_data не найдена! Сначала запустите data_preparation_simple.py")
        return
    
    data, y_all = load_data()
    df_coords = pd.DataFrame(y_all, columns=['gaze_x', 'gaze_y'])
    
    print("\nСоздание признаков...")
    df_features = extract_features(data['left_train'], data['right_train'], data['y_train'])
    print(f"Создано {len(df_features.columns)} признаков")
    
    print("\n1. ТЕПЛОВАЯ КАРТА КОРРЕЛЯЦИИ")
    corr_matrix = df_features.corr()
    plot_heatmap(corr_matrix)
    print("Тепловая карта сохранена: correlation_heatmap.png")
    
    print("\n2. АНАЛИЗ КОРРЕЛИРУЮЩИХ ПЕРЕМЕННЫХ (8 ГРАФИКОВ)")
    plot_8graphs(df_coords, df_features, corr_matrix)
    print("Графики сохранены: correlation_analysis_8graphs.png")
    
    print("\n" + "="*60)
    print("Результаты сохранены:")
    print("  - correlation_heatmap.png")
    print("  - correlation_analysis_8graphs.png")
    print("="*60)

if __name__ == "__main__":
    analyze_gaze_data()
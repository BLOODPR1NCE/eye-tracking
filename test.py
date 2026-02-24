from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def load_data(data_dit='simple_data'):
    files = ['y_train', 'y_val', 'y_test', 'left_train', 'right_train']
    data = {}
    for f in files:
        data[f] = np.load(f'{data_dit}/{f}.npy')

    y_all = np.vstack([data['y_train'], data['y_val'], data['y_test']])
    return data, y_all

def extract_features(left, right, y):
    n = len(left)
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
                square=True, fmt='.3f')
    plt.title("Тепловая карта коррелации переменных")
    plt.tight_layout()
    plt.savefig('heatmap.png', dpi=150)
    plt.show()

def plot_8graphs(df_cords, df_features, corr_matrix):
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    ax = axes.flatten()

    for i, corr in enumerate(['gaze_x', 'gaze_y']):
        ax[i].hist(df_cords[corr], bins=50, alpha=0.7)
        ax[i].set(xlabel=f'{corr} координата', ylabel='Частота', title=f'распределение {corr}')

    sample = df_cords.sample(len(df_cords), random_state=42)
    ax[2].scatter(sample['gaze_x'], sample['gaze_y'], alpha=0.3, s=1, c='blue')
    ax[2].set(xlabel='X', ylabel='Y', title=f'Зависимости X от Y')

    ax[3].hist2d(sample['gaze_x'], sample['gaze_y'], bins=50, cmap='viridis')
    ax[3].set(xlabel='X', ylabel='Y', title=f'2D гистограмма')

    pd.melt(df_cords, value_vars=['gaze_x', 'gaze_y']).pipe(
        lambda x: sns.boxplot(x='variable', y='value', data=x, ax=ax[4], palette=['lightcoral', 'skyblue']))
    ax[4].set(xlabel='Координата', ylabel='Значение', title='Box plot')

    for j, (feat, color) in enumerate(zip(['mean', 'std'], ['green', 'orange'])):
        ax[5+j].scatter(df_features[f'left_{feat}'], df_features[f'right_{feat}'], alpha=0.5, c=color, s=10)
        ax[5+j].set(xlabel=f'Левый глаз ({feat})', ylabel=f'Правый глаз ({feat})', title=f'Корреляция {feat} значений')

    corr_vals = [corr_matrix.iloc[i,j] for i in range(len(corr_matrix)) for j in range(i+1, len(corr_matrix))]
    ax[7].hist(corr_vals, bins=20, alpha=0.7, color='purple')
    ax[7].set(xlabel='Корреляция', ylabel='Частота', title='Распределение корреляций')

    plt.suptitle("Анализ корреляции (8 графиков)")
    plt.tight_layout()
    plt.savefig('8graphs.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze():
    data, y_all = load_data()
    df_cords = pd.DataFrame(y_all, columns=['gaze_x', 'gaze_y'])
    df_features = extract_features(data['left_train'], data['right_train'], data['y_train'])
    corr_matrix = df_features.corr()
    plot_heatmap(corr_matrix)
    plot_8graphs(df_cords, df_features, corr_matrix)

if __name__ == '__main__':
    analyze()
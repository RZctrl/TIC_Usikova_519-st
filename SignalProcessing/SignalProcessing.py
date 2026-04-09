#Виконала Усікова Віра, варіант 10 - за списком групи 519-ст мій варіант 10

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
import os

#параметри сигналу відповідно до варіанту
n = 500 #довжина сигналу у відліках
Fs = 1000 #частота дискретизації
F_max = 21 #максимальна частота сигналу

F_filter = 28 #полоса пропуску фільтру



#Завдання практичної роботи 2
#1 Генерація випадкового сигналу
np.random.seed(42)
r_signal = np.random.normal(0, 10, n)

#2 Визначення відліків часу
time = np.arange(n) / Fs

#3 Розрахунок параметрів ФНЧ
w = F_max / (Fs / 2) #нормування частоти
sos_filter = signal.butter(3, w, btype='low', output='sos') #розрахунок коефіцієнтів фільтру 3 порядку

#4 Фільтрація сигналу
filt_signal = signal.sosfiltfilt(sos_filter, r_signal)

#Підготовка папки для збереження
figures_dir = './figures'
os.makedirs(figures_dir, exist_ok=True)

#5 Відображення результатів
def save_plot(x, y, xlabel, ylabel, title, filename):
    width_inch = 21 / 2.54
    height_inch = 14 / 2.54

    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'{figures_dir}/{filename}.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


#6 Побудова графіку відфільтрованого сигналу за допомогою створеної функції
save_plot(time, filt_signal,
          xlabel='Час (с)',
          ylabel='Амплітуда',
          title='Сигнал з максимальною частотою F_max = 21 Гц',
          filename='filt_signal')

#7 Розрахунок спектру сигналу
spectrum = fft.fft(filt_signal) #повне перетворення Фур'є
spectrum_shifted = np.abs(fft.fftshift(spectrum)) #зсув нульової частоти та обчислення модуля
freq = fft.fftfreq(n, 1/Fs) #частотні відліки
freq_shift = fft.fftshift(freq)

#8 Побудова графіку спектру за допомогою створеної функції
save_plot(freq_shift, spectrum_shifted,
          xlabel='Частота (Гц)',
          ylabel='Амплітуда спектру',
          title='Спектр відфільтрованого сигналу',
          filename='signal_spectr')



#Новий код з практичної роботи 3
def save_grid_plot(data_list, x_axis, titles, suptitle, xlabel, ylabel, filename):
    width_inch = 21 / 2.54
    height_inch = 14 / 2.54
    fig, axes = plt.subplots(2, 2, figsize=(width_inch, height_inch))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(x_axis, data_list[i], linewidth=1)
        ax.set_title(titles[i], fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(labelsize=12)
    fig.supxlabel(xlabel, fontsize=14)
    fig.supylabel(ylabel, fontsize=14)
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/{filename}.png', dpi=600, bbox_inches='tight')
    plt.close(fig)


#Дискретизація, спектри, відновлення, похибки
dt_values = [2, 4, 8, 16]
discrete_signals = []
discrete_spectrums = []
restored_signals = []
var_errors = []
snr_values = []

#Параметри фільтра для відновлення
w_restore = F_filter / (Fs / 2)
sos_restore = signal.butter(3, w_restore, btype='low', output='sos')

for Dt in dt_values:
    #Дискретизація
    discrete = np.zeros(n)
    discrete[::Dt] = filt_signal[::Dt]
    discrete_signals.append(discrete)

    #Спектр дискретизованого сигналу
    spectrum = fft.fft(discrete)
    spectrum_shifted = np.abs(fft.fftshift(spectrum))
    discrete_spectrums.append(spectrum_shifted)

    #Відновлення аналогового сигналу за допомогою ФНЧ
    restored = signal.sosfiltfilt(sos_restore, discrete)
    restored_signals.append(restored)

    #Похибка та метрики
    error = restored - filt_signal
    var_orig = np.var(filt_signal)
    var_err = np.var(error)
    var_errors.append(var_err)
    snr = var_orig / var_err if var_err > 0 else np.inf
    snr_values.append(snr)

titles_signal = [f'Dt = {dt}' for dt in dt_values]
titles_spectrum = [f'Спектр, Dt = {dt}' for dt in dt_values]
titles_restored = [f'Відновлений, Dt = {dt}' for dt in dt_values]

#1 Дискретизовані сигнали
save_grid_plot(discrete_signals, time, titles_signal,'Сигнал з кроком дискретизації Dt = 2, 4, 8, 16',
               'Час (с)', 'Амплітуда сигналу', 'discrete_signals')

#2 Спектри дискретизованих сигналів
save_grid_plot(discrete_spectrums, freq_shift, titles_spectrum,'Спектри дискретизованих сигналів Dt = 2, 4, 8, 16',
               'Частота (Гц)', 'Магнітуда', 'discrete_spectrums')

#3 Відновлені сигнали
save_grid_plot(restored_signals, time, titles_restored,'Відновлені аналогові сигнали з кроком дискретизації Dt = 2, 4, 8, 16',
               'Час (с)', 'Амплітуда', 'restored_signals')

#4 Залежність дисперсії від кроку дискретизації
plt.figure(figsize=(21/2.54, 14/2.54))
plt.plot(dt_values, var_errors, 'o-', linewidth=2, markersize=8)
plt.xlabel('Крок дискретизації Dt', fontsize=14)
plt.ylabel('Дисперсія', fontsize=14)
plt.title('Залежність дисперсії від кроку дискретизації', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'{figures_dir}/variance_vs_dt.png', dpi=600, bbox_inches='tight')
plt.close()

#5 Залежність сигнал-шум від кроку дискретизації
plt.figure(figsize=(21/2.54, 14/2.54))
plt.plot(dt_values, snr_values, 'o-', linewidth=2, markersize=8)  # логарифмічна шкала для наочності
plt.xlabel('Крок дискретизації Dt', fontsize=14)
plt.ylabel('ССШ', fontsize=14)
plt.title('Залежність співвідношення сигнал-шум від кроку дискретизації', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'{figures_dir}/ssh_vs_dt.png', dpi=600, bbox_inches='tight')
plt.close()

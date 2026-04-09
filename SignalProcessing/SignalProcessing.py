#Виконала Усікова Віра, варіант 10 - за списком групи 519-ст мій варіант 10

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft

#задавання параметрів сигналу відповідно до варіанту
n = 500 #довжина сигналу у відліках
Fs = 1000 #частота дискретизації
F_max = 21 #максимальна частота сигналу

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

    #збереження графіку
    plt.savefig(f'./figures/{filename}.png', dpi=600, bbox_inches='tight')
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

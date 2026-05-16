#Виконала Усікова Віра, варіант 10, група 519-ст

from random import randint
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sin, cos, pi
import scipy.fft

#Побудова спектрів
def spectrum(sequence):
    y_spectrum = np.abs(scipy.fft.fftshift(scipy.fft.fft(sequence)))
    x_spectrum = scipy.fft.fftshift(scipy.fft.fftfreq(len(sequence), 1 / len(sequence)))
    return x_spectrum[round(len(x_spectrum)/2):], y_spectrum[round(len(y_spectrum)/2):]

#Генерація випадкової цифрової послідовності
def create_sequence():
    sequence = np.zeros(1000)
    sequence[0:100] = randint(0, 1)
    sequence[100:200] = randint(0, 1)
    sequence[200:300] = randint(0, 1)
    sequence[300:400] = randint(0, 1)
    sequence[400:500] = randint(0, 1)
    sequence[500:600] = randint(0, 1)
    sequence[600:700] = randint(0, 1)
    sequence[700:800] = randint(0, 1)
    sequence[800:900] = randint(0, 1)
    sequence[900:1000] = randint(0, 1)
    return sequence

#Амплітудна модуляція (ASK)
def ask_modulation(frequency, sequence):
    sequence_ask = np.zeros(1000)
    for i in range(0, len(sequence)):
        sequence_ask[i] = sequence[i] * cos(2 * pi * frequency * i / 1000)
    return sequence_ask

#Амплітудна демодуляція
def ask_demodulation(frequency, sequence):
    ask_product = np.zeros(1000)
    ask_demodulated_signal = np.zeros(1000)
    threshold = np.ones(1000) * 25
    sequence_demodulated = np.zeros(1000)

    for i in range(0, len(sequence)):
        ask_product[i] = sequence[i] * cos(2 * pi * frequency * i / 1000)

    for i in range(1, 11):
        S = 0
        for t in range(0, 100):
            S = S + ask_product[t + 100 * (i - 1)]
            ask_demodulated_signal[t + 100 * (i - 1)] = S

    ask_demodulated = 0.5 * (np.sign(ask_demodulated_signal - threshold) + 1)

    for i in range(1, 11):
        for t in range(0, 100):
            sequence_demodulated[t + 100 * (i - 1)] = ask_demodulated[100 * i - 1]

    return ask_demodulated_signal, sequence_demodulated

#Фазова модуляція (PSK)
def psk_modulation(frequency, sequence):
    sequence_psk = np.zeros(1000)
    for i in range(0, len(sequence)):
        sequence_psk[i] = sin(2 * pi * frequency * i / 1000 + sequence[i] * pi + pi)
    return sequence_psk

#Фазова демодуляція
def psk_demodulation(frequency, sequence):
    psk_product = np.zeros(1000)
    psk_demodulated_signal = np.zeros(1000)
    threshold = np.ones(1000) * 25
    sequence_demodulated = np.zeros(1000)

    for i in range(0, len(sequence)):
        psk_product[i] = sequence[i] * sin(2 * pi * frequency * i / 1000)

    for i in range(1, 11):
        S = 0
        for t in range(0, 100):
            S = S + psk_product[t + 100 * (i - 1)]
            psk_demodulated_signal[t + 100 * (i - 1)] = S

    psk_demodulated = 0.5 * (np.sign(psk_demodulated_signal - threshold) + 1)

    for i in range(1, 11):
        for t in range(0, 100):
            sequence_demodulated[t + 100 * (i - 1)] = psk_demodulated[100 * i - 1]

    return psk_demodulated_signal, sequence_demodulated

#Частотна модуляція (FSK)
def fsk_modulation(frequency1, frequency2, sequence):
    sequence_fsk = np.zeros(1000)
    for i in range(0, len(sequence)):
        sequence_fsk[i] = (sequence[i] * sin(2 * pi * frequency1 * i / 1000) +
                           (abs(sequence[i] - 1)) * sin(2 * pi * frequency2 * i / 1000))
    return sequence_fsk

#Частотна демодуляція
def fsk_demodulation(frequency1, frequency2, sequence):
    fsk_product1 = np.zeros(1000)
    fsk_product2 = np.zeros(1000)
    fsk_demodulated_signal1 = np.zeros(1000)
    fsk_demodulated_signal2 = np.zeros(1000)
    sequence_demodulated = np.zeros(1000)

    for i in range(0, len(sequence)):
        fsk_product1[i] = sequence[i] * sin(2 * pi * frequency1 * i / 1000)
        fsk_product2[i] = sequence[i] * sin(2 * pi * frequency2 * i / 1000)

    for i in range(1, 11):
        S1 = 0
        S2 = 0
        for t in range(0, 100):
            S1 = S1 + fsk_product1[t + 100 * (i - 1)]
            fsk_demodulated_signal1[t + 100 * (i - 1)] = S1
            S2 = S2 + fsk_product2[t + 100 * (i - 1)]
            fsk_demodulated_signal2[t + 100 * (i - 1)] = S2

    ask_demodulated = 0.5 * (np.sign(fsk_demodulated_signal1 - fsk_demodulated_signal2) + 1)

    for i in range(1, 11):
        for t in range(0, 100):
            sequence_demodulated[t + 100 * (i - 1)] = ask_demodulated[100 * i - 1]

    return fsk_demodulated_signal1, fsk_demodulated_signal2, sequence_demodulated

#Генерація шуму
def create_noise(mean, standard_deviation, length):
    return np.random.normal(mean, standard_deviation, length)

#Дослідження завадостійкості
def noise_stress(sequence, sequence_modulated, modulation, frequency):
    error_modulated = []
    for i in range(0, 20):
        p = 0
        for m in range(0, 200):
            noise = create_noise(0, 1, 1000)
            sequence_noise = sequence_modulated + i * noise
            if modulation == "ASK":
                _, sequence_demodulated = ask_demodulation(frequency[0], sequence_noise)
            elif modulation == "PSK":
                _, sequence_demodulated = psk_demodulation(frequency[0], sequence_noise)
            elif modulation == "FSK":
                _, _, sequence_demodulated = fsk_demodulation(frequency[0], frequency[1], sequence_noise)
            summa = abs(sum(sequence - sequence_demodulated))
            p += summa / 1000
        error_modulated.append(p / 200)
    return error_modulated


#Відображення результатів
def plot(x, y, axis_x="", axis_y="", title=""):
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(axis_x, fontsize=14)
    ax.set_ylabel(axis_y, fontsize=14)
    plt.title(title, fontsize=14)
    isdir = os.path.isdir('./figures/')
    if not isdir:
        os.mkdir('./figures/')
    fig.savefig('./figures/' + title + '.png', dpi=600)
    plt.close(fig)


#Основний код
def main(ask, psk, fsk1, fsk2):
    sequence = create_sequence()
    x = np.arange(len(sequence)) / 1000

    plot(x, sequence, axis_x="Час, c", axis_y="Амплітуда сигналу",
         title="Згенерована випадкова послідовність")

    sequence_ask = ask_modulation(ask, sequence)
    plot(x, sequence_ask, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Амплітудна модуляція")
    x_spectrum, spectrum_sequence_ask = spectrum(sequence_ask)
    plot(x_spectrum, spectrum_sequence_ask, axis_x="Частота, Гц",
         axis_y="Амплітуда спектру", title="Спектр при амплітудній модуляції")

    sequence_psk = psk_modulation(psk, sequence)
    plot(x, sequence_psk, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Фазова модуляція")
    x_spectrum, spectrum_sequence_psk = spectrum(sequence_psk)
    plot(x_spectrum, spectrum_sequence_psk, axis_x="Частота, Гц",
         axis_y="Амплітуда спектру", title="Спектр при фазовій модуляції")

    sequence_fsk = fsk_modulation(fsk1, fsk2, sequence)
    plot(x, sequence_fsk, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Частотна модуляція")
    x_spectrum, spectrum_sequence_fsk = spectrum(sequence_fsk)
    plot(x_spectrum, spectrum_sequence_fsk, axis_x="Частота, Гц",
         axis_y="Амплітуда спектру", title="Спектр при частотній модуляції")

    noise = create_noise(0, 1, 1000)
    sequence_ask_noise = sequence_ask + noise
    sequence_psk_noise = sequence_psk + noise
    sequence_fsk_noise = sequence_fsk + noise

    plot(x, sequence_ask_noise, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Амплітудна модуляція з шумом")
    plot(x, sequence_psk_noise, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Фазова модуляція з шумом")
    plot(x, sequence_fsk_noise, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Частотна модуляція з шумом")

    ask_demod_signal, ask_demod_seq = ask_demodulation(ask, sequence_ask_noise)
    plot(x, ask_demod_signal, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Демодульований сигнал з амплітудною модуляцією")
    plot(x, ask_demod_seq, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Демодульована послідовність з амплітудною модуляцією")

    psk_demod_signal, psk_demod_seq = psk_demodulation(psk, sequence_psk_noise)
    plot(x, psk_demod_signal, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Демодульований сигнал з фазовою модуляцією")
    plot(x, psk_demod_seq, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Демодульована послідовність з фазовою модуляцією")

    fsk_demod_signal1, fsk_demod_signal2, fsk_demod_seq = fsk_demodulation(fsk1, fsk2, sequence_fsk_noise)
    plot(x, fsk_demod_signal1, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Демодульований сигнал 1 з частотною модуляцією")
    plot(x, fsk_demod_signal2, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Демодульований сигнал 2 з частотною модуляцією")
    plot(x, fsk_demod_seq, axis_x="Час, c", axis_y="Амплітуда сигналу", title="Демодульована послідовність з частотною модуляцією")

    error_ask = noise_stress(sequence, sequence_ask, "ASK", [ask])
    error_psk = noise_stress(sequence, sequence_psk, "PSK", [psk])
    error_fsk = noise_stress(sequence, sequence_fsk, "FSK", [fsk1, fsk2])

    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(np.arange(0, 20), error_ask, linewidth=1)
    ax.plot(np.arange(0, 20), error_psk, linewidth=1)
    ax.plot(np.arange(0, 20), error_fsk, linewidth=1)
    ax.set_xlabel('Діапазон зміни шуму', fontsize=14)
    ax.set_ylabel('Ймовірність помилки', fontsize=14)
    ax.legend(['ASK', 'PSK', 'FSK'], loc=2)
    plt.title('Оцінка завадостійкості трьох видів модуляції', fontsize=14)
    isdir = os.path.isdir('./figures/')
    if not isdir:
        os.mkdir('./figures/')
    fig.savefig('./figures/Оцінка завадостійкості трьох видів модуляції.png', dpi=600)
    plt.close(fig)

if __name__ == "__main__":
    main(ask=40, psk=40, fsk1=60, fsk2=30) #значення частот за варіантом 10 з таблиці методичних матеріалів

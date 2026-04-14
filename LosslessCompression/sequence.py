#Виконала Усікова Віра, варіант 10, група 519-ст
import random
import collections
import math
import string
import matplotlib.pyplot as plt

#задавання параметрів сигналу відповідно до варіанту
surname = "Usikova" #прізвище латинськими літерами
group = "519" #цифри номера групи без букв ст
n_sequence = 100 #довжина кожної послідовності



#1 Генерація 8 тестових послідовностей
#тестова послідовність 1 - 100 елементів 1/0, з яких 10 (мій номер у списку групи) будуть одиницями
def generate_seq_1():
    seq = ['1'] * 10 + ['0'] * (n_sequence - 10)
    random.shuffle(seq)
    return ''.join(seq)

#тестова послідовність 2 - 100 елементів, спочатку йдуть літери прізвища, решта елементів заповнена нулями
def generate_seq_2():
    surname_letters = list(surname)
    zeros = ['0'] * (n_sequence - len(surname_letters))
    return ''.join(surname_letters + zeros)

#тестова послідовність 3 - 100 елементів, заповнена рандомно, але кількість букв не перевищує довжини прізвища
def generate_seq_3():
    surname_letters = list(surname)
    zeros = ['0'] * (n_sequence - len(surname_letters))
    seq = surname_letters + zeros
    random.shuffle(seq)
    return ''.join(seq)

#тестова послідовність 4 - 100 елементів, чергування прізвища та групи до моменту поки послідовність не заповниться
def generate_seq_4():
    alphabet = list(surname) + list(group)
    repeat = n_sequence // len(alphabet)
    remainder = n_sequence % len(alphabet)
    seq = alphabet * repeat + alphabet[:remainder]
    return ''.join(seq)

#тестова послідовність 5 - 100 елементів, перші дві літери прізвища та номер групи, ймовірніст появи елементів = 0,2
def generate_seq_5():
    first_two = surname[:2]
    alphabet = list(first_two) + list(group)
    counts = [20] * len(alphabet)
    seq = []
    for sym, cnt in zip(alphabet, counts):
        seq.extend([sym] * cnt)
    random.shuffle(seq)
    return ''.join(seq)

#тестова послідовність 6 - 100 елементів, перші дві літери прізвища та номер групи
#ймовірніст появи букв = 0,7, а цифр = 0,3
def generate_seq_6():
    letters = list(surname[:2])
    digits = list(group)
    n_letters = int(0.7 * n_sequence)
    n_digits = n_sequence - n_letters
    seq = []
    for _ in range(n_letters):
        seq.append(random.choice(letters))
    for _ in range(n_digits):
        seq.append(random.choice(digits))
    random.shuffle(seq)
    return ''.join(seq)

#тестова послідовність 7 - 100 елементів, символи англійського алфавіту та цифри 0-9, заповнення випадкове
def generate_seq_7():
    alphabet = string.ascii_lowercase + string.digits
    seq = [random.choice(alphabet) for _ in range(n_sequence)]
    return ''.join(seq)

#тестова послідовність 8 - 100 елементів, всі елементи одиниці
def generate_seq_8():
    return '1' * n_sequence

#2 Перевірка відповідності характеристик згенерованих послідовностей
#обчислення ймовірності, ентропію, надмірність, тип розподілу
def analyze_seq(seq, seq_name):
    counts = collections.Counter(seq)
    prob = {sym: count / n_sequence for sym, count in counts.items()}
    entropy = -sum(p * math.log2(p) for p in prob.values())
    alphabet_size = len(prob)
    if alphabet_size > 1:
        source_excess = 1 - entropy / math.log2(alphabet_size)
    else:
        source_excess = 1.0

    #визначення типу розподілу, рівний/нерівний
    mean_prob = sum(prob.values()) / alphabet_size
    is_equal = all(abs(p - mean_prob) < 0.05 * mean_prob for p in prob.values())
    uniform = "рівна" if is_equal else "нерівна"

    return prob, alphabet_size, entropy, source_excess, uniform

def main():
    #Генерація всіх послідовностей
    sequences = [
        generate_seq_1(),
        generate_seq_2(),
        generate_seq_3(),
        generate_seq_4(),
        generate_seq_5(),
        generate_seq_6(),
        generate_seq_7(),
        generate_seq_8()
    ]
    seq_names = [f"Послідовність {i+1}" for i in range(8)]

    #Збереження послідовностей у файл
    with open("all_results/sequence.txt", "w", encoding="utf-8") as f:
        for seq in sequences:
            f.write(seq + "\n")

    #Аналіз та запис результатів
    results = []
    with open("all_results/results_sequence.txt", "w", encoding="utf-8") as f:
        for i, (seq, name) in enumerate(zip(sequences, seq_names), start=1):
            prob, alph_size, entropy, excess, uniform = analyze_seq(seq, name)
            results.append((alph_size, round(entropy, 2), round(excess, 2), uniform))

            f.write(f"{name}:\n")
            f.write(f"Послідовність: {seq}\n")
            f.write(f"Розмір послідовності: {n_sequence} byte\n")
            f.write(f"Розмір алфавіту: {alph_size}\n")
            prob_str = ', '.join([f"{sym}={p:.4f}" for sym, p in prob.items()])
            f.write(f"Ймовірності появи символів: {prob_str}\n")
            mean_p = sum(prob.values()) / alph_size
            f.write(f"Середнє арифметичне ймовірностей: {mean_p:.2f}\n")
            f.write(f"Ймовірність розподілу символів: {uniform}\n")
            f.write(f"Ентропія: {entropy:.4f}\n")
            f.write(f"Надмірність джерела: {excess:.2f}\n\n")

    #Будування таблиці
    headers = ['Розмір алфавіту', 'Ентропія', 'Надмірність', 'Ймовірність']
    row_labels = [f"Послідовність {i+1}" for i in range(8)]
    fig, ax = plt.subplots(figsize=(14/1.54, 8/1.54))
    ax.axis('off')
    table = ax.table(cellText=results, colLabels=headers, rowLabels=row_labels, loc='center', cellLoc='center')
    table.set_fontsize(14)
    table.scale(0.8, 2)
    #При виконанні 5 практичної не дочитала як саме необхідно назвати файл, передивилась файл при виконанні 6 і виправила
    plt.savefig("all_results/Характеристики сформованих послідовностей.png", bbox_inches='tight', dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
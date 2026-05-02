#Виконала Усікова Віра, варіант 10, група 519-ст

import math
import collections
import matplotlib.pyplot as plt

#Функція для зчитування послідовностей з файлу, що був створений на попередніх роботах
def read_sequences(filepath="all_results/sequence.txt"):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

#Функція для обчислення ентропії Шеннона
def compute_entropy(seq):
    counts = collections.Counter(seq)
    probs = [count / len(seq) for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs)

#Функція для перетворення числа у двійковий рядок з заданою довжиною
def float_bin(point, size_cod):
    binary = ""
    for _ in range(size_cod):
        point *= 2
        if point > 1:
            binary += "1"
            point -= 1
        elif point < 1:
            binary += "0"
        else:
            binary += "1"
            break
    return binary

#Арифметичне кодування, кодування АС
def encode_ac(seq, alphabet, probs):
    intervals = []
    low = 0.0
    for p in probs:
        intervals.append((low, low + p))
        low += p

    cur_low, cur_high = 0.0, 1.0
    for ch in seq:
        for i, sym in enumerate(alphabet):
            if sym == ch:
                l, u = intervals[i]
                new_low = cur_low + (cur_high - cur_low) * l
                new_high = cur_low + (cur_high - cur_low) * u
                cur_low, cur_high = new_low, new_high
                break

    point = (cur_low + cur_high) / 2
    size_cod = math.ceil(math.log2(1 / (cur_high - cur_low))) + 1
    binary_code = float_bin(point, size_cod)
    return [point, len(alphabet), alphabet, probs], binary_code

#Функція для декодування АС
def decode_ac(encoded_data, seq_len):
    point, alpha_size, alphabet, probs = encoded_data
    intervals = []
    low = 0.0
    for p in probs:
        intervals.append((low, low + p))
        low += p

    cur_low, cur_high = 0.0, 1.0
    decoded = ""

    for _ in range(seq_len):
        scaled = (point - cur_low) / (cur_high - cur_low)
        for i, (l, u) in enumerate(intervals):
            if l <= scaled < u:
                sym = alphabet[i]
                decoded += sym
                new_low = cur_low + (cur_high - cur_low) * l
                new_high = cur_low + (cur_high - cur_low) * u
                cur_low, cur_high = new_low, new_high
                break
    return decoded

#Кодування Хаффмана, HC
def encode_hc(seq, alphabet, probs):
    if len(alphabet) == 1:
        code = "0" * len(seq)
        symbol_codes = [[alphabet[0], "0"]]
        return code, symbol_codes

    nodes = [[sym, p] for sym, p in zip(alphabet, probs)]
    nodes.sort(key=lambda x: x[1])
    tree = []

    while len(nodes) > 1:
        left = nodes.pop(0)
        right = nodes.pop(0)
        total_prob = left[1] + right[1]
        tree.append([left[0], right[0]])
        nodes.append([left[0] + right[0], total_prob])
        nodes.sort(key=lambda x: x[1])

    tree.reverse()
    symbol_codes = []
    for sym in sorted(alphabet):
        code = ""
        for node in tree:
            if sym in node[0]:
                code += "0"
                if sym == node[0]:
                    break
            else:
                code += "1"
                if sym == node[1]:
                    break
        symbol_codes.append([sym, code])

    encoded = "".join(next(code for s, code in symbol_codes if s == ch) for ch in seq)
    return encoded, symbol_codes

#Функція для декодування коду Хаффмана
def decode_hc(encoded_str, symbol_codes):
    code_map = {code: sym for sym, code in symbol_codes}
    decoded = ""
    buffer = ""
    for bit in encoded_str:
        buffer += bit
        if buffer in code_map:
            decoded += code_map[buffer]
            buffer = ""
    return decoded

#Головна функція для запуску всієї програми та всіх необхідних розрахунків
def main():
    #Зчитування послідовностей
    try:
        full_sequences = read_sequences("all_results/sequence.txt")
    except FileNotFoundError:
        return

    #Обмеження першими 10 символами
    sequences = [seq[:10] for seq in full_sequences]
    seq_names = [f"Послідовність {i+1}" for i in range(len(sequences))]

    #Створення файлу для результатів
    with open("all_results/results_AC_CH.txt", "w", encoding="utf-8") as f:
        f.write("Результати стистення методами арифметичного кодування АС та кодування Хаффмана СН\n\n")

        table_data = []

        for idx, seq in enumerate(sequences, start=1):
            f.write(f"Оригінальна послідовність {idx}: {seq}\n")
            seq_len = len(seq)
            unique = sorted(set(seq))
            counts = collections.Counter(seq)
            probs = [counts[ch] / seq_len for ch in unique]
            entropy = compute_entropy(seq)
            f.write(f"Ентропія: {entropy:.4f}\n")

            f.write("\nДані арифметичного кодування:\n")
            enc_data_ac, enc_bits_ac = encode_ac(seq, unique, probs)
            bps_ac = len(enc_bits_ac) / seq_len
            f.write(f"дані закодованої АС послідовності: {enc_data_ac}\n")
            f.write(f"закодована АС послідовність: {enc_bits_ac}\n")
            f.write(f"значення bps при кодуванні АС: {bps_ac:.2f}\n")
            decoded_ac = decode_ac(enc_data_ac, seq_len)
            f.write(f"декодована АС послідовність: {decoded_ac}\n")
            if decoded_ac != seq:
                f.write("декодована АС послідовність не збігається з оригіналом\n")


            f.write("\nДані кодування Хаффмана:\n")
            enc_bits_hc, symbol_codes = encode_hc(seq, unique, probs)
            bps_hc = len(enc_bits_hc) / seq_len
            f.write("Таблиця кодів:\n")
            f.write("Алфавіт   Код символу\n")
            for sym, code in symbol_codes:
                f.write(f"{sym}         {code}\n")
            f.write(f"дані закодованої СН послідовності: {[enc_bits_hc, symbol_codes]}\n")
            f.write(f"закодована СН послідовність: {enc_bits_hc}\n")
            f.write(f"значення bps при кодуванні СН: {bps_hc:.2f}\n")
            decoded_hc = decode_hc(enc_bits_hc, symbol_codes)
            f.write(f"декодована СН послідовність: {decoded_hc}\n")
            if decoded_hc != seq:
                f.write("декодована СН послідовність не збігається з оригіналом\n")
            table_data.append([round(entropy, 2), round(bps_ac, 2), round(bps_hc, 2)])

    #Побудова таблиці
    headers = ['Ентропія', 'bps АС', 'bps СН']
    row_labels = seq_names
    fig, ax = plt.subplots(figsize=(14/1.54, len(sequences)/1.54))
    ax.axis('off')
    table = ax.table(cellText=table_data, colLabels=headers, rowLabels=row_labels,
                     loc='center', cellLoc='center')
    table.set_fontsize(14)
    table.scale(0.8, 2)
    plt.savefig("all_results/Результати стиснення методами АС та СН.png", bbox_inches='tight', dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
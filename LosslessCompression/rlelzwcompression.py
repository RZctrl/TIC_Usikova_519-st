#Другий файл для виконання практичної роботи 6
#Також було внесено декілька корективів в перший файл - перейменовано рисунок і створено нову папку all_results
#у файлі до практичної написано, що все має бути лише в LosslessCompression, але в такому випадку всі файли змішуються
import math
import collections
import matplotlib.pyplot as plt


#Функції для роботи з результатами, що були отримані раніше
def read_seq(filepath): #зчитує послідовності з файлу, кожна на окремому рядку
    with open(filepath, 'r', encoding='utf-8') as f:
        seq = [line.strip() for line in f if line.strip()]
    return seq

def compute_entropy(seq): #Обчислює ентропію Шеннона
    counts = collections.Counter(seq)
    probs = [count / len(seq) for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    return entropy

#Функції для роботи RLE
def rle_encode(seq): #кодує послідовність методом RLE
    if not seq:
        return "", []
    encode_tuples = []
    current_char = seq[0]
    count = 1
    for i in range(1, len(seq)):
        if seq[i] == current_char:
            count += 1
        else:
            encode_tuples.append((current_char, count))
            current_char = seq[i]
            count = 1
    encode_tuples.append((current_char, count))
    encode_str = ''.join(f"{cnt}{ch}" for ch, cnt in encode_tuples)
    return encode_str, encode_tuples

def rle_decode(encode_tuples): #декодує послідовність зі списку кортежів
    return ''.join(ch * cnt for ch, cnt in encode_tuples)

#Функції для роботи LZW
def lzw_encode(seq, log_file=None): #кодує послідовність алгоритмом LZW
    dictionary = {chr(i): i for i in range(65536)}
    next_code = 65536
    current = ""
    result_codes = []
    total_bits = 0

    for ch in seq:
        new_str = current + ch
        if new_str in dictionary:
            current = new_str
        else:
            code = dictionary[current]
            result_codes.append(code)
            bits = 16 if code < 65536 else 17
            total_bits += bits
            if log_file:
                log_file.write(f"Code: {code}, Element: {current}, bits: {bits}\n")
            dictionary[new_str] = next_code
            next_code += 1
            current = ch

    if current:
        code = dictionary[current]
        result_codes.append(code)
        bits = 16 if code < 65536 else 17
        total_bits += bits
        if log_file:
            log_file.write(f"Code: {code}, Element: {current}, bits: {bits}\n")
    return result_codes, total_bits

def lzw_decode(codes): #декодує список кодів LZW
    dictionary = {i: chr(i) for i in range(65536)}
    next_code = 65536

    if not codes:
        return ""

    result = []
    prev = dictionary[codes[0]]
    result.append(prev)

    for code in codes[1:]:
        if code in dictionary:
            current = dictionary[code]
        elif code == next_code:
            current = prev + prev[0]
        else:
            raise ValueError(f"Невірний код LZW: {code}")
        result.append(current)
        dictionary[next_code] = prev + current[0]
        next_code += 1
        prev = current

    return ''.join(result)

#Основна функція для опрацювання всіх минулих функцій в цьому файлі
def main():
    seq_file = "all_results/sequence.txt" #підключення файлу з послідовностями, що створено на минулій роботі

    sequences = read_seq(seq_file)

    #Створення файлу для запису результатів цієї практичної
    with open("all_results/results_rle_lzw.txt", "w", encoding="utf-8") as f:
        f.write("Результати стиснення методом RLE та LZW")

        table_data = [] #таблиця для графіка

        for idx, seq in enumerate(sequences, start=1):
            f.write(f"\n\nПослідовність {idx}:\n")
            f.write(f"Оригінальна послідовність: {seq}\n")
            original_bits = len(seq) * 16
            f.write(f"Розмір оригінальної послідовності: {original_bits} bits\n")

            entropy = compute_entropy(seq) #Ентропія
            f.write(f"Ентропія: {entropy:.4f}\n")

            enc_str, enc_tuples = rle_encode(seq) #RLE
            encoded_bits_rle = len(enc_str) * 16
            if encoded_bits_rle < original_bits:
                cr_rle = round(original_bits / encoded_bits_rle, 2)
            else:
                cr_rle = '-'
            f.write(f"Кодування RLE:\n")
            f.write(f"Закодована RLE послідовність: {enc_str}\n")
            f.write(f"Розмір закодованої RLE послідовності: {encoded_bits_rle} bits\n")
            f.write(f"Коефіцієнт стиснення RLE: {cr_rle}\n")

            decoded_rle = rle_decode(enc_tuples) #Декодування RLE для перевірки
            f.write(f"Декодована RLE послідовність: {decoded_rle}\n")
            f.write(f"Розмір декодованої RLE послідовності: {len(decoded_rle) * 16} bits\n\n")
            if decoded_rle != seq:
                f.write("ПОМИЛКА: декодована послідовність не збігається з оригіналом\n\n")

            #LZW
            f.write(f"Кодування LZW:\n")
            codes_lzw, total_bits_lzw = lzw_encode(seq, log_file=f)
            if total_bits_lzw < original_bits:
                cr_lzw = round(original_bits / total_bits_lzw, 2)
            else:
                cr_lzw = '-'
            f.write(f"\n\nЗакодована LZW послідовність: {codes_lzw}\n")
            f.write(f"Розмір закодованої LZW послідовності: {total_bits_lzw} bits\n")
            f.write(f"Коефіцієнт стиснення LZW: {cr_lzw}\n")
            # Декодування LZW
            decoded_lzw = lzw_decode(codes_lzw)
            f.write(f"Декодована LZW послідовність: {decoded_lzw}\n")
            f.write(f"Розмір декодованої LZW послідовності: {len(decoded_lzw) * 16} bits\n")
            if decoded_lzw != seq:
                f.write("ПОМИЛКА: декодована LZW послідовність не збігається з оригіналом\n")


            table_data.append([
                round(entropy, 2),
                cr_rle if cr_rle != '-' else None,
                cr_lzw if cr_lzw != '-' else None
            ])

        headers = ['Ентропія', 'КС RLE', 'КС LZW']
        row_labels = [f'Послідовність {i+1}' for i in range(len(sequences))]

        table_display = []
        for row in table_data:
            table_display.append([
                row[0],
                row[1] if row[1] is not None else '-',
                row[2] if row[2] is not None else '-'
            ])

        #Побудова графіка
        fig, ax = plt.subplots(figsize=(14/1.54, len(sequences)/1.54))
        ax.axis('off')
        table = ax.table(cellText=table_display, colLabels=headers, rowLabels=row_labels, loc='center', cellLoc='center')
        table.set_fontsize(14)
        table.scale(0.8, 2)
        plt.savefig("all_results/Результати стиснення методами RLE та LZW.png", bbox_inches='tight', dpi=150)
        plt.close()

#Запуск всього коду
if __name__ == "__main__":
    main()
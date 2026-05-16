#Виконала Усікова Віра, група 519-ст

import random
import ast

chunk_length = 8
assert not chunk_length % 8

#Позиції контрольних бітів
check_bits = [i for i in range(1, chunk_length + 1) if not i & (i - 1)]

#Перетворення символів в бінарний формат
def getCharsToBin(chars):
    assert not len(chars) * 8 % chunk_length
    return ''.join([bin(ord(c))[2:].zfill(8) for c in chars])

#Поблоковий вивід бінарних даних
def getChunkIterator(text_bin, chunk_size=chunk_length):
    for i in range(0, len(text_bin), chunk_size):
        yield text_bin[i:i + chunk_size]

#Отримання інформації про контрольні біти з бінарного блоку даних при кодуванні
def getCheckBitsData(value_bin):
    check_bits_count_map = {k: 0 for k in check_bits}
    for index, value in enumerate(value_bin, 1):
        if int(value):
            bin_char_list = list(bin(index)[2:].zfill(8))
            bin_char_list.reverse()
            for degree in [2 ** int(i) for i, v in enumerate(bin_char_list) if int(v)]:
                if degree in check_bits_count_map:
                    check_bits_count_map[degree] += 1
    check_bits_value_map = {}
    for check_bit, count in check_bits_count_map.items():
        check_bits_value_map[check_bit] = 0 if not count % 2 else 1
    return check_bits_value_map

#Додавання порожніх контрольних біт в бінарні дані
def getSetEmptyCheckBits(value_bin):
    for bit in check_bits:
        value_bin = value_bin[:bit - 1] + '0' + value_bin[bit - 1:]
    return value_bin

#Встановлення значень контрольних біт
def getSetCheckBits(value_bin):
    value_bin = getSetEmptyCheckBits(value_bin)
    check_bits_data = getCheckBitsData(value_bin)
    for check_bit, bit_value in check_bits_data.items():
        value_bin = value_bin[:check_bit - 1] + str(bit_value) + value_bin[check_bit:]
    return value_bin

#Отримання інформації про контрольні біти з бінарного блоку даних при декодуванні
def getCheckBits(value_bin):
    result = {}
    for idx, val in enumerate(value_bin, 1):
        if idx in check_bits:
            result[idx] = int(val)
    return result

#Видалення контрольних біт
def getExcludeCheckBits(value_bin):
    clean_value_bin = ""
    for index, char_bin in enumerate(list(value_bin), 1):
        if index not in check_bits:
            clean_value_bin += char_bin
    return clean_value_bin

#Додавання помилки до бінарної послідовності
def getSetErrors(encoded):
    result = ""
    for chunk in getChunkIterator(encoded, chunk_length + len(check_bits)):
        num_bit = random.randint(1, len(chunk))
        new_bit = '1' if chunk[num_bit - 1] == '0' else '0'
        result += chunk[:num_bit - 1] + new_bit + chunk[num_bit:]
    return result

#Пошук та виправлення помилок при передачі
def getCheckAndFixError(encoded_chunk):
    check_bits_encoded = getCheckBits(encoded_chunk)
    check_item = getExcludeCheckBits(encoded_chunk)
    check_item = getSetCheckBits(check_item)
    check_bits = getCheckBits(check_item)

    if check_bits_encoded != check_bits:
        invalid_bits = []
        for check_bit_encoded, value in check_bits_encoded.items():
            if check_bits[check_bit_encoded] != value:
                invalid_bits.append(check_bit_encoded)
        num_bit = sum(invalid_bits)
        new_bit = '1' if encoded_chunk[num_bit - 1] == '0' else '0'
        encoded_chunk = encoded_chunk[:num_bit - 1] + new_bit + encoded_chunk[num_bit:]
    return encoded_chunk

#Список індексів позицій помилок
def getDiffIndexList(value_bin1, value_bin2):
    diff_index_list = []
    for index, (b1, b2) in enumerate(zip(list(value_bin1), list(value_bin2)), 1):
        if b1 != b2:
            diff_index_list.append(index)
    return diff_index_list

#Кодування даних
def encode(source):
    text_bin = getCharsToBin(source)
    result = ""
    for chunk_bin in getChunkIterator(text_bin):
        chunk_bin = getSetCheckBits(chunk_bin)
        result += chunk_bin
    return text_bin, result

#Декодування даних
def decode(encoded, fix_errors=True):
    fixed_encoded_list = []
    chunk_len = chunk_length + len(check_bits)
    for encoded_chunk in getChunkIterator(encoded, chunk_len):
        if fix_errors:
            encoded_chunk = getCheckAndFixError(encoded_chunk)
        fixed_encoded_list.append(encoded_chunk)

    clean_chunk_list = []
    for encoded_chunk in fixed_encoded_list:
        clean_chunk = getExcludeCheckBits(encoded_chunk)
        clean_chunk_list.append(clean_chunk)

    decoded_value = ""
    for clean_chunk in clean_chunk_list:
        for i in range(0, len(clean_chunk), 8):
            byte_bits = clean_chunk[i:i+8]
            decoded_value += chr(int(byte_bits, 2))
    return decoded_value

#Основний код, збереження результатів в файл
if __name__ == '__main__':
    with open("sequence.txt", "r", encoding="utf-8") as file:
        original_sequences = [line.strip() for line in file if line.strip()]
    original_sequences = [seq.strip("[]'\"") for seq in original_sequences]

    with open("results_hamming.txt", "w", encoding="utf-8") as out_file:
        for idx, sequence in enumerate(original_sequences):
            source = sequence[:10]

            source_bin, encoded = encode(source)
            decoded = decode(encoded)

            encoded_with_error = getSetErrors(encoded)
            diff_index_list = getDiffIndexList(encoded, encoded_with_error)
            decoded_with_error = decode(encoded_with_error, fix_errors=False)
            decoded_without_error = decode(encoded_with_error)

            out_file.write(f"\nОригінальна послідовність: {source}\n")
            out_file.write(f"Оригінальна послідовність в бітах: {source_bin}\n")
            out_file.write(f"Розмір оригінальної послідовності: {len(source_bin)} bits\n")
            out_file.write(f"Довжина блоку кодування: {chunk_length}\n")
            out_file.write(f"Позиція контрольних біт: {check_bits}\n")
            out_file.write(f"Відносна надмірність коду: {len(check_bits) / chunk_length}\n\n")
            out_file.write("Кодування\n")
            out_file.write(f"Закодовані дані: {encoded}\n")
            out_file.write(f"Розмір закодованих даних: {len(encoded)} bits\n")
            out_file.write("Декодування\n\n")
            out_file.write(f"Декодовані дані: {decoded}\n")
            out_file.write(f"Розмір декодованих даних: {len(decoded) * 8} bits\n")
            out_file.write("Внесення помилки\n\n")
            out_file.write(f"Закодовані дані з помилками: {encoded_with_error}\n")
            out_file.write(f"Кількість помилок: {len(diff_index_list)}\n")
            out_file.write(f"Індекси помилок: {diff_index_list}\n")
            out_file.write(f"Декодовані дані без виправлення помилки: {decoded_with_error}\n")
            out_file.write("Виправлення помилки\n\n")
            out_file.write(f"Декодовані дані з виправленням помилки: {decoded_without_error}\n")
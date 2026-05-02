#Виконала Усікова Віра, варіант 10, група 519-ст

import os
from huffman import HuffmanTree
import math
import numpy as np
from scipy import fftpack
from PIL import Image


#Функція для кодування jpeg
def bits_require(n):
    n = abs(n)
    res = 0
    while n > 0:
        n >>= 1
        res += 1
    return res

#Функція для зглажування списків
def flat(lst):
    return [item for sublist in lst for item in sublist]

#Функція для перегортання бітів від'ємних чисел
def bin_flip(binstr):
    if not set(binstr).issubset('01'):
        raise ValueError
    return ''.join('0' if c == '1' else '1' for c in binstr)

#Функція перетворює число на двійковий рядок заданого розміру
def uint_to_bin(number, size):
    return bin(number)[2:][-size:].zfill(size)

#Функція перетворює число на двійковий рядок за правилами jpeg
def int_to_bin(n):
    if n == 0:
        return ''
    binstr = bin(abs(n))[2:]
    return binstr if n > 0 else bin_flip(binstr)

#Функція для генерування зигзаг індексів для матриці
def zigzag_point(rows, cols):
    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

    def move(direction, point):
        r, c = point
        moves = {
            UP: (r - 1, c),
            DOWN: (r + 1, c),
            LEFT: (r, c - 1),
            RIGHT: (r, c + 1),
            UP_RIGHT: (r - 1, c + 1),
            DOWN_LEFT: (r + 1, c - 1)}
        return moves[direction]

    def inbounds(point):
        r, c = point
        return 0 <= r < rows and 0 <= c < cols

    point = (0, 0)
    move_up = True

    for _ in range(rows * cols):
        yield point
        if move_up:
            if inbounds(move(UP_RIGHT, point)):
                point = move(UP_RIGHT, point)
            else:
                move_up = False
                if inbounds(move(RIGHT, point)):
                    point = move(RIGHT, point)
                else:
                    point = move(DOWN, point)
        else:
            if inbounds(move(DOWN_LEFT, point)):
                point = move(DOWN_LEFT, point)
            else:
                move_up = True
                if inbounds(move(DOWN, point)):
                    point = move(DOWN, point)
                else:
                    point = move(RIGHT, point)


def block_to_zigzag(block):
    return np.array([block[point] for point in zigzag_point(block.shape[0], block.shape[1])])

def dct_2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')

#2 версії таблиць квантування
def load_table(component, ver=1):
    if component == 'lum':
        if ver == 1:
            return np.array([
                [2, 2, 2, 2, 3, 4, 5, 6],
                [2, 2, 2, 2, 3, 4, 5, 6],
                [2, 2, 2, 2, 4, 5, 7, 9],
                [2, 2, 2, 4, 5, 7, 9, 12],
                [3, 3, 4, 5, 8, 10, 12, 12],
                [4, 4, 5, 7, 10, 12, 12, 12],
                [5, 5, 7, 9, 12, 12, 12, 12],
                [6, 6, 9, 12, 12, 12, 12, 12]
            ], dtype=np.float32)
        else:
            return np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ], dtype=np.float32)
    else:
        if ver == 1:
            return np.array([
                [3, 3, 5, 9, 13, 15, 15, 15],
                [3, 4, 6, 11, 14, 12, 12, 12],
                [5, 6, 9, 14, 12, 12, 12, 12],
                [9, 11, 14, 12, 12, 12, 12, 12],
                [13, 14, 12, 12, 12, 12, 12, 12],
                [15, 12, 12, 12, 12, 12, 12, 12],
                [15, 12, 12, 12, 12, 12, 12, 12],
                [15, 12, 12, 12, 12, 12, 12, 12]
            ], dtype=np.float32)
        else:
            return np.array([
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]
            ], dtype=np.float32)

#Функція для квантування
def quant(block, component, quant_version):
    q = load_table(component, quant_version)
    return (block / q).round().astype(np.int32)

#Функція для кодування коєфіцієнтів за довжиною рядка
def run_encode(arr):
    nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            nonzero = i
    symbol = []
    value = []
    run_length = 0
    for i, elem in enumerate(arr):
        if i > nonzero:
            symbol.append((0, 0))
            value.append(int_to_bin(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_require(elem)
            symbol.append((run_length, size))
            value.append(int_to_bin(elem))
            run_length = 0
    return symbol, value

#Функція для запису закодованої інформації у файл
def write_to_file(filepath, dc, ac, blocks_count, tables, quant_version):
    with open(filepath, 'w') as f:
        for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
            table = tables[table_name]
            f.write(uint_to_bin(len(table), 16))
            for key, value in table.items():
                if table_name in {'dc_y', 'dc_c'}:
                    f.write(uint_to_bin(key, 4))
                    f.write(uint_to_bin(len(value), 4))
                    f.write(value)
                else:
                    f.write(uint_to_bin(key[0], 4))
                    f.write(uint_to_bin(key[1], 4))
                    f.write(uint_to_bin(len(value), 8))
                    f.write(value)

        f.write(uint_to_bin(blocks_count, 32))

        for b in range(blocks_count):
            for c in range(3):
                category = bits_require(dc[b, c])
                symbol, value = run_encode(ac[b, :, c])
                dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
                ac_table = tables['ac_y'] if c == 0 else tables['ac_c']

                f.write(dc_table[category])
                f.write(int_to_bin(dc[b, c]))
                for i in range(len(symbol)):
                    f.write(ac_table[tuple(symbol[i])])
                    f.write(value[i])




#Функція кодування JPEG
def encode(input_file, output_file, quant_version):
    image = Image.open(input_file).convert('YCbCr')
    npmat = np.array(image, dtype=np.uint8)
    rows, cols = npmat.shape[0], npmat.shape[1]

    if rows % 8 != 0 or cols % 8 != 0:
        raise ValueError

    blocks_count = (rows // 8) * (cols // 8)
    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

    block_index = 0
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            for k in range(3):
                block = npmat[i:i+8, j:j+8, k].astype(np.float32) - 128
                dct_matrix = dct_2d(block)
                quant_matrix = quant(dct_matrix, 'lum' if k == 0 else 'chrom', quant_version)
                zigzag = block_to_zigzag(quant_matrix)
                dc[block_index, k] = zigzag[0]
                ac[block_index, :, k] = zigzag[1:]
            block_index += 1

    H_DC_Y = HuffmanTree(np.vectorize(bits_require)(dc[:, 0]))
    H_DC_C = HuffmanTree(np.vectorize(bits_require)(dc[:, 1:].flat))

    ac_y_symbol = flat(run_encode(ac[i, :, 0])[0] for i in range(blocks_count))
    H_AC_Y = HuffmanTree(ac_y_symbol)
    ac_c_symbol = flat(run_encode(ac[i, :, j])[0] for i in range(blocks_count) for j in [1, 2])
    H_AC_C = HuffmanTree(ac_c_symbol)

    tables = {
        'dc_y': H_DC_Y.value_to_bitstring_table(),
        'ac_y': H_AC_Y.value_to_bitstring_table(),
        'dc_c': H_DC_C.value_to_bitstring_table(),
        'ac_c': H_AC_C.value_to_bitstring_table()
    }

    write_to_file(output_file, dc, ac, blocks_count, tables, quant_version)



#Функції для декодування
class JPEGFileReader:
    TABLE_SIZE_BITS = 16
    BLOCKS_COUNT_BITS = 32
    DC_CODE_LENGTH_BITS = 4
    CATEGORY_BITS = 4
    AC_CODE_LENGTH_BITS = 8
    RUN_LENGTH_BITS = 4
    SIZE_BITS = 4

    def __init__(self, filepath):
        self.__file = open(filepath, 'r')
        self.__buffer = ''

    def __read_str(self, length):
        while len(self.__buffer) < length:
            ch = self.__file.read(1)
            if not ch:
                raise EOFError
            self.__buffer += ch
        data = self.__buffer[:length]
        self.__buffer = self.__buffer[length:]
        return data

    def __read_uint(self, size):
        bin_str = self.__read_str(size)
        return int(bin_str, 2)

    def __int2(self, bin_num):
        return int(bin_num, 2)

    def read_int(self, size):
        if size == 0:
            return 0
        bin_num = self.__read_str(size)
        if bin_num[0] == '1':
            return self.__int2(bin_num)
        else:
            return self.__int2(bin_flip(bin_num)) * -1

    def read_dc_table(self):
        table = {}
        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for _ in range(table_size):
            category = self.__read_uint(self.CATEGORY_BITS)
            code_len = self.__read_uint(self.DC_CODE_LENGTH_BITS)
            code = self.__read_str(code_len)
            table[code] = category
        return table

    def read_ac_table(self):
        table = {}
        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for _ in range(table_size):
            run_length = self.__read_uint(self.RUN_LENGTH_BITS)
            size = self.__read_uint(self.SIZE_BITS)
            code_len = self.__read_uint(self.AC_CODE_LENGTH_BITS)
            code = self.__read_str(code_len)
            table[code] = (run_length, size)
        return table

    def read_blocks_count(self):
        return self.__read_uint(self.BLOCKS_COUNT_BITS)

    def read_huffman_code(self, table):
        code = ''
        while code not in table:
            code += self.__read_str(1)
        return table[code]

#Функцыя для зчитування закодований файлу та відновлення масивів
def read_image_file(filepath, quant_version):
    reader = JPEGFileReader(filepath)
    tables = {}
    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
        if 'dc' in table_name:
            tables[table_name] = reader.read_dc_table()
        else:
            tables[table_name] = reader.read_ac_table()

    blocks_count = reader.read_blocks_count()
    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

    for block_index in range(blocks_count):
        for component in range(3):
            dc_table = tables['dc_y'] if component == 0 else tables['dc_c']
            ac_table = tables['ac_y'] if component == 0 else tables['ac_c']

            category = reader.read_huffman_code(dc_table)
            dc[block_index, component] = reader.read_int(category)

            cells_count = 0
            while cells_count < 63:
                run_length, size = reader.read_huffman_code(ac_table)
                if (run_length, size) == (0, 0):
                    while cells_count < 63:
                        ac[block_index, cells_count, component] = 0
                        cells_count += 1
                else:
                    for _ in range(run_length):
                        ac[block_index, cells_count, component] = 0
                        cells_count += 1
                    if size == 0:
                        ac[block_index, cells_count, component] = 0
                    else:
                        val = reader.read_int(size)
                        ac[block_index, cells_count, component] = val
                    cells_count += 1
    return dc, ac, blocks_count

#Функція для перетворення зигзагів назад у блоки
def zigzag_to_block(zigzag):
    n = int(math.sqrt(len(zigzag)))
    if n * n != len(zigzag):
        raise ValueError
    block = np.empty((n, n), dtype=np.int32)
    for i, point in enumerate(zigzag_point(n, n)):
        block[point] = zigzag[i]
    return block

#Функція для зворотнього квантування
def dequantize(block, component, quant_version):
    q = load_table(component, quant_version)
    return block * q

def idct_2d(image):
    return fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')

#Функція для декодування jpeg та збереження зображення
def decoder(encoded_file, output_image_path, quant_version):
    dc, ac, blocks_count = read_image_file(encoded_file, quant_version)

    block_side = 8
    blocks_per_side = int(math.isqrt(blocks_count))
    image_side = blocks_per_side * block_side
    if blocks_per_side * blocks_per_side != blocks_count:
        pass

    npmat = np.empty((image_side, image_side, 3), dtype=np.uint8)

    block_index = 0
    for i in range(0, image_side, block_side):
        for j in range(0, image_side, block_side):
            for c in range(3):
                zigzag = np.empty(64, dtype=np.int32)
                zigzag[0] = dc[block_index, c]
                zigzag[1:] = ac[block_index, :, c]
                block_2d = zigzag_to_block(zigzag)
                dct_block = dequantize(block_2d, 'lum' if c == 0 else 'chrom', quant_version)
                idct_block = idct_2d(dct_block)
                npmat[i:i+8, j:j+8, c] = np.clip(idct_block + 128, 0, 255).astype(np.uint8)
            block_index += 1

    img = Image.fromarray(npmat, mode='YCbCr').convert('RGB')
    img.save(output_image_path)
    return img


def run_experiment(input_path, output_encoded, output_decoded, quant_version, results_file, texture_name):
    original_size = os.path.getsize(input_path)
    encode(input_path, output_encoded, quant_version)
    decoded_img = decoder(output_encoded, output_decoded, quant_version)
    compressed_size = os.path.getsize(output_encoded)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    w, h = decoded_img.size
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(f"Дані для {texture_name} зображення, таблиця квантування {quant_version}\n")
        f.write(f"Розмір вихідного файла: {original_size} байт\n")
        f.write(f"Розмір файла JPEG: {compressed_size} байт\n")
        f.write(f"Розмір зображення JPEG: {w}x{h}\n")
        f.write(f"Коефіцієнт стиснення= {ratio:.2f}\n\n")


if __name__ == "__main__":
    os.makedirs("Results", exist_ok=True)

    results_txt = "results_jpeg.txt"
    open(results_txt, 'w').close()

    image = {
        "слаботекстурного": "10_3.bmp",
        "середньотекстурного": "10_2.bmp",
        "сильнотекстурного": "10_1.bmp"
    }

    for tex_name, path in image.items():
        if not os.path.exists(path):
            print(f"{path} не знайдено")
            continue

    for quant_ver in [1, 2]:
        for tex_name, img_path in image.items():
            if not os.path.exists(img_path):
                continue
            enc_file = f"Results/encoded_{tex_name}_v{quant_ver}.asf"
            dec_file = f"Results/decoded_{tex_name}_v{quant_ver}.jpg"
            run_experiment(img_path, enc_file, dec_file, quant_ver, results_txt, tex_name)
#Виконала Усікова Віра, група 519-ст

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import random


#1 Зчитування двох сусідніх кадрів
def getFrames(filename, first_frame, second_frame):
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame - 1)
    res1, fr1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, second_frame - 1)
    res2, fr2 = cap.read()
    cap.release()
    if not res1 or not res2:
        raise IOError(f"Не вдалося зчитати кадри")
    return fr1, fr2


#2 Розрахунок кількості блоків зображення, повертає кількість блоків в висоту та ширину
def segmentImage(anchor, blockSize=16):
    h, w = anchor.shape[:2]
    hSegments = int(h / blockSize)
    wSegments = int(w / blockSize)
    return hSegments, wSegments

#3 Центр блоку
def getCenter(x, y, blockSize):
    return int(x + blockSize/2), int(y + blockSize/2)

#4 Область пошуку в опорному кадрі, повертає фрагмент кадру в якому будуть шукати
def getAnchorSearchArea(x, y, anchor, blockSize, searchArea=7):
    h, w = anchor.shape
    cx, cy = getCenter(x, y, blockSize)
    sx = max(0, cx - int(blockSize / 2) - searchArea)
    sy = max(0, cy - int(blockSize / 2) - searchArea)
    ex = min(sx + searchArea * 2 + blockSize, w)
    ey = min(sy + searchArea * 2 + blockSize, h)
    return anchor[sy:ey, sx:ex]


#5 Повернення блоку з області пошуку
def getBlockZone(p, aSearch, tBlock, blockSize):
    px, py = p
    px = px - int(blockSize / 2)
    py = py - int(blockSize / 2)
    px = max(0, px)
    py = max(0, py)
    aBlock = aSearch[py:py + blockSize, px:px + blockSize]
    if aBlock.shape != tBlock.shape:
        return np.zeros(tBlock.shape, dtype=tBlock.dtype)
    return aBlock


#6 Повернення суми абсолютних різниць між блоками
def getMAD(tBlock, aBlock):
    return np.sum(np.abs(tBlock.astype(np.int16) - aBlock.astype(np.int16)))


#7 Пошук збігу
def getBestMatch(targetBlock, anchorSearchArea, blockSize):
    step = 4
    ah, aw = anchorSearchArea.shape
    acy, acx = int(ah / 2), int(aw / 2)
    minMAD = float("+inf")
    minP = None

    while step >= 1:
        p1 = (acx, acy)
        p2 = (acx + step, acy)
        p3 = (acx, acy + step)
        p4 = (acx + step, acy + step)
        p5 = (acx - step, acy)
        p6 = (acx, acy - step)
        p7 = (acx - step, acy - step)
        p8 = (acx + step, acy - step)
        p9 = (acx - step, acy + step)
        pointList = [p1, p2, p3, p4, p5, p6, p7, p8, p9]

        for p in pointList:
            aBlock = getBlockZone(p, anchorSearchArea, targetBlock, blockSize)
            mad = getMAD(targetBlock, aBlock)
            if mad < minMAD:
                minMAD = mad
                minP = p
        step = int(step / 2)

    px, py = minP
    px = px - int(blockSize / 2)
    py = py - int(blockSize / 2)
    px = max(0, px)
    py = max(0, py)
    matchBlock = anchorSearchArea[py:py + blockSize, px:px + blockSize]
    return matchBlock

#8 Пошук блоків зміщення на другому кадрі
def blockSearchBody(anchor, target, blockSize, searchArea=7):
    anchor = anchor.astype(np.uint8)
    target = target.astype(np.uint8)
    h, w = anchor.shape
    hSegments, wSegments = segmentImage(anchor, blockSize)
    predicted = np.ones((h, w), dtype=np.uint8) * 255
    bcount = 0
    for y in range(0, int(hSegments * blockSize), blockSize):
        for x in range(0, int(wSegments * blockSize), blockSize):
            bcount += 1
            targetBlock = target[y:y + blockSize, x:x + blockSize]
            anchorSearchArea = getAnchorSearchArea(x, y, anchor, blockSize, searchArea)
            anchorBlock = getBestMatch(targetBlock, anchorSearchArea, blockSize)
            predicted[y:y + blockSize, x:x + blockSize] = anchorBlock
    assert bcount == int(hSegments * wSegments)
    return predicted


#9 Розрахунок залишкового кадру, повертає різницю між кадрами
def getResidual(target, predicted):
    return target.astype(np.int16) - predicted.astype(np.int16)


#10 Відновлення кадру (декодування)
def getReconstructTarget(residual, predicted):
    reconstructed = np.add(residual, predicted.astype(np.int16))
    return reconstructed.astype(np.uint8)


#11 Розрахунок кількості біт на піксель для кадру
def getBitsPerPixel(im):
    h, w = im.shape
    im_list = im.tolist()
    bits = 0
    for row in im_list:
        for pixel in row:
            bits += math.log2(abs(pixel) + 1)
    return bits / (h * w)


#Основна функція для запуску та виконання всієї програми
def main(anchorFrame, targetFrame, blockSize=16, searchArea=7, saveOutput=True):
    h, w, ch = anchorFrame.shape
    print(f"Розмір кадру: {h}x{w}, каналів: {ch}")

    diffFrameRGB = np.zeros((h, w, ch), dtype=np.uint8)
    predictedFrameRGB = np.zeros((h, w, ch), dtype=np.uint8)
    residualFrameRGB = np.zeros((h, w, ch), dtype=np.int16)
    restoreFrameRGB = np.zeros((h, w, ch), dtype=np.uint8)

    bitsAnchor = []
    bitsDiff = []
    bitsPredicted = []

    for i in range(ch):
        anchor_c = anchorFrame[:, :, i]
        target_c = targetFrame[:, :, i]

        diff_c = cv2.absdiff(anchor_c, target_c)

        predicted_c = blockSearchBody(anchor_c, target_c, blockSize, searchArea)
        residual_c = getResidual(target_c, predicted_c)
        reconstructed_c = getReconstructTarget(residual_c, predicted_c)

        bitsAnchor.append(getBitsPerPixel(anchor_c))
        bitsDiff.append(getBitsPerPixel(diff_c))
        bitsPredicted.append(getBitsPerPixel(residual_c))

        diffFrameRGB[:, :, i] = diff_c
        predictedFrameRGB[:, :, i] = predicted_c
        residualFrameRGB[:, :, i] = residual_c
        restoreFrameRGB[:, :, i] = reconstructed_c

    print("Біт/піксель для оригінального кадру\n")
    print(f"RGB: {sum(bitsAnchor):.3f}, R: {bitsAnchor[0]:.3f}, G: {bitsAnchor[1]:.3f}, B: {bitsAnchor[2]:.3f}\n")
    print("Біт/піксель для простої різниці кадрів\n")
    print(f"RGB: {sum(bitsDiff):.3f}, R: {bitsDiff[0]:.3f}, G: {bitsDiff[1]:.3f}, B: {bitsDiff[2]:.3f}\n")
    print("Біт/піксель для залишку після компенсації руху\n")
    print(f"RGB: {sum(bitsPredicted):.3f}, R: {bitsPredicted[0]:.3f}, G: {bitsPredicted[1]:.3f}, B: {bitsPredicted[2]:.3f}\n")

    outfile = "Results"
    if saveOutput:
        if not os.path.isdir(outfile):
            os.mkdir(outfile)
        cv2.imwrite(f"{outfile}/First frame.png", anchorFrame)
        cv2.imwrite(f"{outfile}/Second frame.png", targetFrame)
        cv2.imwrite(f"{outfile}/Difference between frame.png", diffFrameRGB)
        cv2.imwrite(f"{outfile}/Prediction frame.png", predictedFrameRGB)
        residual_vis = np.abs(residualFrameRGB).astype(np.uint8)
        cv2.imwrite(f"{outfile}/Residual frame.png", residual_vis)
        cv2.imwrite(f"{outfile}/Restore frame.png", restoreFrameRGB)

    barWidth = 0.25
    fig, ax = plt.subplots(figsize=(12, 8))

    P1 = [sum(bitsAnchor), bitsAnchor[0], bitsAnchor[1], bitsAnchor[2]]
    Diff = [sum(bitsDiff), bitsDiff[0], bitsDiff[1], bitsDiff[2]]
    Mpeg = [sum(bitsPredicted), bitsPredicted[0], bitsPredicted[1], bitsPredicted[2]]

    br1 = np.arange(len(P1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    ax.bar(br1, P1, color='r', width=barWidth, edgecolor='grey', label='Розмір для початкового кадру')
    ax.bar(br2, Diff, color='g', width=barWidth, edgecolor='grey', label='Розмір для різниці між кадрами')
    ax.bar(br3, Mpeg, color='b', width=barWidth, edgecolor='grey', label='Розмір для різниці з компенсацією рухів')

    ax.set_xlabel('Компоненти', fontweight='bold', fontsize=15)
    ax.set_ylabel('Біт на піксель', fontweight='bold', fontsize=15)
    ax.set_xticks([r + barWidth for r in range(len(P1))],
                  ['Біт/Піксель RGB', 'Біт/Піксель R', 'Біт/Піксель G', 'Біт/Піксель B'])
    ax.legend()
    plt.savefig(f'{outfile}/Гістограма кількості біт на піксель.png', dpi=600)
    plt.close()

    return None



if __name__ == "__main__":
    random.seed(42)
    fr = random.randint(0, 3000)

    video_file = "sample4.avi"
    try:
        frame1, frame2 = getFrames(video_file, fr, fr + 1)
        main(frame1, frame2, blockSize=16, searchArea=7, saveOutput=True)
    except Exception as e:
        print(f"Помилка")
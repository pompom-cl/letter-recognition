import pandas as pd
import numpy as np

kolom = ['Huruf'] + list(range(784))

data = pd.read_csv('selected.csv', names=kolom)

from scipy.stats import pearsonr

def extract_features(img):
    img = (img > 0).astype(np.uint8)  # binary image

    coords = np.column_stack(np.where(img > 0))
    if coords.shape[0] == 0:
        return [0] * 17

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    onpix = np.count_nonzero(img)

    y_indices, x_indices = coords[:, 0], coords[:, 1]
    x_bar = np.mean(x_indices)
    y_bar = np.mean(y_indices)

    x2bar = np.var(x_indices)
    y2bar = np.var(y_indices)
    xybar = np.mean((x_indices - x_bar) * (y_indices - y_bar))
    x2ybr = np.mean((x_indices ** 2) * y_indices)
    xy2br = np.mean(x_indices * (y_indices ** 2))

    x_edge = np.mean(np.abs(np.diff(img, axis=1)).sum(axis=1))
    y_edge = np.mean(np.abs(np.diff(img, axis=0)).sum(axis=0))

    xegvy = pearsonr(np.abs(np.diff(img, axis=1)).sum(axis=1), np.arange(28))[0]
    yegvx = pearsonr(np.abs(np.diff(img, axis=0)).sum(axis=0), np.arange(28))[0]

    return [
        x_min, y_min, width, height, onpix,
        x_bar, y_bar, x2bar, y2bar, xybar,
        x2ybr, xy2br, x_edge, xegvy, y_edge, yegvx
    ]
feature_rows = []

for index, row in data.iterrows():
    pixels = row[1:].values  # ambil 784 pixel
    pixels = pixels.astype(np.uint8)
    image = pixels.reshape((28, 28))
    features = extract_features(image)
    feature_rows.append([row["Huruf"]] + features)  # simpan label + fitur

kolom_fitur = ['Huruf', 'x_min', 'y_min', 'width', 'height', 'onpix',
               'x_bar', 'y_bar', 'x2bar', 'y2bar', 'xybar',
               'x2ybr', 'xy2br', 'x_edge', 'xegvy', 'y_edge', 'yegvx']

fitur_df = pd.DataFrame(feature_rows, columns=kolom_fitur)

fitur_df.to_csv('converted.csv', index=False)

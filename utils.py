import numpy as np


def calculate_adx(df, window=14):
    df['TR'] = np.maximum((df['High'] - df['Low']),
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                         np.maximum((df['High'] - df['High'].shift(1)), 0), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                         np.maximum((df['Low'].shift(1) - df['Low']), 0), 0)

    df['TR'] = df['TR'].rolling(window=window).sum()
    df['+DM'] = df['+DM'].rolling(window=window).sum()
    df['-DM'] = df['-DM'].rolling(window=window).sum()

    df['+DI'] = 100 * (df['+DM'] / df['TR'])
    df['-DI'] = 100 * (df['-DM'] / df['TR'])

    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / df[['+DI', '-DI']].sum(axis=1)
    df['ADX'] = df['DX'].rolling(window=window).mean()

    return df['ADX']


def calculate_vwma(df, window):
    price_volume = df['Close'] * df['Volume']
    volume_sum = df['Volume'].rolling(window=window).sum()
    vwma = price_volume.rolling(window=window).sum() / volume_sum
    return vwma


def calculate_stochastic(df, window=14):
    high_14 = df['High'].rolling(window=window).max()
    low_14 = df['Low'].rolling(window=window).min()
    df['Stochastic'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    return df['Stochastic']

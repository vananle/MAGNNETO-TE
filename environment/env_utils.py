import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def get_data_size(dataset):
    if 'abilene' in dataset:
        train_size = 1500
        test_size = 1500
    elif 'geant' in dataset:
        train_size = 500
        test_size = 500
    else:
        raise NotImplementedError
    return train_size, test_size


def data_split(base_dir, dataset):
    path = os.path.join(base_dir, dataset, f'{dataset}.npz')

    all_data = np.load(path)
    data = all_data['traffic_demands']

    if len(data.shape) > 2:
        data = np.reshape(data, newshape=(data.shape[0], -1))

    # calculate num node
    T, F = data.shape
    N = int(np.sqrt(F))
    # print('Data shape', data.shape)

    data[data <= 0] = 1e-4
    data[data == np.nan] = 1e-4
    # Train-test split
    if 'abilene' in dataset or 'geant' in dataset:
        train_size, test_size = get_data_size(dataset=dataset)
        total_steps = train_size + test_size
        data_traffic = data[:total_steps]

    else:
        total_steps = data.shape[0]
        data_traffic = data[:total_steps]
        train_size = int(total_steps * 0.5)
        test_size = total_steps - train_size

    # total_steps = 100 if data.shape[0] > 100 else data.shape[0]
    # data_traffic = data[:total_steps]
    # train_size = int(total_steps * 0.7)
    # test_size = total_steps - train_size

    train_size = int(train_size * 0.4)  # in case of not use all training data
    cs_data_size = train_size if train_size < 500 else 500

    train_df, test_df = data_traffic[0:train_size], data_traffic[-test_size:]  # total dataset
    cs_data = data_traffic[0:cs_data_size]

    sc = MinMaxScaler(copy=True)
    sc.fit(data_traffic)

    train_scaled = sc.transform(train_df)
    test_scaled = sc.transform(test_df)
    cs_data_scaled = sc.transform(cs_data)

    # Converting the time series to samples
    n_node = N
    train_df = np.reshape(train_df, newshape=(train_df.shape[0], n_node, n_node))
    test_df = np.reshape(test_df, newshape=(test_df.shape[0], n_node, n_node))
    cs_data = np.reshape(cs_data, newshape=(cs_data.shape[0], n_node, n_node))
    train_scaled = np.reshape(train_scaled, newshape=(train_scaled.shape[0], n_node, n_node))
    test_scaled = np.reshape(test_scaled, newshape=(test_scaled.shape[0], n_node, n_node))
    cs_data_scaled = np.reshape(cs_data_scaled, newshape=(cs_data_scaled.shape[0], n_node, n_node))

    return_data = {
        'train/scaled': train_scaled,
        'test/scaled': test_scaled,
        'cs_data/scaled': cs_data_scaled,
        'scaler': sc,
        'train/gt': train_df,
        'test/gt': test_df,
        'cs_data/gt': cs_data,
    }

    print('train data', train_df.shape)
    print('test data', test_df.shape)

    return return_data


def load_data(base_dir, dataset):
    # loading dataset
    data = data_split(base_dir, dataset)
    return data

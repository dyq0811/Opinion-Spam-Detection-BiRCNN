import numpy as np


def shuffle(data, label):
    """
    Shuffle data and label
    """
    assert len(data) == len(label)
    index_arr = np.arange(len(data))
    np.random.shuffle(index_arr)
    res_data, res_label = [], []
    for i in index_arr:
        res_data.append(data[i])
        res_label.append(label[i])
    return res_data, res_label

def get_set(data, label, test_index):
    """
    Separate the test and training set.
    """
    test_data, test_label, train_data, train_label = [], [], [], []
    for i in range(len(data)):
        if i in test_index:
            test_data.append(data[i])
            test_label.append(label[i])
        else:
            train_data.append(data[i])
            train_label.append(label[i])
    return test_data, test_label, train_data, train_label

#d = ['a', 'b', 'c', 'd', 'e']
#l = [1, 0, 0, 1, 0]
#print(shuffle(d, l))
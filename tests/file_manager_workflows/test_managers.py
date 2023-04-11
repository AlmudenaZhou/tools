import os

import joblib
import pandas as pd
import numpy as np
import pytest

from tools.file_manager_workflows.file_manager_modules import PickleManager, Manager


@pytest.fixture
def file_managers_input_data():
    dtypes_by_column = {'a': float, 'b': int, 'c': str}

    data = pd.DataFrame(np.random.normal(size=(100, 3)), columns=['a', 'b', 'c'])
    data = data.astype(dtypes_by_column)
    return data


def test_create_folder():
    file_path = os.path.join(os.path.dirname(__file__), 'test_files')

    if os.path.exists(file_path):
        os.rmdir(file_path)

    Manager.create_file_path(file_path)

    assert os.path.exists(file_path), 'Manager create_file_path did not make the folder.'
    os.rmdir(file_path)


def test_save_pickle_manager(file_managers_input_data):
    file_path = file_managers_input_data['file_path']
    data = file_managers_input_data['data']

    pickle_manager = PickleManager()
    pickle_manager.save(data, 'test_saved_data', file_path)

    complete_file_path = os.path.join(file_path, 'test_saved_data.pkl')
    loaded_data = joblib.load(complete_file_path)

    check_numeric_inputs = np.allclose(loaded_data.iloc[:, :2], data.iloc[:, :2])
    check_string_inputs = np.all(loaded_data.iloc[:, 2] == data.iloc[:, 2])

    assert check_numeric_inputs, 'The numeric part of the data saved and the original numeric data are not the same'
    assert check_string_inputs, 'The object part of the data saved and the original object data are not the same'

    os.remove(complete_file_path)

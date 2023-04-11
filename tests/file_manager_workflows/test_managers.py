import os

from tools.file_manager_workflows.file_manager_modules import Manager


def test_create_folder():
    file_path = os.path.join(os.path.dirname(__file__), 'test_files')

    if os.path.exists(file_path):
        os.rmdir(file_path)

    Manager.create_file_path(file_path)

    assert os.path.exists(file_path), 'Manager create_file_path did not make the folder.'
    os.rmdir(file_path)

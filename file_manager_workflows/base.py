import logging
import os
from typing import Optional

from tools.file_manager_workflows.file_manager_modules import YamlManager, Manager
from tools.misc import instance_class_from_module_and_name


class ManagerWorkflow:

    def __init__(self):
        self.manager_config_file = self.get_model_manager_config_file()
        self.manager_module = None

    @staticmethod
    def get_model_class_path_from_model(model):
        model_class_path = model.__class__.__str__(model.__class__)
        model_class_path = model_class_path.split("'")[1]
        return model_class_path

    @staticmethod
    def get_model_manager_config_file():
        yaml_manager = YamlManager()
        cwd = os.getcwd()
        config_file_path = os.path.join(cwd, 'tools', 'file_manager_workflows', 'config_files')
        config_file = yaml_manager.load('model_manager_mapping', config_file_path)
        return config_file

    @staticmethod
    def get_file_name_and_path_of_model_config():
        project_name = os.path.split(os.getcwd())[0]
        file_name = f'{project_name}_model_config'
        file_path = os.path.join('tools', 'file_manager_workflows', 'config_files')
        return file_name, file_path

    def change_manager_module(self, class_name):
        manager_name = self.manager_config_file[class_name]
        module_path = 'tools.file_manager_workflows.file_manager_modules'
        self.manager_module = instance_class_from_module_and_name(module_path, manager_name)


class SaveWorkflow(ManagerWorkflow):

    def __init__(self):
        super().__init__()
        self.model_config = {}

    def save_models(self,  models_to_save: dict, file_path: Optional[str] = None, save_separated=True,
                    append_file=True):
        """
        Saves the models included in the models_to_save in a pkl at file_path
        :param file_path: path where the models will be saved.
        If file_path is None it will be saved at the working directory
        :param models_to_save: dict with the model instance as value and the file_name as key. It can add a folder
        in the file_name as {folder_name}/{file_name}.
        :param save_separated: It only used by the custom wrapped models with an attribute self._models
        If True, it will save each model at a separated file. Else, it will save the model all together.
        :param append_file: If True, if the file already exists it will be appended. If False, the content will
        be replaced
        """
        for file_name, model in models_to_save.items():
            model_class_path = self.get_model_class_path_from_model(model)
            class_name = model_class_path.split('.')[-1]
            self.change_manager_module(class_name)
            file_name_path = os.path.split(file_name)
            if len(file_name_path) > 1:
                file_name = file_name_path[-1]
                file_path = os.path.join(file_path, *file_name_path[:-1])
            self._save_model(model, file_name, file_path, save_separated)
        self._save_config_file(append_file)

    def _save_model(self, model, file_name, file_path, save_separated):
        model_class_path = self.get_model_class_path_from_model(model)
        model_id = id(model)
        is_custom = '_models' in model.__dict__.keys()
        if save_separated:

            assert is_custom, 'If save_separated=True, the model needs to be custom.'

            for specific_model_name, model_to_save in model._models.items():
                complete_model_name = f'{file_name}__{specific_model_name}'
                self._add_model_to_model_config(model_class_path, complete_model_name, file_path, True, model_id,
                                                specific_model_name)
                self.manager_module.save(model_to_save, complete_model_name, file_path)

        else:
            self._add_model_to_model_config(model_class_path, file_name, file_path, is_custom, model_id)
            self.manager_module.save(model, file_name, file_path)

    def _add_model_to_model_config(self, model_class_path, file_name, file_path, is_custom, model_id, model_name=None):
        """
        It creates a file with a structure as follows:

        - file_path 1:
            - model_class_path
            - model_name
            - model_id
        .
        .
        .

        :param model_class_path: path of the model class used where it can be imported
        :param file_name: name that the model will use for the file name
        :param file_path: path where the file will be saved
        :param model_name: symbolic name given to the model
        :return:
        """
        complete_file_path = self.manager_module.get_complete_file_path(file_name, file_path)

        self.model_config[complete_file_path] = {
            'model_class_path': model_class_path,
            'model_name': model_name,
            'model_id': model_id,
            'custom': is_custom
        }
        logging.info(f'Added {model_class_path} at {complete_file_path} to config file')

    def _save_config_file(self, append_file=True):
        """
        It will save a file named {project_name}_model_config.yaml. It will be saved at
        tools/file_manager_workflows/config_files.

        :param append_file: If True, if the file already exists it will be appended. If False, the content will
        be replaced. By default is True
        """
        file_name, file_path = self.get_file_name_and_path_of_model_config()
        YamlManager().save(self.model_config, file_name, file_path, append_file)

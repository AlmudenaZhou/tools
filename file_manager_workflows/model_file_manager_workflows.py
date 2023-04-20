import logging
import os
from typing import Optional, Union

import pandas as pd

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
        config_file = yaml_manager.load(file_name='model_manager_mapping', file_path=config_file_path)
        return config_file

    @staticmethod
    def get_file_name_and_path_of_model_config():
        project_name = os.path.split(os.getcwd())[-1]
        file_name = f'{project_name.lower()}_model_config'
        cwd = os.getcwd()
        file_path = os.path.join(cwd, 'tools', 'file_manager_workflows', 'config_files')
        logging.info(f'Model config name: {file_name} \n Path model config: {file_path}')
        return file_name, file_path


class SaveWorkflow(ManagerWorkflow):

    def __init__(self):
        super().__init__()
        self.model_config = {}

    def change_manager_module(self, class_name, save_separated):
        if class_name in self.manager_config_file:
            manager_info = self.manager_config_file[class_name]

            manager_name = manager_info
            if isinstance(manager_info, dict):
                manager_name = manager_info['separated'] if save_separated else manager_info['no_separated']

            module_path = 'tools.file_manager_workflows.file_manager_modules'
            self.manager_module = instance_class_from_module_and_name(module_path, manager_name)
            logging.info(f'Manager module changed to: {manager_name}')
        else:
            self.manager_module = None
            logging.warning(f'The model {class_name} you are trying to save is not in the config file')

    def save_models(self,  models_to_save: list, file_path: Optional[str] = None, save_separated=True,
                    append_to_file=True):
        """
        Saves the models included in the models_to_save in a pkl at file_path
        :param file_path: path where the models will be saved.
        If file_path is None it will be saved at the working directory
        :param models_to_save: list in which each element is a dictionary with the model at the key "model" and the
        file_name at the key file_name or the model itself. It can add a folder in the file_name as
        {folder_name}/{file_name}. Also, metadata can be added as extra parameters of the model.
        If there is no "file_name", it will be considered as the key the __name__ of the model.
        You can have models alone and dictionaries mixed in the list
        :param save_separated: It only used by the custom wrapped models with an attribute self._models
        If True, it will save each model at a separated file. Else, it will save the model all together.
        :param append_to_file: If True, if the file already exists it will be appended. If False, the content will
        be replaced
        """

        for model_info in models_to_save:
            if isinstance(model_info, dict):
                model = model_info['model']
            else:
                model = model_info
                model_info = {}
            file_name = model_info.get('file_name', self.get_model_class_path_from_model(model).split('.')[-1])
            logging.info(f'Saving {file_name}')
            model_class_path = self.get_model_class_path_from_model(model)
            class_name = model_class_path.split('.')[-1]
            self.change_manager_module(class_name, save_separated)
            file_name_path = os.path.split(file_name)
            if len(file_name_path) > 1:
                file_name = file_name_path[-1]
                file_path = os.path.join(file_path, *file_name_path[:-1])
            if self.manager_module is not None:
                filtered_model_info = {key: model_info[key] for key in model_info.keys()
                                       if key not in ['model', 'file_name']}
                self._save_model(model, file_name, file_path, save_separated, **filtered_model_info)
        self._save_config_file(append_to_file)

    def _save_model(self, model, file_name, file_path, save_separated, **kwargs):
        model_class_path = self.get_model_class_path_from_model(model)
        model_id = id(model)
        is_custom = '_models' in model.__dict__.keys()
        if save_separated:

            assert is_custom, 'If save_separated=True, the model needs to be custom.'

            file_path = os.path.join(file_path, file_name)

            assert hasattr(model, '_models'), 'If save_separated = True, the model needs the _models attribute'
            assert isinstance(model._models, dict), 'If save_separated = True, the model._models needs to be a dict'

            for specific_model_name, model_to_save in model._models.items():
                self._add_model_to_model_config(model_class_path, specific_model_name, file_path, True, model_id,
                                                specific_model_name, **kwargs)
                self.manager_module.save(model_to_save, raw_file_name=specific_model_name, file_path=file_path)

        else:
            self._add_model_to_model_config(model_class_path, file_name, file_path, is_custom, model_id, **kwargs)
            self.manager_module.save(model, raw_file_name=file_name, file_path=file_path)

    def _add_model_to_model_config(self, model_class_path, file_name, file_path, is_custom, model_id, model_name=None,
                                   **kwargs):
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

        mandatory_keys_at_save_file = self.model_config[complete_file_path].keys()

        if len(set(kwargs.keys()).intersection(set(mandatory_keys_at_save_file))) != 0:
            raise ValueError(f'Cannot use: {mandatory_keys_at_save_file} at the metadata keys of the model ')

        self.model_config[complete_file_path].update(kwargs)
        logging.info(f'Added {model_class_path} at {complete_file_path} to config file')

    def _save_config_file(self, append_to_file=True):
        """
        It will save a file named {project_name}_model_config.yaml. It will be saved at
        tools/file_manager_workflows/config_files.

        :param append_to_file: If True, if the file already exists it will be appended. If False, the content will
        be replaced. By default is True
        """
        file_name, file_path = self.get_file_name_and_path_of_model_config()
        YamlManager().save(self.model_config, raw_file_name=file_name, file_path=file_path,
                           append_to_file=append_to_file)


class LoadWorkflow(ManagerWorkflow):

    def __init__(self):
        super().__init__()

    @staticmethod
    def load_individual_model(model):
        model_class_path = LoadWorkflow.get_model_class_path_from_model(model)
        class_module_path = model_class_path.split('.')
        class_name = class_module_path[-1]
        module_path = class_module_path[:-1]
        model = instance_class_from_module_and_name(module_path, class_name)
        return model

    @staticmethod
    def _load_config_file():
        file_name, file_path = LoadWorkflow.get_file_name_and_path_of_model_config()
        model_config = YamlManager().load(file_name=file_name, file_path=file_path)
        return model_config

    def load_models(self):
        """
        Loads the models at the instance, specified in models_to_load that are in file_path
        :return: dictionary with the models loaded
        """
        model_config = self._load_config_file()
        model_config = pd.DataFrame(model_config).T

        loaded_models = {}

        for model_id in model_config['model_id'].unique():
            model_config_by_id = model_config[model_config['model_id'] == model_id]
            self._check_model_config_integrity_by_id(model_config_by_id)
            loaded_models[model_id] = self._load_model_by_id(model_config_by_id)

        return loaded_models

    @staticmethod
    def _load_model_by_id(model_config_by_id):
        model_names = model_config_by_id['model_name']
        model_class_path = model_config_by_id['model_class_path'].unique()[0]
        class_name = model_class_path.split('.')[-1]

        if None not in model_names:
            model_path = '.'.join(model_class_path.split('.')[:-1])
            model_instance = instance_class_from_module_and_name(model_path, class_name)
            model_instance._models = {model_name: Manager.load(complete_file_path=comp_file_path)
                                      for comp_file_path, model_name in zip(model_config_by_id.index, model_names)}
            return model_instance

        else:
            assert model_config_by_id.shape[0], 'If there is a None in model names, it can only have saved one model.'
            return Manager.load(complete_file_path=model_config_by_id.index[0])

    @staticmethod
    def _check_model_config_integrity_by_id(model_config_by_id):

        model_id = model_config_by_id["model_id"].unique()[0]
        msg_start = f'Review the model id: {model_id} in the configuration file. '

        n_unique_model_class_path = len(model_config_by_id['model_class_path'].unique())
        n_unique_custom = len(model_config_by_id['custom'].unique())
        model_class_custom_assert_msg = 'There are different values in model_class_path and custom for the same model'
        assert n_unique_model_class_path == n_unique_custom == 1, msg_start + model_class_custom_assert_msg

        model_names = model_config_by_id['model_name'].unique()
        if None in model_names:
            no_model_names_msg = 'If there is a None in model names, it can only have saved one model.'
            assert model_config_by_id.shape[0] == 1, msg_start + no_model_names_msg
        else:
            model_names_msg = 'All the model names must be different.'
            assert len(model_names) == model_config_by_id.shape[0], msg_start + model_names_msg

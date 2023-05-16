from sklearn.pipeline import Pipeline, FeatureUnion


def construct_pipeline_config(pipeline, name=None):
    if isinstance(pipeline, (Pipeline, FeatureUnion)):

        pipeline_attr_name_with_steps = {'Pipeline': 'steps', 'FeatureUnion': 'transformer_list'}

        config_type = pipeline.__class__.__name__
        config = {'type': config_type, 'name': name,
                  'steps': [construct_pipeline_config(new_pipeline, new_name)
                            for new_name, new_pipeline in getattr(pipeline,
                                                                  pipeline_attr_name_with_steps[config_type])]}
    else:

        if pipeline.saved_at_file is None:
            print(f'{name} not saved')

        config = {'name': name, 'type': pipeline.saved_at_file, 'params': None}

    return config

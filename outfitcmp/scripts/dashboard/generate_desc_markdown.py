"""
Description: Generate an experiment's description using markdown format
"""
import os
import markdown2

WORKING_DIR = os.path.dirname(__file__)
DESCRIPTION_TEMPLATE = os.path.join(WORKING_DIR, 'description_template.txt')

def decorate_char(char):
    prefix = '\\' if char in '\`*_{}[]()#+-.!' else ''
    return prefix + char

def decorated_format(data, *args):
    """ Add backslashes to skip markdown formatting """
    new_args = []
    for arg in args:
        new_args.append(''.join([decorate_char(char) for char in str(arg)]))
    return data.format(*new_args)

def generate_description_markdown(dashboard_config, experiment_config, experiment_dir):
    """ Generate an experiment's description """
    with open(DESCRIPTION_TEMPLATE) as md_file:
        data = md_file.read()
    data = decorated_format(
        data,
        experiment_config['experiment_name'],
        experiment_config['description'],
        experiment_config['data_dir'],
        'Regression' if experiment_config['is_regression'] else 'Classification',
        experiment_config['loss'],
        experiment_config['optimizer'],
        experiment_config['num_epoches'],
        experiment_config['batch_size']
    )
    html_data = markdown2.markdown(data)
    result_file = os.path.join(
        experiment_dir,
        dashboard_config['plots_dir'],
        dashboard_config['description_name']
    )
    with open(result_file, 'w+') as html_file:
        html_file.write(html_data)
"""
Description: Update information on dashboard webpage
"""
import os
import yaml
import markdown2

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..')
DASHBOARD_ROOT = os.path.join(ROOT_DIR, 'dashboard')
EXPERIMENTS_ROOT = os.path.join(ROOT_DIR, 'experiments')
DASHBOARD_CONFIG = os.path.join(WORKING_DIR, 'dashboard_config.yaml')
DESCRIPTION_TEMPLATE = os.path.join(WORKING_DIR, 'description_template.txt')


def generate_description_markdown(dashboard_config, experiment_config, experiment_dir):
    """ Generate an experiment's description """
    with open(DESCRIPTION_TEMPLATE) as md_file:
        data = md_file.read()
    data = data.format(
        experiment_config['experiment_name'],
        experiment_config['num_epoches'],
        experiment_config['description']
    )
    html_data = markdown2.markdown(data)
    result_file = os.path.join(
        experiment_dir,
        dashboard_config['plots_dir'],
        dashboard_config['description_name']
    )
    with open(result_file, 'w+') as html_file:
        html_file.write(html_data)

def generate_plots_for_experiment(dashboard_config, experiment_dir):
    """ Generate plots for one experiment """
    with open(os.path.join(experiment_dir, dashboard_config['experiment_config']), 
            encoding='utf8') as yaml_file:
        experiment_config = yaml.load(yaml_file)
    if not os.path.exists(os.path.join(experiment_dir, dashboard_config['plots_dir'])):
        os.makedirs(os.path.join(experiment_dir, dashboard_config['plots_dir']))
    generate_description_markdown(dashboard_config, experiment_config, experiment_dir)


def generate_experiments_dir_js(dashboard_config):
    """ Generate file experiments_dir.js """
    with open(os.path.join(DASHBOARD_ROOT, 'static', 'js', 'experiments_dir.js'), 'w+') as js_file:
        js_file.write("experiments_dir = {};".format(dashboard_config['experiments_dir']))


def execute():
    """ Execute script """
    with open(DASHBOARD_CONFIG, encoding='utf8') as yaml_file:
        dashboard_config = yaml.load(yaml_file)
    generate_experiments_dir_js(dashboard_config)
    for experiment_dir in dashboard_config['experiments_dir']:
        full_path = os.path.join(EXPERIMENTS_ROOT, experiment_dir)
        generate_plots_for_experiment(dashboard_config, full_path)


if __name__ == '__main__':
    execute()

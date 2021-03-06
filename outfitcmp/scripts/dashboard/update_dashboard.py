"""
Description: Update information on dashboard webpage
"""
import os
import yaml
import json
import demjson
import plotly.offline as of_py
import plotly.graph_objs as go

from outfitcmp.scripts.dashboard.generate_desc_markdown import generate_description_markdown, generate_short_description_markdown
from outfitcmp.scripts.dashboard.generate_model_results import generate_model_results
from outfitcmp.scripts.dashboard.generate_histogram import generate_histogram

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..', '..')
DASHBOARD_ROOT = os.path.join(ROOT_DIR, 'dashboard')
EXPERIMENTS_ROOT = os.path.join(ROOT_DIR, 'trained_models')
DASHBOARD_CONFIG = os.path.join(WORKING_DIR, 'dashboard_config.yaml')

def produce_graph_data(key, graph):
    """
    Extract the data for a graph
    """
    return (list(range(0, len(graph[key]))), graph[key])

def plot_metrics(experiment_dir, dashboard_config, plot_name, first_metric, second_metric):
    """
    Plot two given metrics on one graph
    """
    log_path = os.path.join(experiment_dir, dashboard_config['log_file'])
    with open(log_path) as log_file:
        graph = demjson.decode(log_file.read())
    x_values, y_values = produce_graph_data(first_metric, graph)
    trace1 = go.Scatter(
        x=x_values,
        y=y_values,
        marker=dict(color='#000099'),
        name=first_metric)
    x_values, y_values = produce_graph_data(second_metric, graph)
    trace2 = go.Scatter(
        x=x_values,
        y=y_values,
        marker=dict(color='#ffa500'),
        name=second_metric)
    of_py.plot(
        # {
        #     'data': [trace1, trace2],
        #     'layout': {'font': dict(size=24)}
        # },
        [trace1, trace2],
        filename=os.path.join(experiment_dir, dashboard_config['plots_dir'], plot_name),
        # image='png',
        # image_filename='_'.join([experiment_dir, plot_name[:-5]]),
        # image_width=600, image_height=500,
        # output_type='file',
        auto_open=False
    )

def generate_plots_for_experiment(dashboard_config, experiment_dir, experiment_config):
    """ Generate plots for one experiment """
    if not os.path.exists(os.path.join(experiment_dir, dashboard_config['plots_dir'])):
        os.makedirs(os.path.join(experiment_dir, dashboard_config['plots_dir']))
    else:
        results_filename = os.path.join(
            experiment_dir, 
            dashboard_config['plots_dir'], 
            dashboard_config['results_json_name'])
        with open(results_filename) as json_data:
            results = json.load(json_data)
        return results
    if 'manual_features' in experiment_dir or 'clothes_comparation' in experiment_dir or 'manual_features_plus_128floats' in experiment_dir or '128floats' in experiment_dir:
        generate_short_description_markdown(dashboard_config, experiment_config, experiment_dir)
    else:
        generate_description_markdown(dashboard_config, experiment_config, experiment_dir)
        plot_metrics(experiment_dir, dashboard_config, dashboard_config['loss_plot_name'], 'loss', 'val_loss')
        if experiment_config['is_regression']:
            plot_metrics(
                experiment_dir,
                dashboard_config,
                dashboard_config['mae_plot_name'],
                'mean_absolute_error', 'val_mean_absolute_error'
            )
            plot_metrics(
                experiment_dir,
                dashboard_config,
                dashboard_config['mse_plot_name'], 
                'mean_squared_error', 'val_mean_squared_error'
            )
        else:
            plot_metrics(experiment_dir, dashboard_config, dashboard_config['acc_plot_name'], 'acc', 'val_acc')
    generate_histogram(dashboard_config, experiment_dir, experiment_config)
    return generate_model_results(dashboard_config, experiment_config, experiment_dir)

def generate_experiments_dir_js(data):
    """ Generate file experiments_dir.js """
    with open(os.path.join(DASHBOARD_ROOT, 'static', 'js', 'experiments_dir.js'), 'w+') as js_file:
        js_file.write("experiments_dir = {};".format(data))

def execute():
    """ Execute script """
    with open(DASHBOARD_CONFIG, encoding='utf8') as yaml_file:
        dashboard_config = yaml.load(yaml_file)
    data = []
    for experiment_dir in os.listdir(EXPERIMENTS_ROOT):
        experiment_dict = {'name': experiment_dir, 'modifications' : []}
        full_path = os.path.join(EXPERIMENTS_ROOT, experiment_dir)
        for modification_dir in os.listdir(full_path):
            print(experiment_dir, modification_dir)
            full_modif_path = os.path.join(full_path, modification_dir)
            config_name = dashboard_config['experiment_config']
            if experiment_dir in ['manual_features', 'clothes_comparation', 'manual_features_plus_128floats', '128floats']:
                config_name = 'experiment_config.yaml'
            with open(os.path.join(full_modif_path, config_name), 
                    encoding='utf8') as yaml_file:
                experiment_config = yaml.load(yaml_file)
            results = generate_plots_for_experiment(dashboard_config, full_modif_path, experiment_config)
            experiment_dict['modifications'].append({
                'name' : modification_dir,
                'is_regression' : str(experiment_config['is_regression']),
                'results': results
            })
        data.append(experiment_dict)
    generate_experiments_dir_js(data)


if __name__ == '__main__':
    execute()

"""
Description: Update information on dashboard webpage
"""
import os
import yaml
import demjson
import plotly.offline as of_py
import plotly.graph_objs as go
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model

from generate_desc_markdown import generate_description_markdown
from generate_model_results import generate_model_results

WORKING_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(WORKING_DIR, '..', '..')
DASHBOARD_ROOT = os.path.join(ROOT_DIR, 'dashboard')
EXPERIMENTS_ROOT = os.path.join(ROOT_DIR, 'experiments')
DASHBOARD_CONFIG = os.path.join(WORKING_DIR, 'dashboard_config.yaml')

def generate_model_scheme(experiment_dir, dashboard_config):
    """
    Save a model's architecture as image 
    """
    model_path = os.path.join(
        experiment_dir,
        dashboard_config['model_file']
    )
    scheme_path = os.path.join(
        experiment_dir,
        dashboard_config['plots_dir'],
        dashboard_config['scheme_name']
    )
    with open(model_path, 'r') as json_file:
        model = model_from_json(json_file.read())
    plot_model(model, to_file=scheme_path, show_shapes=True)

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
        [trace1, trace2],
        filename=os.path.join(experiment_dir, dashboard_config['plots_dir'], plot_name),
        auto_open=False
    )

def generate_plots_for_experiment(dashboard_config, experiment_dir):
    """ Generate plots for one experiment """
    with open(os.path.join(experiment_dir, dashboard_config['experiment_config']), 
            encoding='utf8') as yaml_file:
        experiment_config = yaml.load(yaml_file)
    if not os.path.exists(os.path.join(experiment_dir, dashboard_config['plots_dir'])):
        os.makedirs(os.path.join(experiment_dir, dashboard_config['plots_dir']))
    else:
        #return
        pass
    generate_description_markdown(dashboard_config, experiment_config, experiment_dir)
    plot_metrics(experiment_dir, dashboard_config, dashboard_config['loss_plot_name'], 'loss', 'val_loss')
    plot_metrics(experiment_dir, dashboard_config, dashboard_config['acc_plot_name'], 'acc', 'val_acc')
    generate_model_scheme(experiment_dir, dashboard_config)
    generate_model_results(dashboard_config, experiment_config, experiment_dir)

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

"""
Description: Generate html with experiment's results using markdown format
"""
import os
import json
import markdown2

from outfitcmp.scripts.dashboard.generate_desc_markdown import decorated_format
from outfitcmp.scripts.dashboard.estimate_model import estimate

WORKING_DIR = os.path.dirname(__file__)
DESCRIPTION_TEMPLATE = os.path.join(WORKING_DIR, 'results_template.txt')

def generate_model_results(dashboard_config, experiment_config, experiment_dir):
    """ Generate an experiment's results """
    results = estimate(experiment_dir, experiment_config)
    with open(DESCRIPTION_TEMPLATE) as md_file:
        data = md_file.read()
    data = decorated_format(
        data,
        results['acc_0'],
        results['acc_1'],
        results['acc_2'],
        results['precision'],
        results['recall'],
        results['MAE'],
        results['MSE'],
        results['pairs']
    )
    html_data = markdown2.markdown(data, extras=["tables", "wiki-tables"])
    result_file = os.path.join(
        experiment_dir,
        dashboard_config['plots_dir'],
        dashboard_config['results_name']
    )
    with open(result_file, 'w+') as html_file:
        html_file.write(html_data)
    result_json_file = os.path.join(
        experiment_dir,
        dashboard_config['plots_dir'],
        dashboard_config['results_json_name']
    )
    with open(result_json_file, 'w+', encoding='utf8') as json_file:
        json.dump(results, json_file)
    return results
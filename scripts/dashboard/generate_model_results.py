"""
Description: Generate html with experiment's results using markdown format
"""
import os
import markdown2

from generate_desc_markdown import decorated_format

WORKING_DIR = os.path.dirname(__file__)
DESCRIPTION_TEMPLATE = os.path.join(WORKING_DIR, 'results_template.txt')

def generate_model_results(dashboard_config, experiment_config, experiment_dir):
    """ Generate an experiment's results """
    with open(DESCRIPTION_TEMPLATE) as md_file:
        data = md_file.read()
    data = decorated_format(
        data,
        "TODO",
        "TODO",
        "TODO",
        "TODO",
        "TODO"
    )
    html_data = markdown2.markdown(data, extras=["tables", "wiki-tables"])
    result_file = os.path.join(
        experiment_dir,
        dashboard_config['plots_dir'],
        dashboard_config['results_name']
    )
    with open(result_file, 'w+') as html_file:
        html_file.write(html_data)

    return {"acc_0": "TODO", "acc_1": "TODO", "acc_2": "TODO", "MAE": "TODO", "pairs": "TODO"}
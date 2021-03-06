"""
Description: Generate a histogram for a comparation of 
    distributions of true and predicted labels
"""
import os
import yaml
import plotly.offline as of_py
import plotly.graph_objs as go

from outfitcmp.scripts.dashboard.utils import load_predictions

def generate_histogram(dashboard_config, experiment_dir, experiment_config):
    ''' Generate a histogram '''
    results = load_predictions(experiment_dir, experiment_config)
    trace1 = go.Bar(
        x=[i for i in range(1,11)],
        y=[list(results['y_true']).count(i) for i in range(1,11)],
        name='y_true',
        marker=dict(
            color='#5eb8ff',
        ),
        opacity=0.75
    )
    trace2 = go.Bar(
        x=[i for i in range(1,11)],
        y=[list(results['y_pred_class']).count(i) for i in range(1,11)],
        name='y_pred',
        marker=dict(
            color='#005b9f'
        ),
        opacity=0.75
    )
    data = [trace1, trace2]

    layout = go.Layout(
        xaxis=dict(
            title='Class',
            dtick=1
        ),
        yaxis=dict(
            title='Count'
        ),
        bargap=0.2,
        bargroupgap=0.1,
        #font=dict(size=20)
    )
    fig = go.Figure(data=data, layout=layout)
    of_py.plot(
        fig,
        filename=os.path.join(
            experiment_dir,
            dashboard_config['plots_dir'],
            dashboard_config['histogram_name']),
        # image='png',
        # image_filename='_'.join([experiment_dir, dashboard_config['histogram_name'][:-5]]),
        # image_width=600, image_height=500,
        # output_type='file',
        auto_open=False
    )

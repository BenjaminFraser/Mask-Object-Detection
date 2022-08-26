import dash
import dash_daq as daq
import flask
import glob
import os

from dash import html, dcc
from dash.dependencies import Input, Output

# main directories for our app to load images & pose features
DATA_DIR = ('/Users/benjaminfraser/Documents/Programming/' + 
            'Mask-Object-Detection/data/custom_mask_dataset_cleaned')
IMAGE_DIR = os.path.join(DATA_DIR, 'Images')
POSE_RESULTS_PATH = os.path.join(DATA_DIR, 'alphapose_results/alphapose-results.json')
ANNOTATIONS_IMAGE_DIR = os.path.join(DATA_DIR, 'alphapose_results/vis')

list_of_images = [x for x in os.listdir(IMAGE_DIR) 
                    if (x.endswith('.jpg') or x.endswith('.png'))]

print(f"Number of files found: {len(list_of_images)}")

static_image_route = '/static/'
pose_image_route = '/poses/'

app = dash.Dash()

app.layout = html.Div([
    html.H1("Pose Dataset Explorer"),
    html.H2("Image Selection:"),
    dcc.Dropdown(
        id='image-dropdown',
        options=[{'label': i, 'value': i} for i in list_of_images],
        value=list_of_images[0],
        style={'height' : '50%', 'width':'50%'}
    ),
    html.Br(),
    html.Img(id='image', style={'height':'40%', 'width':'40%'}),
    html.Br(),
    html.Br(),
    daq.BooleanSwitch(id="show-pose-button", on=False,
                     color="green", label='Show Poses:', 
                     labelPosition='top', style={'height' : '50%', 'width':'50%'}),
])


@app.callback(
    Output('image', 'src'),
    [Input('image-dropdown', 'value'),
     Input('show-pose-button', 'on')])
def update_image_src(value, on):
    """ Callback function to obtain route to desired image to display 
        and update the html so that this image is presented.
        If pose features are selected via the toggle button, load the associated
        image showing the pose features. """ 
    if on:
        return pose_image_route + value
    else:
        return static_image_route + value


## TO-DO: Add a toggle button, which introduces the model and makes predictions
#         on the currently selected image.

# Add a static image route that serves images from local system
@app.server.route(f'{static_image_route}<image_path>')
def serve_image(image_path):
    image_name = image_path
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(IMAGE_DIR, image_name)


@app.server.route(f'{pose_image_route}<image_path>')
def serve_pose_image(image_path):
    image_name = image_path
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(ANNOTATIONS_IMAGE_DIR, image_name)

if __name__ == '__main__':
    app.run_server(debug=True)
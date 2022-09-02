#!/usr/bin/env python

# import dependencies
import cv2
import dash
import dash_bootstrap_components as dbc
import flask
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.tools as tls
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torchvision

from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
from flask import Flask, Response
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

# set device as appropriate
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# set basic directories and filepaths for dashboard
PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.join(PROJECT_DIR, 'data/custom_mask_dataset_cleaned')
IMAGE_DIR = os.path.join(DATA_DIR, 'Images')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')

# get a list of all of our images
list_of_images = [x for x in os.listdir(IMAGE_DIR) 
                    if (x.endswith('.jpg') or x.endswith('.png'))]

# supported result plot types to show on app
PLOT_RESULT_TYPES = ['Normal', 'Masks']


# define a class map for our model(s) - reserve 0 for background
OBJECT_CLASS_MAP = {1 : 'with_mask',
                    2 : 'without_mask'}

# 3 total classes - including background
N_CLASSES = len(OBJECT_CLASS_MAP.keys()) + 1

# set appropriate path to model (depending on where it was saved above)
MODEL_NAME = "frcnn_mask_dect_260822.pth"
SAVED_MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

def get_model_object_detector(num_classes, pretrained=True):
    """ Load pre-trained Faster RCNN object detection model
        and build a new prediction head for our UAV task 
    """
    if pretrained:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    else: 
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                        pretrained_backbone=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


TRAINED_MODEL = get_model_object_detector(N_CLASSES, pretrained=False)

# load model weights gained from training
TRAINED_MODEL.load_state_dict(torch.load(SAVED_MODEL_PATH, 
                                map_location=DEVICE))

# set model evaluation mode and send to our device
TRAINED_MODEL.eval()
TRAINED_MODEL.to(DEVICE)


# start flask server and dash application
server = Flask(__name__)
app = dash.Dash(__name__, server=server, 
                external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Facemask Detector'


# define main html layout for our application
app.layout = html.Div(
    [

        # simple narbar 
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Home", href="#")),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem("Support", header=True),
                        dbc.DropdownMenuItem("Support", href="#"),
                ],
                    nav=True,
                    in_navbar=True,
                    label="More",
                    ),
                ], brand="FaceMask Platform", 
                   brand_href="#",
                   color="dark", 
                   dark=True
            ),

        # app title
        dbc.Row([
                html.Div([
                    html.H1("Face-Mask Detection"),
                    dcc.Markdown(''' 
                        *Human Face Mask Analysis, powered by Object Detection.*
                        '''),
                ], style={"width": "100%", 
                          "text-align": "center",
                          "padding-top" : 10,
                          "background-color": '#F8F9F9'})
            ]),

        # image selection
        dbc.Row([

                dbc.Col([
                        html.H3("Select Image:", 
                                style={"width": "100%", 
                                       "text-align": "right"})
                    ], width=6),

                dbc.Col([
                        dcc.Dropdown(
                            id='image-dropdown',
                            options=[{'label': i, 'value': i} for i in list_of_images],
                            value=list_of_images[0],
                            style={'height' : '60%', 'width':'100%', 
                                   'text-align' : 'left', 'background-color' : '#A5E8FF'})
                    ], width=2)
            ], style={'padding-top' : 20, "background-color": '#F8F9F9'}),


        # threshold selection
        # selection of constant velocity coefficient (0-1.0)
        dbc.Row([

                dbc.Col([
                        html.H5("Prediction Threshold:", 
                                style={"width": "100%", 
                                       "text-align": "center"}),
                        ], width=3),

                dbc.Col([
                        dcc.Slider(0.05, 1.0, 0.05, value=0.5, 
                                   id='threshold-value')], width=7)
            ], style={'padding-top' : 30, "background-color": '#F8F9F9'}),


    # horizontal line to divide top section from results below
    dbc.Row(
        dbc.Col(
            html.Hr(style={'borderWidth': "1.0vh", "width": "100%", 
                           "backgroundColor": "#BBBBBB","opacity":"1"}),
                    width={'size':10, 'offset':1}),

            style={"background-color": '#F8F9F9'}), 


    # html content to be generated for selected area
    html.Div(children=[html.Div(id='results-section')])

    ])


def make_predictions(torch_img, model, threshold=0.2):
    """ Model inference function on new single image """
    with torch.no_grad():
        prediction = model([torch_img.to(DEVICE)])[0]
    
    # get bounding boxes and scores
    boxes = prediction['boxes'].data.cpu().numpy()
    scores = prediction['scores'].data.cpu().numpy()
    classes = prediction['labels'].data.cpu().numpy()
    
    # keep predictions above designated threshold
    boxes = boxes[scores >= threshold].astype(np.int32)
    classes = classes[scores >= threshold]
    scores = scores[scores >= threshold]
    
    return boxes, scores, classes


def create_pred_figure(img, target, classes, confidences):
    """ Plot image with associated bounding-box(es) on plotly
        figure.
    """
    # set up figure with base image 
    fig = px.imshow(np.array(img))
    
    # get co-ords for each box and add to plot
    for i, box in enumerate(target):
        xmin, ymin, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        
        # get object class id and convert to class name
        class_id = classes[i]
        class_name = OBJECT_CLASS_MAP[class_id]
        confidence = str(confidences[i])
        
        color = 'limegreen' if class_id == 1 else 'red'

        # add bbox to figure
        fig.add_trace(
            go.Scatter(
                x=[xmin + xi for xi in [0, 0, width, width, 0]],
                y=[ymin + yi for yi in [0, height, height, 0, 0]],
                mode='lines', line=dict(color=color))
        )

        # add labels to figure with pred confidence
        fig.add_trace(
            go.Scatter(
            x=[xmin + (width / 2)],
            y=[ymin - 10],
            text=[confidence[:4]],
            mode="text",
            textfont=dict(size=10, color=color))
        )

    # Hide the axes and the tooltips for the figure
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=0, b=0, l=20, r=20),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            linewidth=0),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            linewidth=0),
        hovermode=False,
        showlegend=False)

    return fig


@server.route('/static/<path:path>')
def serve_static(path):
    """ Serve item from static DIR from path given """ 
    return flask.send_from_directory(path)


@app.callback(
    Output('results-section', 'children'),
    Input('image-dropdown', 'value'),
    Input('threshold-value', 'value'))
def load_chosen_area(image_ref, threshold):
    """ Callback function to show image and associated object detection results,
        according to the value of the dropdown menu (image-dropdown).

    Args:
        image_ref (str) : image name (from dropdown menu) to display.
        plot_type (str) : type of results plot to show for chosen image.
    """ 

    # load image into suitable format
    img_id = str(image_ref[:-4])
    image_filepath = os.path.join(IMAGE_DIR, image_ref)
    img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # preprocess image suitably for our model
    img_proc = img.astype(np.float32) / 255.0
    img_proc = torchvision.transforms.ToTensor()(img_proc)

    # make predictions on current image, using selected threshold
    pred_boxes, pred_scores, pred_classes = make_predictions(img_proc, 
                                                             TRAINED_MODEL,
                                                             threshold=threshold)
    
    # get plotly figure with annotated predictions
    pred_fig = create_pred_figure(img, pred_boxes, pred_classes, pred_scores)

    # get mask proportion in scene, add to our total results
    mask_count = np.bincount(pred_classes, minlength=2)[1]
    mask_prop = np.array(mask_count / pred_classes.shape[0])

    # create a simple pie chart of mask usage
    mask_labels = ['Mask','No Mask']
    mask_pie_counts = [mask_count, pred_classes.shape[0] - mask_count]
    mask_pie_fig = go.Figure(data=[go.Pie(labels=mask_labels, 
                             values=mask_pie_counts)])

    # create figure with normal image
    normal_fig = px.imshow(np.array(img))

    # Hide the axes and the tooltips for the figure
    normal_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=0, b=0, l=20, r=20),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            linewidth=0
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            linewidth=0
        ),
        hovermode=False
    )

    print(f"Plotting normal fig for {image_ref}")


    return_content = html.Div([

        # row to display input video and associated heatmap predictions
        dbc.Row(
            [
                dbc.Col([
                    html.H4("Original Image", 
                            style={"width": "100%", "text-align": "center"}),

                    dcc.Graph(id='normal-image', figure=normal_fig),

                    ], style={"background-color": '#ecfaff ', 'padding-top' : 5, 
                              'padding-bottom' : 5, "text-align": "center"}, 
                       width=6),
                dbc.Col([
                    html.H4("Predictions", style={"width": "80%", "text-align": "center"}),

                    # image showing predictions
                    dcc.Graph(id='prediction-image', figure=pred_fig),

                    ], style={"background-color": '#f0fbf5 ', 'padding-top' : 5, 
                              'padding-bottom' : 5, "text-align": "center"}, 
                       width=6),    
            ],
        ),

        dbc.Row(
            [
                dbc.Col([

                        html.H5(f"Total Persons: {pred_classes.shape[0]}", 
                            style={"text-align": "center", 'padding-top' : 40}),
                        html.H5(f"Mask Count: {mask_count}", 
                            style={"text-align": "center", 'padding-top' : 20}),
                        html.H5(f"Mask Proportion: {mask_prop*100:.1f}%", 
                            style={"text-align": "center", 'padding-top' : 20}),
                    ],
                       style={"background-color": '#ecfaff ', 'padding-top' : 10, 
                              'padding-bottom' : 5, "text-align": "center"}, 
                        width=6),
                
                # column with pie chart
                dbc.Col([
                    html.H4("Mask Usage Summary", 
                            style={"width": "100%", "text-align": "center"}),

                    dcc.Graph(id='mask-pie-chart', figure=mask_pie_fig),

                    ], style={"background-color": '#f0fbf5 ', 'padding-top' : 10, 
                              'padding-bottom' : 5, 'padding-right' : 20,
                              "text-align": "center"}, 
                       width=6),
            ],
        ),

    ])

    return return_content


if __name__ == '__main__':
    app.run_server(debug=True) 

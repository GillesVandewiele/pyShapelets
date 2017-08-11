from flask import Flask, render_template
import numpy as np
import operator
from bokeh.embed import components
from bokeh.plotting import figure

import json

app = Flask(__name__)


@app.route("/")
def chart():
    """
    N_SHAPELETS = 15
    shapelets = []
    for _ in range(N_SHAPELETS):
        shapelet_size = np.random.randint(10, 50)
        shapelets.append((np.random.uniform(size=shapelet_size), np.random.random()))

    level = 0
    root = ShapeletTree(shapelet=shapelets[0][0], distance=shapelets[0][1])
    nodes = [root]
    shapelets.remove(shapelets[0])

    while len(nodes):
        new_nodes = []
        for node in nodes:
            if len(shapelets):
                child_shapelet, child_distance = shapelets[0][0], shapelets[0][1]
                node.left = ShapeletTree(shapelet=child_shapelet, distance=child_distance)
                new_nodes.append(node.left)
                shapelets.remove(shapelets[0])
            if len(shapelets):
                child_shapelet, child_distance = shapelets[0][0], shapelets[0][1]
                node.right = ShapeletTree(shapelet=child_shapelet, distance=child_distance)
                new_nodes.append(node.right)
                shapelets.remove(shapelets[0])
        nodes = new_nodes

    root.left.left = None
    """

    json, scripts = plot_shapelet_tree(shap_tree)
    print(json)

    return render_template("tree.html", node_json=json, node_scripts=scripts)


def plot_shapelet(shapelet):
    ts = figure(plot_width=100, plot_height=100, active_drag="pan",
                active_scroll="wheel_zoom")

    ts.xaxis.visible = False
    ts.toolbar.logo = None
    ts.toolbar_location = None
    ts.title.text = str(np.around(shapelet.distance, 3))

    ts.line(x=range(len(shapelet.shapelet)), y=shapelet.shapelet)

    return ts


def get_shapelet_html(shapelet):
    if shapelet.distance is not None:
        ts = figure(plot_width=100, plot_height=100, active_drag="pan",
                    active_scroll="wheel_zoom")

        ts.xaxis.visible = False
        ts.toolbar.logo = None
        ts.toolbar_location = None
        ts.title.text = str(np.around(shapelet.distance, 3))

        ts.line(x=range(len(shapelet.shapelet)), y=shapelet.shapelet)

        node_script, node_div = components(ts)
        node_div = node_div.replace('"', "'").replace("\n", "")

        return node_script, node_div
    else:
        return max(shapelet.class_probabilities.items(), key=operator.itemgetter(1))[0], None


def plot_shapelet_tree(shapelet_tree, samples=[]):
    tree = {}
    tree['chart'] = {'container': '#tree-simple', 'node': {'collapsable': True}}
    scripts = []
    nodes_to_plot = [(shapelet_tree, None, 1)]
    while len(nodes_to_plot):
        new_nodes = []
        for node in nodes_to_plot:
            node_script, node_div = get_shapelet_html(node[0])
            if node_div is not None:
                scripts.append(node_script)
                node_object = {'innerHTML': node_div, 'children': []}
            else:
                node_object = {'text': {'name': str(node_script)}, 'children': []}
            if node[2]:
                node_object['HTMLclass'] = 'left_child'
            else:
                node_object['HTMLclass'] = 'right_child'

            if node[1] is None:
                tree['nodeStructure'] = node_object
            else:
                node[1].append(node_object)

            if node[0].left is not None:
                new_nodes.append((node[0].left, node_object['children'], 1))
            if node[0].right is not None:
                new_nodes.append((node[0].right, node_object['children'], 0))
        nodes_to_plot = new_nodes
    return json.dumps(tree), scripts


def run(shapelet_tree):
    global shap_tree
    shap_tree = shapelet_tree
    app.run()

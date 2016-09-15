from plotly.graph_objs import *


def get_brain_figure(g, atlas_data, plot_title=''):
    """
    Returns the plotly figure object for vizualizing a 3d brain network.

    g: igraph object of brain

    atlas_data: pandas DataFrame containing the x,y,z coordinates of
    each brain region


    Example
    -------
    import plotly
    plotly.offline.init_notebook_mode()

    fig = get_brain_figure(g, atlas_data)
    plotly.offline.iplot(fig)
    """

    # grab the node positions from the centroids file
    V = atlas_data.shape[0]
    node_positions_3d = pd.DataFrame(columns=['x', 'y', 'z'], index=range(V))
    for r in range(V):
        node_positions_3d.loc[r] = atlas_data.loc[r, ['x', 'y', 'z']].tolist()

    # grab edge endpoints
    edge_x = []
    edge_y = []
    edge_z = []
    for e in g.es:
        source_pos = node_positions_3d.loc[e.source]
        target_pos = node_positions_3d.loc[e.target]

        edge_x += [source_pos['x'], target_pos['x'], None]
        edge_y += [source_pos['y'], target_pos['y'], None]
        edge_z += [source_pos['z'], target_pos['z'], None]

    # node style
    node_trace = Scatter3d(x=node_positions_3d['x'],
                           y=node_positions_3d['y'],
                           z=node_positions_3d['z'],
                           mode='markers',
                           # name='regions',
                           marker=Marker(symbol='dot',
                                         size=6,
                                         color='red'),
                           # text=[str(r) for r in range(V)],
                           text=atlas_data['name'],
                           hoverinfo='text')

    # edge style
    edge_trace = Scatter3d(x=edge_x,
                           y=edge_y,
                           z=edge_z,
                           mode='lines',
                           line=Line(color='black', width=.5),
                           hoverinfo='none')

    # axis style
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False)

    # overall layout
    layout = Layout(title=plot_title,
                    width=800,
                    height=900,
                    showlegend=False,
                    scene=Scene(xaxis=XAxis(axis),
                                yaxis=YAxis(axis),
                                zaxis=ZAxis(axis)),
                    margin=Margin(t=50),
                    hovermode='closest')

    data = Data([node_trace, edge_trace])
    fig = Figure(data=data, layout=layout)

    return fig

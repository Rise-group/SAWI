import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import shapely as sh
import igraph as ig

def ckdnearest(gdfA_:gpd.GeoDataFrame, gdfB_:gpd.GeoDataFrame):
    '''
    Function to search for the nearest point from gdB for each point from gdA
    gdfs must be in cartesian coordinates to calculate the length and uniquely indexed

    Extracted from: https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
    
    Args:
        gdfA_ (gpd.GeoDataFrame): A GeoDataFrame containing the source points
        gdfB_ (gpd.GeoDataFrame): A GeoDataFrame containing the target points to find the nearest ones to those in gdfA_. 
    Return:
        gdf (pd.DataFrame): A DataFrame containing:
            All columns from gdfA_, suffixed with '_from'.
            All columns from gdfB_, suffixed with '_to'.
            A 'length' column containing the calculated distances between the nearest points, rounded to four decimal places.
    '''
    # Generar copias de los geodataframes
    gdfA = gdfA_.copy()
    gdfB = gdfB_.copy()
    # Cambiar el nombre de las columnas
    gdfA.columns = [col+'_from' for col in gdfA.columns]
    gdfB.columns = [col+'_to' for col in gdfB.columns]
    gdfA = gdfA.rename(columns={'geometry_from':'geometry'})
    gdfB = gdfB.rename(columns={'geometry_to':'geometry'})
    # nam = 'geometry_'+kind
    nA = np.array(list(gdfA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdfB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdfB_nearest = gdfB.iloc[idx].rename(columns={"geometry":'geometry_to'}).reset_index(drop=True)
    gdf = pd.concat(
        [
            gdfA.rename(columns={"geometry":'geometry_from'}).reset_index(),
            gdfB_nearest,
            pd.Series(dist, name='length')
        ], 
        axis=1)
    for i in range(len(gdf)):
        gdf.loc[i,'length'] = np.round(gdf.loc[i,'length'],4)
    
    return gdf

def create_links(gdf:gpd.GeoDataFrame, elev:bool, crs='epsg:4326'):
    '''
    Function to create links that conect each centroid and the nearest node
    '''
    # Create a LineString between 'geometry_centroid' (Point) and 'geometry_node' (Point)
    gdf['pre_geometry'] = gdf.apply(lambda row: sh.geometry.LineString([row['geometry_from'], row['geometry_to']]).wkt, axis=1)
    gdf['pre_geometry'] = gdf['pre_geometry'].apply(sh.wkt.loads)

    gdf = gdf.rename(columns={'pre_geometry':'geometry'})
    gdf['ID'] = gdf['ID_from']+'_'+gdf['ID_to']
    if elev == True:
        gdf['elev_from'] = gdf['elev_from'].astype(float)
        gdf['elev_to'] = gdf['elev_to'].astype(float)
        gdf['grade_abs'] = np.absolute(np.round((gdf['elev_from']-gdf['elev_to'])/gdf['length'],4))
    
    # Create another GeoDataFrame and set the crs again. This is necesary because the GeoDataFrame 
    # transform into a DataFrame when we eliminated some columns
    try:
        gdf = gpd.GeoDataFrame(gdf[['ID','ID_from','ID_to','grade_abs','length','geometry']],geometry='geometry')
    except KeyError:
        gdf = gpd.GeoDataFrame(gdf[['ID','ID_from','ID_to','length','geometry']],geometry='geometry')
    gdf = gdf.set_crs(crs=crs)
    gdf = gdf.rename(columns={'ID_from':'from','ID_to':'to'})
    
    return gdf

# def get_links_grade(nodes, nearest, links, elevation_name='elevation'):
#     '''
    
#     '''
#     links['grade_abs'] = ''
#     # Calculate the grade as the slope of a line
#     for ilinks in links.index:
#         elevation_O = nodes.loc[ilinks,elevation_name]
#         elevation_D = nearest.loc[ilinks,elevation_name]
#         length = links.loc[ilinks,'length']
#         grade = (elevation_D-elevation_O)/length
#         links.loc[ilinks,'grade_abs'] = np.absolute(np.round(grade,4))

#     links = links.reset_index()

#     return links

def get_links_grade(links, from_gdf, to_gdf):
    '''
    
    '''
    links['grade_abs'] = ''
    from_gdf = from_gdf.set_index('ID')
    to_gdf = to_gdf.set_index('ID')
    # Calculate the grade as the slope of a line
    for ilinks in links.index:
        elevation_O = from_gdf.loc[links.loc[ilinks,'from'],'elevation']
        elevation_D = to_gdf.loc[links.loc[ilinks,'to'],'elevation']
        length = links.loc[ilinks,'length']
        grade = (elevation_D-elevation_O)/length
        links.loc[ilinks,'grade_abs'] = np.absolute(np.round(grade,4))

    # links = links.reset_index()

    return links

def assign_walk_speed_Naismith_Langmuir(gdf:gpd.GeoDataFrame, units='m/s'):
    '''
    Asigna velocidad de caminata en función del gradiente (pendiente) usando el método Naismith-Langmuir.
    '''
    # Pedestrian walking speed in m/s
    walk_speed_m = {'s1': {0: 0.0, 1: 0.0, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4, 6: 0.5, 7: 0.6, 8: 0.7, 9: 0.8, 10: 0.9, 11: 1.0}, 
                    's2': {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5, 6: 0.6, 7: 0.7, 8: 0.8, 9: 0.9, 10: 1.0, 11: 5.0}, 
                    'speed': {0: 1.3888888888888888, 1: 0.9803921569444444, 2: 0.6172839505555555, 3: 0.45045045055555555, 
                              4: 0.35460992916666667, 5: 0.29239766083333335, 6: 0.24875621888888888, 7: 0.2164502163888889, 
                              8: 0.19157088111111112, 9: 0.17182130583333335, 10: 0.15576324, 11: 0.053418803333333334}}
    # Pedestrian walking speed in km/h
    walk_speed_k = {'s1': {0: 0.0, 1: 0.0, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4, 6: 0.5, 7: 0.6, 8: 0.7, 9: 0.8, 10: 0.9, 11: 1.0},
                    's2': {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5, 6: 0.6, 7: 0.7, 8: 0.8, 9: 0.9, 10: 1.0, 11: 5.0}, 
                    'speed': {0: 5.0, 1: 3.529411765, 2: 2.222222222, 3: 1.621621622, 4: 1.276595745, 5: 1.052631579, 
                              6: 0.895522388, 7: 0.779220779, 8: 0.689655172, 9: 0.618556701, 10: 0.560747664, 11: 0.192307692}}

    if units == 'm/s':
        walk_speed_data = pd.DataFrame.from_dict(walk_speed_m)
    elif units == 'km/h':
        walk_speed_data = pd.DataFrame.from_dict(walk_speed_k)

    # Verificación de la columna 'grade_abs'
    if 'grade_abs' not in gdf.columns or gdf['grade_abs'].isna().astype(int).sum() != 0:
        raise AttributeError("'grade_abs' column is missing")

    # Inicializa la columna 'speed' a 0
    gdf['speed'] = 0.0

    # Reescribe las condiciones para calcular las velocidades de manera vectorizada
    for j in range(len(walk_speed_data)):
        # Selecciona las filas donde la pendiente 'grade_abs' cae en el intervalo de s1 a s2
        mask = (gdf['grade_abs'] > walk_speed_data.loc[j, 's1']) & (gdf['grade_abs'] <= walk_speed_data.loc[j, 's2'])
        gdf.loc[mask, 'speed'] = walk_speed_data.loc[j, 'speed']

    # Asignar velocidad a los segmentos con pendiente 0
    gdf.loc[gdf['grade_abs'] == 0, 'speed'] = walk_speed_data['speed'].iloc[0]
    # Para aquellos que superen el máximo valor de pendiente, asigna el último valor de velocidad
    gdf.loc[gdf['grade_abs'] > walk_speed_data['s2'].max(), 'speed'] = walk_speed_data['speed'].iloc[-1]

    return gdf

def organize_nodes(nodes):
    '''
    Function to organize the nodes for its export
    '''
    # Get columns with the coordinates x and y of each point
    nodes['coord_X'] = np.round(nodes['geometry'].x,4)
    nodes['coord_Y'] = np.round(nodes['geometry'].y,4)
    nodes = nodes.rename(columns={'osmid':'ID','elevation':'coord_Z'})
    # Create the type
    nodes['Type'] = 'network'
    # Organize the gdf
    nodes = nodes[['ID','coord_X','coord_Y','coord_Z','Type','geometry']]
    
    # Transform 'ID' column into str
    nodes['ID'] = nodes['ID'].astype(str)
    for i in range(len(nodes)):
        # Get the int part of the str
        nodes.loc[i,'ID'] = nodes.loc[i,'ID'].split('.')[0]
    
    return nodes


def organize_edges(edges):
    '''
    
    '''
    edges = edges.round(3)
    edges = edges.rename(columns={'fid': 'ID'})
    edges['Type'] = 'network'
    edges[['ID', 'from', 'to']] = edges[['ID', 'from', 'to']].astype(str).apply(lambda x: x.str.split('.').str[0])
    
    return edges[['ID', 'from', 'to', 'length', 'grade_abs', 'speed', 'weight', 'Type', 'geometry']]


def organize_origins(origins):
    '''
    Function to organize the origins for its export
    '''
    origins = origins.rename(columns={'elevation':'coord_Z'})
    # Get columns with the coordinates x and y of each point
    origins['coord_X'] = np.round(origins['geometry'].x,4)
    origins['coord_Y'] = np.round(origins['geometry'].y,4)
    origins['Type'] = 'origin'
    
    # Organize the gdf
    origins = origins[['ID','coord_X','coord_Y','coord_Z','Type','geometry']]
    
    return origins


def organize_origins_links(origins_links):
    '''
    Function to organize the origin links for its export
    '''
    origins_links['ID'] = origins_links['from']
    origins_links['Type'] = 'origin'
    # Organize the gdf
    origins_links = origins_links[['ID','from','to','length','grade_abs','speed','weight','Type','geometry']]

    # # Transform 'from' and 'to' columns into str
    # origins_links['to'] = origins_links['to'].astype(str)
    # origins_links['from'] = origins_links['from'].astype(str)
    # for i in range(len(origins_links)):
    #     # Get the int part of the str
    #     origins_links.loc[i,'to'] = origins_links.loc[i,'to'].split('.')[0]
    #     origins_links.loc[i,'from'] = origins_links.loc[i,'from'].split('.')[0]
    
    return origins_links


def organize_destinations(destinations,blocks,origins):
    '''
    Function to organize the destinations for its export
    '''
    destinations = destinations[['ID','elevation','Type','CSIMBOL','geometry']]
    # get the ID of the block that contains a destination, this eliminates the destinations
    # that are not inside a block, so we have to correct this in the next steps
    join = destinations.sjoin(blocks,how='inner',predicate='intersects')
    # Search the missing indexes
    l_missing_index = []
    for i in destinations.index:
        if i not in join.index:
            l_missing_index.append(i)
    # Add the missing index from the original gdf to the join
    for i in range(len(l_missing_index)):
        join.loc[l_missing_index[i]] = destinations.loc[l_missing_index[i]]
    # Organize the columns
    join = join.sort_index()
    join = join[['ID','elevation','Type','CSIMBOL','MANZ_CCNCT','geometry']]
    join['coord_X'] = np.round(join['geometry'].x,4)
    join['coord_Y'] = np.round(join['geometry'].y,4)
    join = join.rename(columns={'elevation':'coord_Z'})
    join = join[['ID','coord_X','coord_Y','coord_Z','Type','MANZ_CCNCT','CSIMBOL','geometry']]
    # Create status for the destinations outside or inside of block
    for i in join.index:
        if i in l_missing_index:
            join.loc[i,'Status'] = 'OutOfBlock'
        else:
            join.loc[i,'Status'] = 'InsideOfBlock'
    # For the destinations "OutOfBlock" we need the get the nearest block
    # Change the columns names becouse in the ckdnearest function we need others names, so we use the suffix R
    # to identify the original columns
    if len(join.loc[join['Status']=='OutOfBlock']) != 0:
        near = ckdnearest(join.loc[join['Status']=='OutOfBlock'],origins)
        near = near.rename(columns={'geometry_from':'geometry_OutOfBlock'})
        near['MANZ_CCNCT'] = near['ID_to']
        near = near[['ID_from','coord_X_from','coord_Y_from','coord_Z_from','Type_from','MANZ_CCNCT','CSIMBOL_from','geometry_OutOfBlock','Status_from']]
        # Rename the columns to its originals names
        near = near.rename(columns={'ID_from':'ID','geometry_OutOfBlock':'geometry','coord_X_from':'coord_X','coord_Y_from':'coord_Y',
                                    'coord_Z_from':'coord_Z','Type_from':'Type','MANZ_CCNCT_from':'MANZ_CCNCT'})
        # the gdf near contains only block's ID of the destinations that doesn't intersects with a block, so we need to add this data
        # to the gdf join where all the others destinations are.
        combined_df = pd.merge(join, near[['ID', 'MANZ_CCNCT']], on='ID', how='inner', suffixes=('', '_new')).set_index('ID')
        join = join.set_index('ID')
        join.loc[combined_df.index, 'MANZ_CCNCT'] = combined_df['MANZ_CCNCT_new']
        join = join.reset_index()
    join = join.set_crs('epsg:4326', allow_override=True)
            
    return join


def organize_destinations_links(destinations_links, destinations):
    '''
    Function to organize the destination links for its export
    '''
    destinations_links['ID'] = destinations_links['from']
    destinations_links = destinations_links.set_index('from').merge(destinations[['ID','Type']].set_index('ID'), left_index=True, right_index=True, how='left').reset_index()
    # Organize the gdf
    # destinations_links = destinations_links.rename(columns={'osmid':'to'})
    destinations_links = destinations_links[['ID','from','to','length','grade_abs','speed','weight','Type','geometry']]

    # # Transform 'from' and 'to' columns into str
    # destinations_links['to'] = destinations_links['to'].astype(str)
    # destinations_links['from'] = destinations_links['from'].astype(str)
    # for i in range(len(destinations_links)):
    #     # Get the int part of the str
    #     destinations_links.loc[i,'to'] = destinations_links.loc[i,'to'].split('.')[0]
    #     destinations_links.loc[i,'from'] = destinations_links.loc[i,'from'].split('.')[0]
    
    return destinations_links


def OD_matrix_igraph(graph_nodes, graph_edges):
    '''
    Calculates the shortest path between origin and destination nodes in a weighted graph using the igraph library.

    Args:
        graph_nodes (pd.DataFrame): A pandas DataFrame containing information about the nodes in the graph. It must include:
            'ID': A column with unique identifiers for each node.
            'Type': A column indicating the type of each node (origin, destin or network).
        graph_edges (pd.DataFrame): A pandas DataFrame containing information about the edges in the graph. It must include:
            'from': A column specifying the source node ID for each edge.
            'to': A column specifying the destination node ID for each edge.
            'weight': A column containing the weight (e.g., distance, cost) associated with each edge.
    Returns:
        result_df (pd.DataFrame): A pandas DataFrame with the following columns:
            'Origin': IDs of the origin nodes.
            'Destin': IDs of the destination nodes.
            'Weight': Shortest path between each origin and destination node.
    '''
    # Create dictionary mapping IDs to indexes
    id_to_index = {id_: index for index, id_ in enumerate(graph_nodes['ID'])}
    index_to_id = {index: id_ for index, id_ in enumerate(graph_nodes['ID'])}
    # Map links (edges) to indexes
    edges = [(id_to_index[from_], id_to_index[to]) for from_, to in zip(graph_edges['from'], graph_edges['to'])]
    weights = graph_edges['weight'].tolist()
    # Create the empty graph
    g = ig.Graph()
    # Add nodes
    g.add_vertices(len(graph_nodes))
    # Add edges with weights
    g.add_edges(edges)
    g.es['weight'] = weights
    # Specify an undirected graph
    g.to_undirected()
    # Define source and destination groups
    id_source_group = graph_nodes[graph_nodes['Type']=='origin']['ID'].to_list()
    id_target_group = graph_nodes[graph_nodes['Type']=='destin']['ID'].to_list()
    index_source_group = [id_to_index[i] for i in id_source_group]
    index_target_group = [id_to_index[i] for i in id_target_group]
    # Calculate the lengths of the shortest paths
    distances = g.distances(source=index_source_group, target=index_target_group, weights='weight')
    # Create lists to store results
    origins = []
    destins = []
    weight = []
    # Collect shortest path information
    for i, src in enumerate(index_source_group):
        for j, tgt in enumerate(index_target_group):
            path_length = distances[i][j]

            # if path_length < float('inf'):  # Verificar si hay un camino disponible
            #     origins.append(index_to_id[src])
            #     destins.append(index_to_id[tgt])
            #     weight.append(path_length)

            origins.append(index_to_id[src])
            destins.append(index_to_id[tgt])
            # Check if there is an available path
            if path_length == float('inf'):
                weight.append(-1)
            else:
                weight.append(path_length)


    # Create the DataFrame with the results
    result_df = pd.DataFrame({
        'Origin': origins,
        'Destin': destins,
        'Weight': weight
    })
    # Sort the DataFrame
    result_df = result_df.sort_values('Weight',ascending=False)

    return result_df


def contour_accessibility(OD_df: pd.DataFrame, threshold: int, weight_column='Weight', attr='relative') -> pd.DataFrame:
    '''
    Calculate contour accessibility for a determined threshold.

    Args:
        OD_df (pd.DataFrame): Origin-Destination DataFrame with columns 'Origin', 'Destin', and 'Weight'
        threshold (int): Travel time threshold for accessibility calculation
        weight_column (str): Name of the column where weights are stored
        attr (str): Type of attractiveness
            absolute: considers the attractiveness of each destination as 1
            relative: considers the attractiveness of each destination as 1/N, with N the total number of destinations

    Returns:
        acc_i_df (pd.DataFrame): DataFrame with 'Origin' and corresponding 'Acc_i' accessibility scores
    '''
    # Return empty accessibility DataFrame if the OD matrix is empty
    if OD_df.empty:
        return pd.DataFrame(columns=['Origin', 'Acc_i'])

    # Ensure 'Origin' and 'Destin' columns are strings
    OD_df['Origin'] = OD_df['Origin'].astype(str)
    OD_df['Destin'] = OD_df['Destin'].astype(str)

    # Filter according to the threshold
    OD_df_filtered = OD_df[OD_df[weight_column] < threshold]
    
    # Calculate attractiveness
    if attr == 'relative':
        unique_destins = pd.unique(OD_df_filtered['Destin'])
        atr_j_value = 1 / len(unique_destins)
    elif attr == 'absolute':
        atr_j_value = 1
    
    # Calculate accessibility for each origin
    acc_i = OD_df_filtered.groupby('Origin')['Destin'].apply(lambda x: x.nunique() * atr_j_value).reset_index()
    acc_i.columns = ['Origin', 'Acc_i']

    # Create a new DataFrame with all unique origins
    unique_origins = pd.DataFrame(pd.unique(OD_df['Origin']), columns=['Origin'])

    # Merge the unique origins DataFrame with the accessibility DataFrame
    acc_i_df = unique_origins.merge(acc_i, on='Origin', how='left')

    # Replace NaN accessibility values with 0
    acc_i_df['Acc_i'] = acc_i_df['Acc_i'].fillna(0)

    return acc_i_df


def weights_PCA(pca_df,percentage):

    # https://youtu.be/BiuwDI_BbWw?si=in2217gVLEJWIsMg
    
    # Standardized dataset
    df_stan = pca_df
    # Compute the covariance matrix
    covariance_matrix = df_stan.cov()
    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Sort the eigenvectors by descending eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]
    # Store the information in DataFrames
    eigenvalues_df = pd.DataFrame(eigenvalues,index=[f'PC{i+1}' for i in range(len(eigenvalues))]).T
    eigenvectors_df = pd.DataFrame(eigenvectors,columns=eigenvalues_df.columns,index=[a[-3:] for a in pca_df.columns])
    # Scores
    scores = pd.DataFrame(np.array(df_stan)@eigenvectors,columns=eigenvalues_df.columns)
    # Variance
    explained_variance = scores.var().to_frame()
    explained_variance_ratio = explained_variance/explained_variance.sum()
    explained_variance_ratio_cumsum = explained_variance_ratio.cumsum()
    # Factor loadings
    FL = eigenvectors_df.copy()
    for i in FL.index:
        for j in FL.columns:
            FL.loc[i,j] = FL.loc[i,j]*np.sqrt(eigenvalues_df.loc[0,j])
    # Select the number of component that explained the desire variability
    n_PC = 0
    for i in explained_variance_ratio_cumsum.index:
        if explained_variance_ratio_cumsum.loc[i,0] > percentage:
            n_PC = int(i[-1])
            break
    filtered_FL = FL[[f'PC{i+1}' for i in range(n_PC)]]
    filtered_FL_squared = filtered_FL*filtered_FL
    exp_var = filtered_FL_squared.sum()
    exp_tot = exp_var/exp_var.sum()
    filtered_FL_squared_exp_var = (filtered_FL_squared/exp_var).round(2)
    # Calculate pre-weights as the FL_squared_exp_var for the component with more 
    # impact in each variable times the exp_tot of the component
    pre_weights = pd.DataFrame(index=FL.index,columns=[0])
    for i,j in filtered_FL_squared_exp_var.idxmax(axis=1).items():
        # pre_weights.loc[i,0] = FL_squared_exp_var.loc[i,j]*exp_tot.loc[j,0]
        pre_weights.loc[i,0] = filtered_FL_squared_exp_var.loc[i,j]*exp_tot.loc[j]
    # Calculate the weights
    weights = pre_weights.copy()
    weights = weights/weights.sum()
    weights = weights.rename(columns={0:'w'})
    
    return n_PC, explained_variance_ratio[:n_PC], weights


def extract_survey_information(data,method=''):
    
    
    # Auxiliar dictionary with the AHP values
    dic = {1:9.0,2:7.0,3:5.0,4:3.0,5:1.0,6:1/3,7:1/5,8:1/7,9:1/9}
    
    # Define the names of the criterias
    if method == 'criteria':
        names = ['soc','env','eco']
    else:
        names = ['edu','spo','cul','fin','par','hea']
    AHP_df = pd.DataFrame(columns=names,index=names)
    # Principal diagonal with ones
    for i in AHP_df.index:
        AHP_df.loc[i,i] = 1.0
    # Replace the large strings with their abreviations
    if method == 'criteria':
        data = data.replace(['Social','Ambiental','Económico'],names)
    else:
        data = data.replace(['Educación','Deportes','Cultura','Finanzas','Parques','Salud'],names)
    # Replace the numbers from excel to AHP values
    data = data.replace(dic)
    # Extract the information from the AHP answers
    for i in data.index:
        AHP_df.loc[data.loc[i,'one'],data.loc[i,'two']] = data.loc[i,'value']
    # Fill the others values
    for i in range(len(AHP_df)):
        for j in range(i+1, len(AHP_df)):
            AHP_df.iat[j, i] = 1 / AHP_df.iat[i, j]

    return AHP_df

def calculate_average(matrices, method='geometric'):
    if method == 'arithmetic':
        average_matrix = np.mean(matrices, axis=0)
    elif method == 'geometric':
        # Aplica el logaritmo natural a cada DataFrame
        log_dataframes = [df.map(np.log) for df in matrices]
        # Calcula la media de los valores logarítmicos
        mean_log_df = pd.concat(log_dataframes).groupby(level=0).mean()
        # Aplica la exponencial para obtener la media geométrica utilizando DataFrame.map
        average_matrix = mean_log_df.map(np.exp)

    reference_matrix = matrices[0]
    average_df = pd.DataFrame(average_matrix, index=reference_matrix.index, columns=reference_matrix.columns)
    average_df = average_df.apply(pd.to_numeric, errors='coerce')
    
    return average_df

def calculate_consistency_and_pre_weights(matriz):
    '''
    
    '''
    matriz_norm = matriz/matriz.sum()
    matriz_norm_sum_filas = matriz_norm.sum(axis=1)
    vector_prom = matriz_norm_sum_filas/len(matriz_norm_sum_filas)
    vector_fila_total = matriz@vector_prom
    vector_cociente = vector_fila_total/vector_prom
    lambda_max = vector_cociente.mean()
    n = len(matriz)
    CI = (lambda_max-n)/(n-1)
    RI_dict = {1: 0, 2: 0, 3: 0.52, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49} # españa
    # RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49} # google
    CR = CI/RI_dict[n]

    check = 0
    n = 1
    matriz_mult = matriz
    vectores_propios = pd.DataFrame()
    while check == 0:
        if n == 1:
            matriz_mult = matriz_mult@matriz_mult
            matriz_mul_sum_filas = matriz_mult.sum(axis=1)
            matriz_mul_sum_filas_norm = matriz_mul_sum_filas/matriz_mul_sum_filas.sum()
            vectores_propios[n] = np.round(matriz_mul_sum_filas_norm,4)
            n += 1
        else:
            matriz_mult = matriz_mult@matriz_mult
            matriz_mul_sum_filas = matriz_mult.sum(axis=1)
            matriz_mul_sum_filas_norm = matriz_mul_sum_filas/matriz_mul_sum_filas.sum()
            vectores_propios[n] = np.round(matriz_mul_sum_filas_norm,4)
            if (np.abs(vectores_propios[n]-vectores_propios[n-1])).sum() < 1e-6:
                check = 1
            n += 1
    vector_propio = vectores_propios[n-1]
    
    return CR, vector_propio
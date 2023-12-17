import numpy as np
import pandas as pd
import itertools as it
from scipy.spatial.distance import cdist
from geopy.distance import geodesic
from multiprocessing.dummy import Pool


def get_distance(on_node: pd.Series, to_node: pd.Series, projection: np.ndarray) -> float:
    if on_node['to_node_id'] == to_node['from_node_id']:
        return geodesic(tuple(projection), (on_node['end_lat'], on_node['end_lon'])).m
    elif on_node['from_node_id'] == to_node['to_node_id']:
        return geodesic(tuple(projection), (on_node['start_lat'], on_node['start_lon'])).m
    else:
        return 0.0


def gcs(string: str) -> pd.Series:
    '''Превращает строку координат в серию координат'''
    data = string.split()
    data = [float(data[2][:-1]), float(data[1][1:]), float(data[4][:-1]), float(data[3])]
    return pd.Series(data)


def load_nodes(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="windows-1251").set_index('link_id')
    df[['start_lat', 'start_lon', 'end_lat', 'end_lon']] = df['geometry'].apply(gcs)
    df = df[['from_node_id', 'to_node_id', 'length', 'start_lat', 'start_lon', 'end_lat', 'end_lon']]
    return df


def load_stations(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="windows-1251", sep=';')
    df['km'] = df['km'] + df['pk']/10
    df = df.drop(columns=['pk', 'esr6']).rename(columns={'Участок': 'way'})
    return df


def BFS_SP(start: int, end: int, nodes: pd.DataFrame) -> list:
    explored = []
    queue = [[start]]
    if start == end:
        return [start]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node not in explored:
            neighbours = nodes.loc[nodes['from_node_id'] == node]['to_node_id'].to_numpy()
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                if neighbour == end:
                    return new_path
            explored.append(node)
    return


def get_road(path: list, nodes: pd.DataFrame) -> list:
    road = []
    flag = False
    for i in range(len(path)):
        if flag:
            road.append(buf.loc[buf['to_node_id'] == path[i]].index[0])
            flag = False
        x = nodes.loc[nodes['from_node_id'] == path[i]][['from_node_id', 'to_node_id']]
        if len(x) > 1:
            buf = x
            flag = True
        elif len(x) == 0:
            return road
        else:
            road.append(x.index[0])
    return road


def find_route(start: list, end: list, nodes: pd.DataFrame) -> (tuple, float, np.ndarray):
    if start == [None] or end == [None]:
        return None, None, None
    combinations = list(it.product(start, end))
    dists = []
    paths = []
    for route in combinations:
        print(route[0][0])
        print(route[1][0])
        first = nodes.at[route[0][0], 'to_node_id']
        second = nodes.at[route[1][0], 'to_node_id']
        path = BFS_SP(first, second, nodes)
        if path == None:
            dists.append(999999.0)
            paths.append(None)
            continue
        path = get_road(path, nodes)
        if route[0][0] != path[0]:
            path.insert(0, route[0][0])
        if route[1][0] == path[-2]:
            path = path[:-1]
        paths.append(np.array(path, dtype=int))
        start_distance = get_distance(nodes.loc[path[0]], nodes.loc[path[1]], route[0][1])/1000
        end_distance =  get_distance(nodes.loc[path[-1]], nodes.loc[path[-2]], route[1][1])/1000
        d = start_distance + end_distance
        d += nodes.loc[path[1:-1]]['length'].sum()/1000
        dists.append(d)
    dists = np.array(dists)
    if np.all(dists == 999999.0):
        return None, None, None
    shortest = np.argmin(dists)
    return combinations[shortest], dists[shortest], paths[shortest].astype(int)


def find_nearest_nodes(points: pd.DataFrame | pd.Series | list | np.ndarray | tuple, nodes: pd.DataFrame, try_number = 100, max_proj_distance_m = 50, max_search_distance_m = 100) -> list:    
    start_coords = nodes[['start_lat', 'start_lon']].to_numpy()
    end_coords = nodes[['end_lat', 'end_lon']].to_numpy()
    dxy = end_coords - start_coords
    det = dxy[:,0]*dxy[:,0] + dxy[:,1]*dxy[:,1]

    def build_projections(point: np.ndarray, inds_nodes: np.ndarray = np.arange(len(nodes))) -> np.ndarray:
        point = np.repeat(point, len(inds_nodes), axis=0)
        x = dxy[inds_nodes, 0]*(point[:, 0]-start_coords[inds_nodes, 0])
        y = dxy[inds_nodes, 1]*(point[:, 1]-start_coords[inds_nodes, 1])
        a = (x + y)/det[inds_nodes]
        x = start_coords[inds_nodes, 0] + (a * dxy[inds_nodes, 0])
        y = start_coords[inds_nodes, 1] + (a * dxy[inds_nodes, 1])
        projections = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        return projections
    
    def check_on_node(node_index: int, projection: tuple) -> bool:
        metres_start = geodesic(tuple(start_coords[node_index]), projection).m
        metres_end = geodesic(tuple(end_coords[node_index]), projection).m
        metres_between = geodesic(tuple(start_coords[node_index]), tuple(end_coords[node_index])).m
        if metres_start < metres_between and metres_end < metres_between:
            return True
        return False

    def check_distance(point: np.ndarray, destination: np.ndarray, max_distance_m: int) -> bool:
        if geodesic(tuple(point), tuple(destination)).m <= max_distance_m:
            return True
        return False
    
    single_flag = False
    if isinstance(points, pd.DataFrame):
        points = points[['lat', 'lon']].to_numpy()
    elif isinstance(points, pd.Series):
        points = points[['lat', 'lon']].to_numpy().reshape(1, -1)
        single_flag = True
    elif isinstance(points, (list, np.ndarray)):
        if isinstance(points[0], (int, float)):
            points = np.array(points).reshape(1, -1)
            single_flag = True
        else:
            points = np.array(points)
    elif isinstance(points, tuple):
        points = np.array(points).reshape(1, -1)
        single_flag = True
    
    nearest_nodes = []
    for i in range(points.shape[0]):
        p = points[i].reshape(1, -1)
        below_max = []

        dists_start = cdist(p, start_coords, 'euclidean')[0]
        indexes_start = list(np.argsort(dists_start)[:try_number])
        indexes_start = list(filter(lambda i: check_distance(p[0], start_coords[i], max_search_distance_m), indexes_start))
        projections_start = build_projections(p, indexes_start)
        for i, proj in zip(indexes_start, projections_start):
            if check_on_node(i, tuple(proj)) and check_distance(p[0], proj, max_proj_distance_m):
                below_max.append([i, proj])

        dists_end = cdist(p, end_coords, 'euclidean')[0]
        indexes_end = list(np.argsort(dists_end)[:try_number])
        indexes_end = list(filter(lambda i: check_distance(p[0], end_coords[i], max_search_distance_m), indexes_end))
        projections_end = build_projections(p, indexes_end)
        for i, proj in zip(indexes_end, projections_end):
            if check_on_node(i, proj) and check_distance(p[0], proj, max_proj_distance_m):
                below_max.append([i, proj])
        
        if len(below_max) == 0:
            nearest_nodes.append(find_nearest_nodes(p, nodes, try_number*2, max_proj_distance_m*2, max_search_distance_m*2))
            continue
        elif len(below_max) == 1:
            nearest_nodes.append(below_max)
            continue
        else:
            below_max = [[key, value] for key, value in dict(below_max).items()] 
            nearest_nodes.append(below_max)
    if single_flag:
        return nearest_nodes[0]
    else:
        return nearest_nodes
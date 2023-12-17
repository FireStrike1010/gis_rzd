import numpy as np
import pandas as pd
import folium
from geopy.distance import geodesic
from process_functions import *
import pickle

class Way:
    def __init__(self, way: pd.DataFrame, nodes: pd.DataFrame = None, name: str = None, process: bool = False, logging: bool = False) -> None:
        if name == None:
            name = way['way'].mode()
            if len(name) == 0:
                self.name = ''
                if logging:
                    print("Лучше задайте имя маршрута")
            else:
                self.name = str(name[0])
        else:
            self.name = str(name)
        self.way = way.copy().drop(columns=['way']).sort_values(by='km').reset_index()
        self.quantity = len(self.way)
        self.length = self.way.at[self.quantity-1, 'km'] - self.way.at[0, 'km']
        self.bounds = self._get_bounds()
        self.mean_point = self._get_mean_point()
        self.nearest_nodes = None
        self.graph = None
        self.distances = None
        self.routes = None
        self.difs = None
        self._nodes = nodes
        self._selected_nodes_flag = False
        self._processed = False
        self._logging = logging
        if process:
            self._build_graph()

    def _build_graph(self, nodes: pd.DataFrame = None, make_nodes_copy: bool = True) -> None:
        if nodes == None:
            if isinstance(self._nodes, pd.DataFrame):
                nodes = self._nodes
            else:
                if self._logging:
                    print(f'{self.name} --- Не существует привязки к нодам (карте железных дорог)')
                return
        if self._logging:
            print(f'{self.name} --- Началась обработка')
        graph = [(None, None)]
        distances = [0]
        routes = [np.array([], dtype=int)]
        if self.nearest_nodes == None:
            self._find_nearest_nodes(nodes)
        for i in range(self.quantity-1):
            if self._logging:
                print(f'Постройка маршрута {i+1}/{self.quantity} станций')
            comb, dist, route = find_route(self.nearest_nodes[i], self.nearest_nodes[i+1], self._nodes)
            graph.append(comb)
            distances.append(dist)
            routes.append(route)
        self.graph = graph
        self.distances = np.array(distances)
        self.routes = routes
        self.difs = np.array(((self.way['km'] - self.way['km'].shift(1)).fillna(0) - self.distances))
        self._processed = True
        if self._logging:
            print(f'{self.name} --- Общая погрешность в маршруте: {np.sum(np.absolute(self.difs))} км.')
            print(f'{self.name} --- Обработка завершена')
        if make_nodes_copy:
            self._copy_self_nodes(self.routes, nodes)
    
    def _find_nearest_nodes(self, nodes: pd.DataFrame = None, try_number = 10, max_proj_distance_m = 100, max_search_distance_m = 200) -> list:
        if not isinstance(nodes, pd.DataFrame):
            if isinstance(self._nodes, pd.DataFrame):
                nodes = self._nodes
            else:
                if self._logging:
                    print(f'{self.name} --- Не существует привязки к нодам (карте железных дорог)')
                return
        for i, row in self.way.iterrows():
            self.nearest_nodes = find_nearest_nodes(self.way, nodes, try_number, max_proj_distance_m, max_search_distance_m)
        if self._logging:
            print(f'{self.name} --- Привязка к ближайшим нодам осуществлена')
    def _copy_self_nodes(self, nodes_indexes: list | np.ndarray = None, nodes: pd.DataFrame = None) -> None:
        if not isinstance(nodes, pd.DataFrame):
            if isinstance(self._nodes, pd.DataFrame):
                nodes = self._nodes
            else:
                if self._logging:
                    print(f'{self.name} --- Не существует привязки к нодам (карте железных дорог)')
                return
        if nodes_indexes == None:
            self._nodes = nodes.copy()
        else:
            nodes_indexes = np.hstack(nodes_indexes)
            try:
                nodes_indexes = np.unique(nodes_indexes)
            except:
                nodes_indexes = nodes_indexes.astype(np.float32)
                nodes_indexes = nodes_indexes[~np.isnan(nodes_indexes)]
                nodes_indexes = np.unique(nodes_indexes)
            self._nodes = nodes.iloc[nodes_indexes].copy()
            self._selected_nodes_flag = True
        if self._logging:
            print(f'{self.name} --- Копия нодов сохранена')

    def _get_bounds(self, cap: float | int = 0) -> tuple:
        return (round(self.way['lat'].min()-cap, 8), round(self.way['lat'].max()+cap, 8), round(self.way['lon'].min()-cap, 8), round(self.way['lon'].max()+cap, 8))
    
    def _get_mean_point(self) -> tuple:
        return (self.way['lat'].mean(), self.way['lon'].mean())
        
    def plot(self, m: folium.Map = None, center_point: tuple = None, zoom=8) -> folium.Map:
        nodes = self._nodes
        if not isinstance(m, folium.Map):
            if center_point == None:
                center_point = self.mean_point
            m = folium.Map(center_point, zoom_start=zoom, tiles='openstreetmap')
        for _, row in self.way.iterrows():
            folium.Marker(location=[row['lat'], row['lon']],
                    tooltip=row['name'],
                    popup=row['name']+' - '+self.name+' - '+str(row['km'])+' км',
                    icon=folium.Icon(color="green")
                    ).add_to(m)
        if self._selected_nodes_flag:
            for index, row in nodes.iterrows():
                folium.PolyLine([(row['start_lat'], row['start_lon']), (row['end_lat'], row['end_lon'])], tooltip=str(index)+' - '+str(row['length'])+' м').add_to(m)
        elif isinstance(nodes, pd.DataFrame) and isinstance(self.routes, list):
            for path in self.routes[1:]:
                for index, row in nodes.loc[nodes.index.isin(path)].iterrows():
                    folium.PolyLine([(row['start_lat'], row['start_lon']), (row['end_lat'], row['end_lon'])], tooltip=str(index)+' - '+str(row['length'])+' м').add_to(m)
        return m
    
    def _plot_find_coords(self, point: dict) -> folium.Map:
        m = self.plot(None, tuple(point['coords']), zoom=20)
        tooltip = str(round(point['km_p'], 1)) + ' км.пикет'
        popup = f"""
        <h5>{point['name']}\n</h5>
        <h6>{round(point['km_p'], 1)} км.пикет\n</h6>
        <h6>Координаты:\n{list(point['coords'])}\n</h6>
        <h6>Погрешность: {int(point['diameter']*500)} м.\n</h6>
        <table border='1px solid black'>
        <tr>
            <th border='1px solid black'>Станция</th>
            <th border='1px solid black'>Расстояние в км</th>
        </tr>
        """
        stations = sorted(point['stations'], key=lambda x: x[1])
        for i in stations:
            popup += f"""
            <tr>
                <td border='1px solid black'>{i[0]}</td>
                <td border='1px solid black'>{round(i[1], 3)}</td>
            </tr>"""
        popup += '</table>'
        if point['diameter']*500 <= 10:
            folium.Marker(tuple(point['coords']),
                          tooltip=tooltip,
                          popup=popup,
                          icon=folium.Icon(color='red')
                          ).add_to(m)
        else:
            folium.Circle(tuple(point['coords']),
                          radius=point['diameter']*500,
                          color='red',
                          fill_color='lightred',
                          tooltip=tooltip,
                          popup=popup
                          ).add_to(m)
        return m
    
    def find_coords(self, km_p: float, return_map = True) -> (dict, folium.Map):
        ans = dict()
        ans['name'] = self.name
        ans['km_p'] = km_p
        km = self.way['km'].to_numpy(dtype=float)
        if km_p < km[0]:
            ans['coords'] = list(map(float, self.way.iloc[0][['lat', 'lon']].to_list()))
            ans['diameter'] = (km[0] - km_p)*2
            ans['stations'] = [[self.way.at[0, 'name'], round(km[0] - km_p, 3)]]
            if self._logging:
                print(f'{self.name} --- Точка {km_p} находится за рамками маршрута')
            if return_map:
                return (ans, self._plot_find_coords(ans))
            return (ans, None)
        elif km_p > km[-1]:
            ans['coords'] = list(map(float, self.way.iloc[-1][['lat', 'lon']].to_list()))
            ans['diameter'] = (km_p - km[-1])*2
            ans['stations'] = [[self.way.at[self.quantity-1, 'name'], round(km_p - km[-1], 3)]]
            if self._logging:
                print(f'{self.name} --- Точка {km_p} находится за рамками маршрута')
            if return_map:
                return (ans, self._plot_find_coords(ans))
            return (ans, None)
        for i in range(self.quantity):
            if km_p <= km[i]:
                if km_p == km[i]:
                    ans['coords'] = list(map(float, self.way.iloc[i][['lat', 'lon']].to_list()))
                    ans['diameter'] = 0
                    ans['stations'] = [[self.way.at[i, 'name'], 0]]
                    if return_map:
                        return (ans, self._plot_find_coords(ans))
                    return (ans, None)
                break
        nodes = self._nodes
        ans['stations'] = [[self.way.at[i-1, 'name'], round(km_p - km[i-1], 3)], [self.way.at[i, 'name'], round(km[i] - km_p, 3)]]
        diameter = abs(self.difs[i])
        if not isinstance(self.routes[i], np.ndarray):
            if km_p - km[i-1] < km[i] - km_p:
                ans['diameter'] = (km_p - km[i-1])*2
                lat = self.way.at[i-1, 'lat']
                lon = self.way.at[i-1, 'lon']
            else:
                ans['diameter'] = (km[i] - km_p)*2
                lat = self.way.at[i, 'lat']
                lon = self.way.at[i, 'lon']
            ans['coords'] = list(map(float, [round(lat, 8), round(lon, 8)]))
            if return_map:
                return (ans, self._plot_find_coords(ans))
            return (ans, None)
        if km_p - km[i-1] < km[i] - km_p:
            route = self.routes[i]
            km_p = km_p - km[i-1]
            diameter *= (km_p/((km[i]-km[i-1])/2))
            km_p *= 1000
            projection = list(filter(lambda x: x[0] == route[0], self.nearest_nodes[i-1]))[0][1]
        else:
            route = self.routes[i][::-1]
            km_p = km[i] - km_p
            diameter *= (km_p/((km[i]-km[i-1])/2))
            km_p *= 1000
            projection = list(filter(lambda x: x[0] == route[0], self.nearest_nodes[i]))[0][1]
        cumsum = get_distance(nodes.loc[route[0]], nodes.loc[route[1]], projection)
        ans['diameter'] = diameter
        for index, node in enumerate(route[1:]):
            if cumsum > km_p:
                break
            cumsum += nodes.at[node, 'length']
        node = nodes.loc[route[index]]
        prev = nodes.loc[route[index-1]]
        cumsum -= node['length']
        distance = km_p - cumsum
        prop = distance/geodesic((node['start_lat'], node['start_lon']), (node['end_lat'], node['end_lon'])).m
        if prev['to_node_id'] == node['from_node_id']:
            lat = node['start_lat'] + (node['end_lat'] - node['start_lat'])*prop
            lon = node['start_lon'] + (node['end_lon'] - node['start_lon'])*prop
        else:
            lat = node['end_lat'] + (node['start_lat'] - node['end_lat'])*prop
            lon = node['end_lon'] + (node['start_lon'] - node['end_lon'])*prop
        ans['coords'] = list(map(float, [round(lat, 8), round(lon, 8)]))
        if return_map:
            return (ans, self._plot_find_coords(ans))
        return (ans, None)
    
    def __str__(self) -> str:
        return self.name
    
    def _repr_html_(self) -> str:
        return self.way._repr_html_()


def save_way(way: Way, filepath: str = None) -> None:
    if filepath == None:
        filepath = 'PickleWays/'
        filepath += way.name.replace(' ', '_').replace('~', '-').replace('.', '')
        filepath += '.pickle'
    with open(filepath, 'wb') as f:
        pickle.dump(way, f)


def open_way(filepath: str) -> Way:
    with open(filepath, 'rb') as f:
        way = pickle.load(f)
        way._logging = False
    return way
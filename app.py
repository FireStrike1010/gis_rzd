from flask import Flask, render_template, redirect, url_for, request, jsonify, make_response
from gevent.pywsgi import WSGIServer
from gevent import monkey
from way import *

def init_ways(folder_path: str = 'PickleWays'):
    from os import walk, path
    paths = []
    ways = {}
    for root, _, files in walk(folder_path):
        for file in files:
            if len(file) > 7 and file[-7:] == '.pickle':
                paths.append(path.join(root, file))
    for i, p in enumerate(paths):
        var_name = 'way' + str(i)
        globals()[var_name] = open_way(p)
        name = globals()[var_name].name
        ways[name] = globals()[var_name]
    return ways


class Page:
    def __init__(self, ways: dict):
        self.ways = ways
        self.last_way = 'Выберите участок'
        self.ways_name = list(self.ways.keys())
        self.select_drop = ['Выберите участок'] + sorted(self.ways_name)
        self.km = str(0)
        self.p = str(0)
        self.map = ''
        self.point = {'name': None, 'km_p': None, 'coords': [None, None], 'str_coords': '', 'diameter': None, 'stations': None}
        self.created_point = False
        self.show_map_point = True
        self.map_links = {'yandex': None, 'google': None, 'osm': None}
    
    def create_map_links(self, point: dict) -> None:
        coords = point['coords']
        self.map_links['yandex'] = f"https://yandex.ru/maps/?pt={coords[1]},{coords[0]}&z=15&l=map"
        self.map_links['google'] = f"http://www.google.com/maps/place/{coords[0]},{coords[1]}"
        self.map_links['osm'] = f"http://www.openstreetmap.org/?mlat={coords[0]}&mlon={coords[1]}&zoom=12"
    
    def update_page(self, way_name, km, p, show_map_point) -> None:
        self.last_way = way_name
        self.km = str(km)
        self.p = str(p)
        self.show_map_point = show_map_point
        if way_name == 'Выберите участок':
            return
        if 'Выберите участок' in self.select_drop:
            self.select_drop.remove('Выберите участок')
        index = self.select_drop.index(self.last_way)
        self.select_drop[0], self.select_drop[1:] = self.select_drop[index], self.select_drop[:index]+self.select_drop[index+1:]

    def find_point(self) -> None:
        if self.last_way == 'Выберите участок':
            return
        km_p = int(self.km) + int(self.p)/10
        way = self.ways[self.last_way]
        self.point, map = way.find_coords(km_p, self.show_map_point)
        if self.show_map_point:
            self.map = map._repr_html_()
        self.create_map_links(self.point)
        self.point['str_coords'] = f"{self.point['coords'][1]} {self.point['coords'][0]}"
        self.created_point = True
    
    def render_way_map(self) -> None:
        if self.last_way == 'Выберите участок':
            return
        self.map = self.ways[self.last_way].plot()._repr_html_()
    
    def render_point_map(self) -> None:
        if self.last_way == 'Выберите участок':
            return
        self.map = self.ways[self.last_way]._plot_find_coords(self.point)._repr_html_()

ways = init_ways('../PickleWays')

app = Flask(__name__)
app.json.ensure_ascii = False


@app.route('/', methods=['GET', 'POST'])
def main():
    page = Page(ways)
    if request.method == 'POST':
        way_name = request.form.get('way')
        km = int(request.form['km'])
        p = int(request.form['p'])
        show_map = bool(request.form.get('show_map'))
        page.update_page(way_name, km, p, show_map)
        if 'search_km_p' in request.form:
            page.find_point()
            if not show_map:
                page.render_way_map()
        elif 'show_way' in request.form:
            page.created_point = False
            page.render_way_map()
        return render_template('index.html', page=page)
    return render_template('index.html', page=page)

@app.route('/about_us/')
def about_us():
    return render_template('about_us.html')

@app.route('/about_api/')
def about_api():
    return render_template('about_api.html')

@app.route('/api/')
def api_all_ways():
    ways_names = list(ways.keys())
    return make_response(jsonify(ways_names))

@app.route('/api/<way_name>/')
def api_way(way_name):
    way = ways.get(way_name)
    response = []
    if not isinstance(way, Way):
        return make_response(jsonify(response))
    for _, i in way.way.iterrows():
        response.append(i[['name', 'km', 'lat', 'lon']].to_list())
    return make_response(jsonify(response))

@app.route('/api/<way_name>/<km_p>/')
def api_find_coords(way_name, km_p):
    way = ways.get(way_name)
    response = {}
    if not isinstance(way, Way):
        return make_response(jsonify(response))
    try:
        km_p = float(km_p)
    except:
        return make_response(jsonify(response))
    response, _ = way.find_coords(km_p, False)
    return make_response(jsonify(response))


monkey.patch_all(ssl=False)
if __name__ == '__main__':
    gevent_server = WSGIServer(('localhost', 80), app)
    gevent_server.serve_forever()

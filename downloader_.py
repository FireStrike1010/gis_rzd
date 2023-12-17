import sys
from os import walk, remove, rmdir
import requests

osm_dir = './osmfile'
csv_dir = './csvfile'

def download(area_name: str = None, bbox: tuple = None) -> None:
    global osm_dir
    if requests.get('https://overpass-api.de').status_code != 200:
        print('Unable to connect to OpenRailwayMap')
        exit()
    if bbox == None:
        download_osm_data_from_overpass(subarea_names=area_name, download_dir=osm_dir)
    else:
        download_osm_data_from_overpass(bboxs=bbox, download_dir=str(osm_dir+'/'+area_name))

def process(area_name: str, bbox: tuple = None, delete_osm: bool = False) -> None:
    global osm_dir, csv_dir
    output_folder = csv_dir + '/' + area_name
    if bbox == None:
        area_name = osm_dir + '/' + area_name
    else:
        area_name = osm_dir + '/' + area_name + '/' + 'map_1.osm'
    if '.osm' not in area_name:
        area_name += '.osm'
    net = get_network_from_file(filename=area_name, check_boundary=True)
    save_network(net, output_folder=output_folder)
    if delete_osm:
        remove(area_name)
        area_name = area_name[:len(area_name)-area_name[::-1].index('/')-1]
        rmdir(area_name)

def install(area_name: str = None, bbox: tuple = None, delete_osm: bool = True) -> None:
    last_filename = get_last_filename() + 1
    area_name = 'map_'+str(last_filename)
    download(area_name, bbox)
    process(area_name, bbox, delete_osm)

def get_last_filename() -> int:
    global osm_dir
    def map_filter(filename: str) -> bool:
        if 'map_' in filename:
            return True
        else:
            return False
    for root, dirs, files in walk(osm_dir):
        files = list(filter(map_filter, list(files)))
        files = [int(i[4]) for i in files]
        if len(files) == 0:
            return 0
        else:
            return max(files)

args = sys.argv[1:]
if len(args) == 0:
    print('...Try "install area_name" or use "help"...')
    exit()

if args[0] in ('-h', 'help'):
    print('''-l; -list --- show list of downloaded ".osm" files (maps) and ".csv" files (processed tables)
-h; help --- print instructions
download --- download railway nodes (".osm" map) from OpenRailwayMap. Example: download Kaliningrad
process --- recompile railway nodes (".osm" map) to processed table (".csv" table). Example: process ./osmfile/Kaliningrad.osm
install --- download + process. Example: install Kalinigrad''')
    exit()

elif args[0] in ('-l', '-list'):
    print('--- Downloaded railway nodes ---')
    for root, dirs, files in walk(osm_dir):
        for file in files:
            print(file)
    print('--- Installed railway nodes ---')
    for root, dirs, files in walk(csv_dir):
        for dir in dirs:
            print(dir)

elif args[0] in ('install', 'download', 'process'):
    command = args[0]
    args = args[1:]
    try:
        args = tuple(map(float, args))
        bound_methot = True
    except:
        bound_methot = False
    fail_names = []
    if len(args) == 0:
        print('Type area name')
        exit()

    if bound_methot and len(args) % 4 != 0:
        print('Type all 4 numbers')
        exit()

    if command == 'download':
        from osm2rail.download_by_Overpass import download_osm_data_from_overpass
        if bound_methot:
            last_filename = get_last_filename()
            for _ in range(len(args) // 4):
                bounds = args[:3]
                last_filename += 1
                args = args[4:]
                area_name = 'map_'+str(last_filename)
                try:
                    download(area_name=area_name, bbox=tuple(bounds))
                except:
                    fail_names.append(tuple(bounds))
        else:
            for i in args:
                try:
                    download(area_name=i)
                except:
                    fail_names.append(i)

    if command == 'process':
        from osm2rail.network import get_network_from_file
        from osm2rail.writer import save_network
        for i in args:
            if 1:
                process(i)
            else:
                fail_names.append(i)

    if command == 'install':
        from osm2rail.download_by_Overpass import download_osm_data_from_overpass
        from osm2rail.network import get_network_from_file
        from osm2rail.writer import save_network
        if bound_methot:
            last_filename = get_last_filename()
            for _ in range(len(args) // 4):
                bounds = tuple(map(lambda x: round(x, 7), args[:4]))
                last_filename += 1
                args = args[4:]
                area_name = 'map_'+str(last_filename)
                try:
                    download(area_name=area_name, bbox=bounds)
                    print(f'Processing "{area_name}.osm"...')
                    process(area_name=area_name, bbox=bounds, delete_osm=True)
                    print('Done.')
                except:
                    fail_names.append(tuple(bounds))
        else:
            for i in args:
                try:
                    download(i)
                    process(i)
                except:
                    fail_names.append(i)

    if len(fail_names) != 0:
        print(f"Failed to {args}:")
        print('\n'.join(fail_names))
    print('...fin...')
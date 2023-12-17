from app import *
from way import *
from gevent.pywsgi import WSGIServer
from gevent import monkey

monkey.patch_all()
if __name__ == '__main__':
    gevent_server = WSGIServer(('0.0.0.0', 5000), app)
    gevent_server.serve_forever()
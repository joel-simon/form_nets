import json
from flask import Flask
from evolution import make_formnets
nets = make_formnets()
app = Flask(__name__, static_folder='public', static_url_path='')

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/net/<int:nid>', methods=['GET'])
def get_net(nid):
    return json.dumps(nets[nid].toJSON())

if __name__ == '__main__':
    app.run(debug=True)

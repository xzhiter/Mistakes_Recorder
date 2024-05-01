import base64

import requests

from AuthV3Util import add_auth_params

APP_KEY = ""
APP_SECRET = ""


def set_key(app_key, app_secret):
    global APP_KEY, APP_SECRET
    APP_KEY = app_key
    APP_SECRET = app_secret


def create_request(path):
    q = read_file_as_base64(path)
    data = {'q': q, 'angle': "0"}
    params = add_auth_params(APP_KEY, APP_SECRET, data)
    params.update(data)
    res = do_call('https://openapi.youdao.com/ocr_writing_erase', params)
    return base64.b64decode(res.json()["eraseEnhanceImg"])


def do_call(url, params):
    response = requests.post(url, params)
    return response


def read_file_as_base64(path):
    f = open(path, 'rb')
    data = f.read()
    return str(base64.b64encode(data), 'utf-8')

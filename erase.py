import base64

import requests

from AuthV3Util import add_auth_params

APP_KEY = ""
APP_SECRET = ""


class APIError(Exception):
    def __init__(self, value):
        self.value = value


def set_key(app_key, app_secret):
    global APP_KEY, APP_SECRET
    APP_KEY = app_key
    APP_SECRET = app_secret


def create_request(file):
    q = read_file_as_base64(file)
    data = {'q': q, 'angle': "0"}
    params = add_auth_params(APP_KEY, APP_SECRET, data)
    params.update(data)
    res = do_call('https://openapi.youdao.com/ocr_writing_erase', params)
    try:
        return base64.b64decode(res.json()["eraseEnhanceImg"])
    except KeyError:
        text = f"服务器返回了错误的信息。\n详细信息：{res.text}"
        raise APIError(text)


def do_call(url, params):
    response = requests.post(url, params)
    return response


def read_file_as_base64(file):
    data = file.read()
    return str(base64.b64encode(data), 'utf-8')

import hashlib
import time
import uuid


def add_auth_params(app_key, app_secret, q):
    salt = str(uuid.uuid1())
    curtime = str(int(time.time()))
    sign = calculate_sign(app_key, app_secret, q, salt, curtime)
    params = {'appKey': app_key,
              'salt': salt,
              'curtime': curtime,
              'signType': 'v3',
              'sign': sign}
    return params


def calculate_sign(app_key, app_secret, q, salt, curtime):
    str_src = app_key + get_input(q["q"]) + salt + curtime + app_secret
    return encrypt(str_src)


def encrypt(str_src):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(str_src.encode('utf-8'))
    return hash_algorithm.hexdigest()


def get_input(input_):
    if input_ is None:
        return input_
    input_len = len(input_)
    return input_ if input_len <= 20 else input_[0:10] + str(input_len) + input_[input_len - 10:input_len]

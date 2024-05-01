import hashlib
import time
import uuid


def add_auth_params(appKey, appSecret, q):
    salt = str(uuid.uuid1())
    curtime = str(int(time.time()))
    sign = calculate_sign(appKey, appSecret, q, salt, curtime)
    params = {'appKey': appKey,
              'salt': salt,
              'curtime': curtime,
              'signType': 'v3',
              'sign': sign}
    return params


def calculate_sign(appKey, appSecret, q, salt, curtime):
    strSrc = appKey + get_input(q["q"]) + salt + curtime + appSecret
    return encrypt(strSrc)


def encrypt(strSrc):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(strSrc.encode('utf-8'))
    return hash_algorithm.hexdigest()


def get_input(input):
    if input is None:
        return input
    input_len = len(input)
    return input if input_len <= 20 else input[0:10] + str(input_len) + input[input_len - 10:input_len]

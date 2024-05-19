import os

if __name__ == "__main__":
    os.system("pyinstaller -w main.py --hidden-import \"scipy.special._cdflib\" --hidden-import \"numpy.lib\" --hidden-import \"email.mime\" --hidden-import  \"email.mime.text\" --hidden-import \"Cryptodome\"")
    os.mkdir("./dist/main/_internal/cnocr")
    os.system("copy \".\\label_cn.txt\" \".\\dist\\main\\_internal\\cnocr\"")
    input()

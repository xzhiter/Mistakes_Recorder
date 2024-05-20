import os

if __name__ == "__main__":
    os.system("pyinstaller -w main.py --hidden-import "
              "\"numpy.libs\" --hidden-import \"email.mime\" --hidden-import  \"email.mime.text\" --hidden-import "
              "\"Cryptodome.Cipher.AES\" --hidden-import \"smtplib\"")
    input()

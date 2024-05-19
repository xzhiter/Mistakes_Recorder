import smtplib
from email.mime.text import MIMEText

from Cryptodome.Cipher import AES

def send(text):
    """已屏蔽涉及开发者隐私的内容"""
    send_address = "xxx@xxx.com"
    my_cipher = AES.new(b"xxxxxxxxxxxxxxxx", AES.MODE_ECB)
    password = str(my_cipher.decrypt(b'xxxxxxxxxxxxxxxx'), 'utf-8')
    server = smtplib.SMTP_SSL('smtp.xxx.com', 465)
    server.login(send_address, password)
    content = text
    msg = MIMEText(content, 'plain', 'utf-8')
    msg['From'] = 'xxx<xxx@xxx.com>'
    msg['To'] = 'xxx<xxx@xxx.com>'
    msg['Subject'] = 'feedback'
    server.sendmail(send_address, ['xxx@xxx.com'], msg.as_string())

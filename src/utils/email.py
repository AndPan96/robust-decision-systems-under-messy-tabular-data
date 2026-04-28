import smtplib
import os
from email.mime.text import MIMEText
from typing import cast

def send_email(subject: str, body: str):

    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("EMAIL_RECEIVER")

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = cast(str, sender)
    msg["To"] = cast(str, receiver)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:

        server.login(cast(str, sender), cast(str, password))
        server.send_message(msg)


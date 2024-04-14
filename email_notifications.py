import smtplib
from config import EMAIL_USER, EMAIL_PASS, RECEIVER_EMAIL

def send_email(subject, content):
    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.starttls()
    smtp_server.login(EMAIL_USER, EMAIL_PASS)
    message = f"Subject: {subject}\n\n{content}"
    smtp_server.sendmail(EMAIL_USER, RECEIVER_EMAIL, message)
    smtp_server.quit()
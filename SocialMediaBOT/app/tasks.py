from app import celery, mail
from flask_mail import Message

@celery.task
def send_notification(email, subject, message):
    msg = Message(subject, sender='your_email@gmail.com', recipients=[email])
    msg.body = message
    mail.send(msg)
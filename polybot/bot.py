import telebot
import collections
from loguru import logger
import os
import time
from telebot.types import InputFile
from img_proc import Img
import requests
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

class Bot:
    def __init__(self, token, telegram_chat_url, s3_bucket, yolo_service_url):
        self.telegram_bot_client = telebot.TeleBot(token)
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60, certificate=open('/app/YOURPUBLIC.pem', 'r'))
        self.s3_bucket = s3_bucket
        self.yolo_service_url = yolo_service_url
        self.s3 = boto3.client('s3')

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def upload_to_s3(self, file_path, s3_key):
        try:
            self.s3.upload_file(file_path, self.s3_bucket, s3_key)
            logger.info(f'Uploaded {file_path} to S3 bucket {self.s3_bucket}')
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error(f"S3 upload error: {e}")
            raise

    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')
        try:
            if not self.is_current_msg_photo(msg):
                self.send_text(msg['chat']['id'], 'Please send a photo for processing.')
                return

            img_path = self.download_user_photo(msg)
            s3_key = os.path.basename(img_path)
            self.upload_to_s3(img_path, s3_key)

            response = requests.post(f'{self.yolo_service_url}/predict', params={'imgName': s3_key})
            response.raise_for_status()
            prediction_summary = response.json()

            # Format the prediction summary
            detected_objects = collections.Counter(obj['class'] for obj in prediction_summary['labels'])
            formatted_summary = ', '.join([f"{obj}: {count}" for obj, count in detected_objects.items()])
            result_text = f"Objects detected: {formatted_summary}"

            self.send_text(msg['chat']['id'], result_text)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.send_text(msg['chat']['id'], "Something went wrong... please try again.")

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

class ImageProcessingBot(Bot):
    def __init__(self, token, telegram_chat_url, s3_bucket, yolo_service_url):
        super().__init__(token, telegram_chat_url, s3_bucket, yolo_service_url)

class QuoteBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')
        if msg["text"] != 'Please dont do that':
            self.send_text_with_quote(msg['chat']['id'], msg["text"], quoted_msg_id=msg["message_id"])
        else:
            self.send_text(msg['chat']['id'], 'I am so sorry!!')

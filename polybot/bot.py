import telebot
from loguru import logger
import os
import time
import requests
import boto3
from telebot.types import InputFile
import uuid
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file if present

class Bot:
    def __init__(self, token, telegram_chat_url):
        self.telegram_bot_client = telebot.TeleBot(token)
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)
        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = "photos"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        local_path = os.path.join(folder_name, file_info.file_path.split('/')[-1])
        with open(local_path, 'wb') as photo:
            photo.write(data)

        return local_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(chat_id, InputFile(img_path))


class ObjectDetectionBot(Bot):
    def __init__(self, token, telegram_chat_url, s3_bucket_name, yolo5_url):
        super().__init__(token, telegram_chat_url)
        self.s3_client = boto3.client('s3')
        self.s3_bucket_name = s3_bucket_name
        self.yolo5_url = yolo5_url

    def upload_to_s3(self, file_path, s3_key):
        try:
            self.s3_client.upload_file(file_path, self.s3_bucket_name, s3_key)
            logger.info(f'Uploaded {file_path} to S3 bucket {self.s3_bucket_name}')
        except Exception as e:
            logger.error(f'S3 upload error: {e}')
            raise

    def request_yolo5_prediction(self, img_name):
        response = requests.post(f'{self.yolo5_url}/predict', params={'imgName': img_name})
        response.raise_for_status()
        return response.json()

    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            photo_path = self.download_user_photo(msg)
            img_name = os.path.basename(photo_path)
            s3_key = f'photos/{img_name}'

            # Upload the photo to S3
            self.upload_to_s3(photo_path, s3_key)

            # Send an HTTP request to the YOLOv5 service for prediction
            try:
                prediction = self.request_yolo5_prediction(img_name)
                # Send the returned results to the Telegram end-user
                labels = prediction.get('labels', [])
                result_text = f'Prediction results for {img_name}:\n'
                class_counts = {}
                for label in labels:
                    class_name = label['class']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                for class_name, count in class_counts.items():
                    result_text += f"{class_name}: {count}\n"

                self.send_text(msg['chat']['id'], result_text)

                # Optionally, send the predicted image back to the user
                predicted_img_path = prediction.get('predicted_img_path')
                if predicted_img_path:
                    local_predicted_img_path = os.path.join('static/data', predicted_img_path)
                    self.send_photo(msg['chat']['id'], local_predicted_img_path)
            except Exception as e:
                self.send_text(msg['chat']['id'], f"Error during prediction: {str(e)}")
        else:
            self.send_text(msg['chat']['id'], 'Please send an image.')

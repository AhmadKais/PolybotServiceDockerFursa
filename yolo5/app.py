import time
from pathlib import Path
from flask import Flask, request, jsonify
from detect import run  # Ensure you have the detect.py in the correct location
import uuid
import yaml
from loguru import logger
import os
import boto3
import pymongo
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from pymongo.errors import ConnectionFailure, OperationFailure

# Set environment variables for bucket name and MongoDB connection
images_bucket = 'kais-image-bucket'
mongo_uri = 'mongodb://localhost:27017,localhost:27018,localhost:27019/?replicaSet=myReplicaSet'

# Load class names from COCO dataset
with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

app = Flask(__name__)

# Set up MongoDB connection
try:
    mongo_client = pymongo.MongoClient(mongo_uri)
    mongo_db = mongo_client['predictions']
    mongo_collection = mongo_db['prediction_summaries']
except (ConnectionFailure, OperationFailure) as e:
    logger.error(f"MongoDB connection error: {e}")

# Set up S3 client
s3 = boto3.client('s3')


def download_from_s3(img_name, bucket_name):
    static_dir = Path('/usr/src/app/static')
    local_path = static_dir / img_name

    # Ensure the static directory exists
    os.makedirs(static_dir, exist_ok=True)

    try:
        s3.download_file(bucket_name, img_name, str(local_path))
    except (NoCredentialsError, PartialCredentialsError) as e:
        logger.error(f"S3 download error: {e}")
        raise
    return local_path


def upload_to_s3(file_path, bucket_name, s3_key):
    try:
        s3.upload_file(str(file_path), bucket_name, s3_key)
    except (NoCredentialsError, PartialCredentialsError) as e:
        logger.error(f"S3 upload error: {e}")
        raise


@app.route('/predict', methods=['POST'])
def predict():
    prediction_id = str(uuid.uuid4())
    logger.info(f'Prediction: {prediction_id}. Start processing')

    img_name = request.args.get('imgName')
    if not img_name:
        return jsonify({'error': 'imgName parameter is required'}), 400

    try:
        # Download image from S3
        original_img_path = download_from_s3(img_name, images_bucket)
        logger.info(f'Prediction: {prediction_id}/{original_img_path}. Download img completed')

        # Predict objects in the image
        run(
            weights='yolov5s.pt',
            data='data/coco128.yaml',
            source=str(original_img_path),
            project='static/data',
            name=prediction_id,
            save_txt=True
        )
        logger.info(f'Prediction: {prediction_id}/{original_img_path}. Done')

        # Predicted image path
        predicted_img_path = Path(f'static/data/{prediction_id}/{img_name}')

        # Upload predicted image to S3
        upload_to_s3(predicted_img_path, images_bucket, f'static/data/{prediction_id}/{img_name}')

        # Parse prediction labels and create a summary
        pred_summary_path = Path(f'static/data/{prediction_id}/labels/{img_name.split(".")[0]}.txt')
        if pred_summary_path.exists():
            with open(pred_summary_path) as f:
                labels = f.read().splitlines()
                labels = [line.split(' ') for line in labels]
                labels = [{
                    'class': names[int(l[0])],
                    'cx': float(l[1]),
                    'cy': float(l[2]),
                    'width': float(l[3]),
                    'height': float(l[4]),
                } for l in labels]

            logger.info(f'Prediction: {prediction_id}/{original_img_path}. Prediction summary:\n\n{labels}')

            prediction_summary = {
                'prediction_id': prediction_id,
                'original_img_path': str(original_img_path),
                'predicted_img_path': str(predicted_img_path),
                'labels': labels,
                'time': time.time()
            }

            # Store prediction summary in MongoDB
            try:
                mongo_collection.insert_one(prediction_summary)
            except Exception as e:
                logger.error(f"MongoDB insert error: {e}")

            return jsonify(prediction_summary)
        else:
            return jsonify({'error': 'Prediction result not found'}), 404
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)

# Sentiment Classification Using Tensorflow (With Attention)

This is a sentiment classifier machine learning model trained on IMDB Movie review dataset.

This classifier works this way. First it loads raw data, cleans it, creates data dictionary, creates words to integer mapping and saves all the train, validation and dictionary files in disk.

Here the model has been built using Bi-Directional LSTM cell with attention. With attention mechanism a LSTM/RNN network works better.For more details on attention mechanism you can refer this link: https://medium.com/@joealato/attention-in-nlp-734c6fa9d983

To run the training job you have to download Glove Vector from here: http://nlp.stanford.edu/data/glove.twitter.27B.zip

There are two way you can run the model: Local training and Gcloud training.
Both process has been mentioned in cloud_training.ipynb notebook.

Steps to follow to run training job:

Go to project directory and run:
$ python -m src.task  --root_path=${PWD}  --train_data_path='data/train'  --val_data_path='data/test'  --resources_path='resources'  --output_dir='output'  --job-dir=./tmp 

(You can add other parameters also. Check task.py for supported parameters)

After training to serve the model for prediction:
$ sudo apt-get install tensorflow-model-server 

Then run this command to start serving queue
$ tensorflow_model_server \
    --rest_api_port=8501 \
    --model_name=lstm \
    --model_base_path="${MODEL_DIR}" 

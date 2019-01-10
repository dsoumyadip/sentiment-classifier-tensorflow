# SentimentClassifier_TF

###### This is a sentiment classifier machine learning model trained on IMDB Movie review dataset.
### LSTM units: 
###### 200 
### Embedding vector:
###### 50 dimensional Glove vector
### After training to run the model:
###### $ sudo apt-get install tensorflow-model-server
###### $ tensorflow_model_server \
######  --rest_api_port=8501 \
######  --model_name=lstm \
######  --model_base_path="${MODEL_DIR}" 

###### * {MODEL_DIR} = Where exported model file(.pb file) is present.


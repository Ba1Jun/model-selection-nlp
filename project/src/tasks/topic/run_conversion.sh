DATA_PATH=project/resources/data/ag_news

# prepare and split data
python3 project/src/tasks/topic/convert_news.py ag_news $DATA_PATH --text_column text --label_column label -rs 4012
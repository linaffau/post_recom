import os

from fastapi import FastAPI

from typing import List

from schema import PostGet

from catboost import CatBoostClassifier

from datetime import datetime
import pandas as pd
import pickle

from sqlalchemy import create_engine


# функции для загрузки модели
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("/my/super/path")
    return CatBoostClassifier().load_model(model_path)


# функция для выгрузки признаков чанками из базы данных для модели
def batch_load_sql(query: str) -> pd.DataFrame:
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
        
    )
    conn = engine.connect().execution_options(stream_results = True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize = 200000):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

# функция для загрузки трех таблиц из базы данных: users, таблица с новыми признаками и все взаимодействия, где был лайк
def load_features():

    # информация о всех юзерах
    users_table = pd.read_sql(
        "SELECT * FROM public.user_data",
        con = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
        )
    
    # информация о всех постах из таблицы с новыми признакми
    posts_table = pd.read_sql(
        "SELECT * FROM public.afeo_final_project_dl",
        con = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
        )

    # информация о пролайканх потах
    liked_posts_query = """
            SELECT DISTINCT post_id, user_id
            FROM public.feed_data
            WHERE action = 'like'
            """
     
    liked_posts_table = batch_load_sql(liked_posts_query)

    return users_table, posts_table, liked_posts_table


# выгружаю модель
model = load_models()

# выгружаю признаки из трех таблиц
users_table, posts_table, liked_posts_table = load_features()

# функция для рекоммендации
def get_recommended_posts(id: int,
                          time: datetime,
                          limit: int):

    # выгружаю признаки нужного юзера
    user_features = users_table[users_table["user_id"] == id].drop("user_id", axis = 1)

    # выгружаю признаки псотов
    posts_features = posts_table.drop(["text", "index"], axis = 1)
    # информация о самих постах
    posts_info = posts_table[["post_id", "text", "topic"]]

    # соединяю пользователя со всеми постами
    user_dict = dict(zip(user_features.columns, user_features.values[0]))
    posts_user_features = posts_features.assign(**user_dict).set_index("post_id")

    # добавляю в таблицу месяц, день недели и час рекомендации
    posts_user_features["weekday"] = time.weekday()
    posts_user_features["hour"] = time.hour

    # предсказание моделью
    predictions = model.predict_proba(posts_user_features)[:, 1]
    posts_user_features["predictions"] = predictions

    # убираю записи, где данный пользователь уже ставил лайк
    liked_posts_ids = liked_posts_table[liked_posts_table["user_id"] == id]["post_id"].values
    filtered_predictions = posts_user_features[~ posts_user_features.index.isin(liked_posts_ids)]

    # оставляю топ limit предсказаний
    recommended_posts = filtered_predictions.sort_values("predictions", ascending = False)[:limit].index

    return [
        PostGet(**{
            "id": i,
            "text": posts_info[posts_info["post_id"] == i]["text"].values[0],
            "topic": posts_info[posts_info["post_id"] == i]["topic"].values[0]
            }
                )
        for i in recommended_posts
        ]

# end-point
app = FastAPI()
    
@app.get("/post/recommendations/", response_model = List[PostGet])
def get_recommended_feed(id: int,
                         time: datetime,
                         limit: int  = 5) -> List[PostGet]:

    return get_recommended_posts(id, time, limit)

from sentence_transformers import SentenceTransformer
import collections
import torch
import re
import time
import io

import tensorflow

import nltk
from pymystem3 import Mystem
nltk.download('punkt')

import fitz
import numpy as np
import pandas as pd
import streamlit as st

from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

@st.cache(allow_output_mutation=True)
def load_model():
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    return model


def load_cvs_emb():
    # загружаем базу эмбеддингов резюме после работы модели
    cvs_reloaded = np.load('data_tags.npy', allow_pickle=True)
    return cvs_reloaded



def load_cvs():
    # загружаем датасет резюме
    cvs = pd.read_csv("data_ready.csv", delimiter=';')
    return cvs


# функция предподготовки текста (только очистка - для работы модели и преобразования в эмбеддинг)
def clean_text(text):
    # приводим текст к нижнему регистру
    text = text.lower()
    # создаем регулярное выражение для удаления лишних символов
    regular = r'[\*+\#+\№\"\-+\+\=+\?+\&\^\.+\;\,+\>+\(\)\/+\:\\+\©\—\!\•\|\{\]\[\}\'\–\«\»]'
    # регулярное выражение для замены ссылки на "URL"
    regular_url = r'(http\S+)|(www\S+)|([\w\d]+www\S+)|([\w\d]+http\S+)'
    # удаляем лишние символы
    text = re.sub(regular, ' ', text)
    # заменяем ссылки на "URL"
    text = re.sub(regular_url, r'URL', text)
    # возвращаем токенезированые данные
    text = word_tokenize(text)
    # возвращаем очищенные данные в виде строки
    text = ' '.join(text)
    # удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    return text


def save_uploadedfile(uploadedfile):
    with open('tmp.pdf', "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Файл добавлен успешно!")


def open_uploadedfile():
    with fitz.open('tmp.pdf') as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text



def convert_df(df):
    return df.to_csv().encode('utf-8')


def main():
    st.title("Система поиска похожих резюме")
    st.write(
        "Загрузите проторезюме, и система покажет наиболее похожие на него резюме из базы!")
    st.image("1cv.png")

    uploaded_file = st.file_uploader("Загрузите PDF-файл с проторезюме", type=['pdf'])
    if uploaded_file is not None:
        save_uploadedfile(uploaded_file)
        text = open_uploadedfile()
        proto_cv = clean_text(text)
        rec_num = st.slider(
            'Выберите количество рекомендаций',
            5, 50, 10)

        st.write('Количество: ', rec_num)
        if st.button('Показать похожие резюме'):
            with st.spinner('Ищем похожие резюме...'):
                start_time = time.time()

                result = recommend(proto_cv, rec_num)
                st.dataframe(result)
                st.write("Время поиска: {:.2f}s".format(time.time() - start_time))
                st.balloons()

                csv = convert_df(result)
                st.download_button(
                    label="Скачать список рекомендаций",
                    data=csv,
                    file_name="recommendation.csv",
                    mime="csv",
                    key='download-csv')


def recommend(proto_cv, rec_num):
    model = load_model()

    # функция преобразования текста в эмбеддинги
    def labse(text):
        e = model.encode(text)
        return e

    # преобразуем проторезюме в эмбеддинг
    proto_cv_emb = labse(proto_cv)

    # сравнение по полному тексту проторезюме
    proto_cv_sim = cosine_similarity(load_cvs_emb(), proto_cv_emb.reshape(1, -1))
    # преобразуем в одномерный массив
    proto_cv_sim_1d = proto_cv_sim.ravel()

    # создаем датафрейм с колонками сходств
    dataset_sim = pd.DataFrame({'proto_cv_sim': proto_cv_sim_1d}, columns=['proto_cv_sim'])

    result = dataset_sim.sort_values('proto_cv_sim', ascending=False).head(rec_num)

    # создаем список индексов рекомендаций
    max_sim_indx = list(result.index)

    recommendation = load_cvs().iloc[max_sim_indx].copy()

    return recommendation


if __name__ == '__main__':
    main()

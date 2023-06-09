РЕКОМЕНДАТЕЛЬНЫЙ СЕРВИС ПОСТОВ ДЛЯ СОЦИАЛЬНОЙ СЕТИ

В данном проекте я создала рекомендательный сервис текстовых постов для социальной сети, которые понравятся пользователям с наибольшей вероятностью. Данный рекомендательный сервис выводит топ n-постов для каждого пользователя в определенный момент времени. Посты выводятся в следующем виде:
- идентификатор поста;
- текст поста;
- тема поста.

Основные шаги по созданию рекомендательного сервиса:
- Загрузка данных из базы данных
- Обработка данных в Jupyter Notebook: исследование данных, обработка признаков, создание новых признаков
- Тренировка модели в Jupyter Notebook, оценка ее качества на тестовой выборке, выбор "лучшей" модели, выбор "лучших" параметров модели
- Сохранение модели
- Загрузка таблиц с новыми признаками в базу данных
- Написание рекомендательного сервиса:
    - выгрузка необходимых таблиц с признаками из базы данных
    - получение предсказаний модели
    - возвращение ответа

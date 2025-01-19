# Телеграм бот

Он будет принимать картинки на вход и классифицировать их. 

Создаем окружение:

``` python3 -m venv venv ```

``` source venv/bin/activate ```

``` pip install -r requirements.txt ```

Тренируем модель:

``` python model/model.py ```

Поднимаем бота:

``` docker build -t cifar10_bot .  ```

``` docker run -d -p 443:443 --restart unless-stopped cifar10_bot ```

[Сайт проекта](https://an4lo1.fvds.ru/)

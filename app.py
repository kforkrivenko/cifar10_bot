import os
import torch
import torchvision.transforms as transforms
import ssl
from PIL import Image
from torchvision.models import resnet18
from model.model import Net
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.types import ContentType, BotCommand, ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils.executor import start_webhook
from aiogram.contrib.middlewares.logging import LoggingMiddleware
import pandas as pd
import matplotlib.pyplot as plt
# база данных
from database import init_db, update_images, update_images_correct, get_stats

# Загрузка модели
MODEL_PATH = "model/cifar10_model.pth"

# Загрузка переменных из .env
load_dotenv()

# Получение токена и домена из переменных окружения
TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_HOST = os.getenv("DOMAIN_NAME")
WEBHOOK_PATH = '/webhook'
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

# Настройки сервера
WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = 8443

model = Net()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

# Telegram Bot
bot = Bot(TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

# Классы CIFAR-10
CIFAR10_CLASSES = [
    "Самолет", "Автомобиль", "Птичка", "Кошка", "Олень",
    "Собака", "Лягушка", "Лошадь", "Корабль", "Грузовик"
]

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_feedback_keyboard():
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    buttons = [KeyboardButton(text="Да"), KeyboardButton(text="Нет")]
    keyboard.add(*buttons)
    return keyboard


async def predict_image(file_path):
    image = Image.open(file_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    return CIFAR10_CLASSES[predicted.item()]


@dp.message_handler(commands=["start"])
async def cmd_start(message: types.Message):
    await message.reply("Привет! Отправь мне изображение, и я попробую его классифицировать.")


# Обработчик изображений
@dp.message_handler(content_types=[ContentType.PHOTO])
async def handle_photo(message: types.Message):
    user_id = message.from_user.id
    username = message.from_user.username or "unknown"

    # Сохраняем изображение
    photo = message.photo[-1]
    file_path = f"data/images/{user_id}_{photo.file_unique_id}.jpg"
    await photo.download(file_path)

    prediction = await predict_image(file_path)

    await update_images(user_id, username)

    await message.reply(f"Я думаю, что это: {prediction}. Это правда?", reply_markup=get_feedback_keyboard())


# Обработчик ответов на предсказания
@dp.message_handler(lambda message: message.text.lower() in ["да", "нет"])
async def handle_feedback(message: types.Message):
    user_id = message.from_user.id
    username = message.from_user.username or "unknown"

    feedback = message.text.lower()
    if feedback == "да":
        await update_images_correct(user_id, username)
        await message.reply("Отлично, я угадал!")
    else:
        await message.reply("Жаль, я ошибся. Буду стараться лучше!")


# Команда для отображения статистики
@dp.message_handler(commands=["stats"])
async def show_stats(message: types.Message):
    stats = await get_stats()

    data = {
        'username': [],
        'отправленных картинок': [],
        'корректность': []
    }

    for stat in stats:
        data['username'].append(stat[0])
        data['отправленных картинок'].append(stat[1])
        data['корректность'].append(round(stat[2] / stat[1], 2) if stat[1] > 0 else 0)
    
    df = pd.DataFrame(data)

    # Настраиваем внешний вид таблицы
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Сохраняем изображение
    plt.savefig('table.png', bbox_inches='tight', dpi=150)

    # Отправляем изображение в Telegram
    with open('table.png', 'rb') as photo:
        await bot.send_photo(message.chat.id, photo)

    # Удаляем файл после отправки
    os.remove('table.png')

async def set_bot_commands():
    commands = [
        BotCommand(command='/start', description='Запустить бота'),
        BotCommand(command='/stats', description='Статистика всех пользователей'),
    ]
    await bot.set_my_commands(commands)

async def on_startup(dp):
    await bot.set_webhook(WEBHOOK_URL)
    await set_bot_commands()

async def on_shutdown(dispatcher):
    # Удаляем Webhook
    await bot.delete_webhook()

if __name__ == "__main__":
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile='data/fullchain.pem', keyfile='data/privkey.pem')
    init_db()

    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
        ssl_context=ssl_context,
    )
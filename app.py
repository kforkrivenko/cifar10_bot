import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request
import telebot
from telebot import TeleBot
from torchvision.models import resnet18
from model.model import Net

# Загрузка модели
MODEL_PATH = "model/cifar10_model.pth"
os.environ["BOT_TOKEN"] = "7557788515:AAEr6bk1ljM1aoenj4nZtB6c4SyK_5xt_-A"

model = Net()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Telegram Bot
TOKEN =os.getenv('BOT_TOKEN')
bot = TeleBot(TOKEN)

# Flask-сервер
app = Flask(__name__)

# Классы CIFAR-10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Предобработка изображения
def preprocess_image(image_path):
    transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Обработка сообщений Telegram
@bot.message_handler(content_types=["photo"])
def handle_image(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    img_path = "input.jpg"
    with open(img_path, "wb") as img_file:
        img_file.write(downloaded_file)

    # Предсказание
    input_tensor = preprocess_image(img_path)
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = CIFAR10_CLASSES[predicted.item()]

    print(f"Я думаю, что это: {predicted_class}")

    bot.reply_to(message, f"Я думаю, что это: {predicted_class}")

@bot.message_handler(content_types=["text"])
def handle_text(message):
    bot.reply_to(message, "Привет")

# Flask endpoint для webhook
@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    json_string = request.stream.read().decode("utf-8")
    update = bot.process_new_updates([telebot.types.Update.de_json(json_string)])
    return "!", 200

if __name__ == "__main__":
    bot.remove_webhook()
    bot.set_webhook(url=f"https://an4lo1.fvds.ru/{TOKEN}")
    app.run(host='0.0.0.0', port=443, ssl_context=('data/fullchain.pem', 'data/privkey.pem'))

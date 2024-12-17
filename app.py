import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request
from telebot import TeleBot
from torchvision.models import resnet18

# Загрузка модели
MODEL_PATH = "model/cifar10_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
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
        transforms.Resize((30, 30)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

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

    bot.reply_to(message, f"Я думаю, что это: {predicted_class}")

# Flask endpoint для webhook
@app.route("/" + TOKEN, methods=["POST"])
def webhook():
    json_string = request.stream.read().decode("utf-8")
    update = bot.process_new_updates([telebot.types.Update.de_json(json_string)])
    return "!", 200

if __name__ == "__main__":
    bot.remove_webhook()
    bot.set_webhook(url=f"https://5.35.99.243/{TOKEN}")
    app.run(host="0.0.0.0", port=5000)

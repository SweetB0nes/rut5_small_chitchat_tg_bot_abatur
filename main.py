import telebot
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
import torch
import config

bot = telebot.TeleBot(config.TOKEN)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_tokenizer(model_name):
	return T5Tokenizer.from_pretrained(model_name)

def load_model(model_path):
	return T5ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)

def generate(model, tokenizer, text,
			 do_sample=True, max_length=64,top_p=0.5, repetition_penalty = 2.5,num_return_sequences=1):

	inputs = tokenizer(text, return_tensors='pt').to(DEVICE)
	with torch.no_grad():
		gen = model.generate(
			**inputs,
			do_sample=do_sample, top_p=top_p, num_return_sequences=num_return_sequences,
			repetition_penalty=repetition_penalty,
			max_length=max_length)
		for h in gen:
			answer = (tokenizer.decode(h, skip_special_tokens=True))
	return answer


tokenizer = load_tokenizer(config.model_name)
model = load_model(config.model_path)

print('MODEL_GPT SUCCESSFUL')


@bot.message_handler(commands=['start'])
def send_welcome(message):
	bot.reply_to(message, f'Приветствую, {message.from_user.first_name}')

@bot.message_handler(content_types=['text'])
def send_message(message):
	try:
		user = message.text
		user_text = user
		generator = generate(model, tokenizer, text=user_text)
		bot.send_message(message.chat.id, text=generator)
	except :
		bot.send_message(message.chat.id, 'Упс, что-то пошло не так :( Обратитесь в службу поддержки!')


bot.polling()
"""
Amaç:
    1. Python projesinin Docker içinde çalışan vLLM servisine bağlanıp bağlanmadığını test et.
    2. Modelden (gemma3:4b) kısa bir cevap alalım ve ortamın doğru kurulduğunu kontrol et.

Adımlar:
    1. gerekli kütüphaneleri içeriye aktararak başla
    2. .env dosyasında ki bağlantı ayarlarını oku
    3. OpenAI uyumlu client nesnesini oluştur
    4. vLLM üzerinde çalışan modele kısa bir istek gönder
    5. Gelen cevabı terminale yazdır

Kurulumlar:
pip install openai python-dotenv
"""

# 1. gerekli kütüphaneleri içeriye aktararak başla
import os
from dotenv import load_dotenv
from openai import OpenAI

# 2. .env dosyasında ki bağlantı ayarlarını oku
load_dotenv()
base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
api_key = os.getenv("VLLM_API_KEY", "dummy-key")
model_name = os.getenv("VLLM_MODEL", "google/gemma-3-4b-it")

# 3. OpenAI uyumlu client nesnesini oluştur
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

# 4. vLLM üzerinde çalışan modele kısa bir istek gönder
response = client.chat.completions.create(
    model = model_name,
    messages=[
        {
            "role": "system", # system prompt
            "content": "Sen Türkçe cevap veren yardımcı bir yapay zeka asistanısın."
        },
        {
            "role": "user", # user prompt
            "content": "On-prem LLM nedir? 2 cümleyle açıkla."
        }
    ],
    temperature=0.3,
    max_tokens=400
)

# 5. Gelen cevabı terminale yazdır
answer = response.choices[0].message.content
print(f"Model Cevabı: \n{answer}")
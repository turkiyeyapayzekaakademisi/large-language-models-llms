"""
Gemini API ile Streaming ve Rate Limit Yönetimi

1. Terminal üzerinden mesaj al ve gemini'a gönder
2. Model cevabını streaming response ile terminal de parça parça göster
3. Bunlara paralel olarak API'dan gelen rate limit hatalarını yakalayarak otomatik yeniden deneme mekanizması ekle

Adımlar:
    1. Gerekli kütüphaneleri içeriye aktar
    2. API key ve model ayarlarını okuma işlemi
    3. Gemini client nesnesi oluştur
    4. Sistem mesajını ve model parametrelerini tanımla
    5. Rate limit ve retry mantığını tanımla
    6. Konuşma döngüsü başlat
    7. Kullanıcı mesajını terminalden al
    8. Streaming response ile Gemini API'ye istek gönder
    9. Gelen cevabı token token terminalde yazdır.

Kurulumlar:
    pip install google-genai python-dotenv tenacity
"""

# 1. Gerekli kütüphaneleri içeriye aktar
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

# 2. API key ve model ayarlarını okuma işlemi
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# 3. gemini client nesnesi oluştur
client = genai.Client(api_key=api_key)


# 4. Sistem mesajını ve model parametrelerini tanımla
system_message = """
    Sen yapay zeka eğitimi veren deneyimli bir eğitmensin. 
    Cevaplarını Türkçe ver.
    Konuları yada sana sorulan soruları sade, anlaşılır ve örneklerle açıkla.
    Gereksiz yere uzun cevap üretme.
"""

generation_config = types.GenerateContentConfig(
    system_instruction=system_message,
    temperature=0.7, # yaratıcı yada garantici
    top_p=0.95, # moldein token seçerken toplam olasılık havuzunun ne kadarını dikkate alacağı
    top_k=140, # modelin her adımda değelerndireceği en olası token sayısını sınırlar
    max_output_tokens=2048 # model cevabının maksimum uzunluğu
)

# 5. Rate limit ve retry mantığını tanımla
logging.basicConfig(level = logging.WARNING)
logger = logging.getLogger(__name__)

# _deneme_sayaci = 0 # test - commente al

@retry( # tenacity'nin retry decorator - fonksyionu otomatik olarak yeniden çağırmak için kullanılır
    retry=retry_if_exception_type((genai_errors.ClientError, genai_errors.ServerError, Exception)), # yani client error, server error veya exception alırsak retry yapalım
    stop = stop_after_attempt(4), # en fazla 4 deneme yapar, 4. denemede hata alırsa tamamen durur
    wait = wait_exponential(multiplier=1, min = 2, max = 16), # denemeler arası bekleme: 2s - 4s - 8s - 16s
    before_sleep=before_sleep_log(logger, logging.WARNING) # Her denemeden önce terminale bir uyarı logu yazar
)
def send_streaming_request(user_message: str):
    """
        kullanıcı sorgusunu alır
        gemini a gönderir
        gelen cevabı return eder
    """

    # # test başlangıç - deneme bitince commente al
    # global _deneme_sayaci
    # _deneme_sayaci += 1
    # if _deneme_sayaci < 3:
    #     print(f"[TEST] Deneme {_deneme_sayaci} - hata simüle ediliyor...")
    #     raise Exception("429 rate limit simülasyonu")
    # _deneme_sayaci = 0
    # # test bitiş

    llm_response = client.models.generate_content_stream(
        model = model_name, 
        contents= user_message,
        config = generation_config 
    )

    return llm_response

# send_streaming_request("deneme")
"""
[TEST] Deneme 1 - hata simüle ediliyor...
WARNING:__main__:Retrying __main__.send_streaming_request in 2 seconds as it raised Exception: 429 rate limit simülasyonu.
[TEST] Deneme 2 - hata simüle ediliyor...
WARNING:__main__:Retrying __main__.send_streaming_request in 2 seconds as it raised Exception: 429 rate limit simülasyonu.
"""

# llm_response = send_streaming_request("makine öğrenmesi nedir")
# for chunk in llm_response:
#     if chunk.text:
#         print(chunk.text, end = "", flush = True)

# 6. Konuşma döngüsü başlat
while True:

    # 7. kullanıcı mesajını terminalden alma
    user_message = input("Kullanıcı: ").strip()

    if user_message.lower() in ["q", "quit"]:
        break

    # 8. steraming response ile gemini api'ye istek gönderme
    try:
        stream = send_streaming_request(user_message)

        for chunk in stream:
            if chunk.text:
                print(chunk.text, end = "", flush = True)

        print("")
    except Exception as e:
        print(e)
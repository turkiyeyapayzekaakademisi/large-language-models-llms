"""
Gemini API ile Duygu Analizi ve Token Sayımı

1. Kullanıcı terminal üzerinden bir müşteri yorumu yazar
2. Gemini API ile sonucu analiz edip yapılandırılmış JSON çıktısı ile terminalde gösterelim
3. Token kullanımını ve tahmini maliyeti hesapla

Adımlar:
    1. Gerekli kütüphaneleri içe aktar
    2. API key ve model ayarlarını okuyalım
    3. Gemini client nesnesi oluştur
    4. JSON çıktı şemasını tanımla
    5. Sistem mesajını ve model parametrelerini hazırla
    6. Kullanıcıdan müşteri yorumu alalım
    7. Gemini API'ye structured output isteği gönder
    8. JSON cevabını terminale yazdır
    9. Token kullanımını ve tahmini maliyeti hesapla

Kurulumlar:
    pip install google-genai python-dotenv
"""
# 1. gerekli kütüphanelerin içeriye aktarılması
import json 
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 2. api key ve model ayarlarını okuma
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# 3. gemini client nesnesi oluştur
client = genai.Client(api_key=api_key)

# 4. JSON çıktı şemasını tanımla
review_schema = {
    "type": "object",
    "properties": {
        "duygu": {
            "type": "string",
            "enum": ["olumlu", "olumsuz", "nötr"]
        },
        "kategori":{
            "type": "string",
            "enum": ["ürün kalitesi", "teslimat", "fiyat", "müşteri hizmetleri", "diğer"]
        },
        "ozet": {
            "type": "string"
        },
        "aksiyon_onerisi":{
            "type": "string"
        }
    },
    "required": [
        "duygu",
        "kategori",
        "ozet",
        "aksiyon_onerisi"
    ]
}

# 5. Sistem mesajını ve model parametrelerini hazırla
system_message = """
    Sen müşteri yorumlarını analiz eden bir yapay zeka asistanısın.
    Sadece verilen müşteri yorumuna göre analiz yap.
    Çıktıyı mutlaka istenen JSON şemasına uygun üret.
    Ek açıklama, markdown veya JSON formatı dışında metin yazma.
"""

generation_config = types.GenerateContentConfig(
    system_instruction=system_message,
    temperature=0.1,
    top_p=0.9,
    top_k=40,
    max_output_tokens=500,
    candidate_count=1, # tek analiz sonucu
    response_mime_type="application/json", # modelden json formatında çıktı istediğimizi belirtiyoruz
    response_schema=review_schema # modelin üretmesini beklediğimiz json yapısı
)

# 6. Kullanıcıdan müşteri yorumu alalım
customer_review = input("Müşteri yorumu yazınız: ")

# 7. Gemini API'ye structured output isteği gönder
prompt = f"""
    Aşağıdaki müşteri yorumunu analiz et.
    Müşteri yorumu:
    {customer_review}
"""

response = client.models.generate_content(
    model=model_name,
    contents=prompt,
    config=generation_config
)

# 8. JSON cevabını terminale yazdır
try:
    parsed_response = json.loads(response.text)
    print(json.dumps(
        parsed_response, ensure_ascii=False, indent=4
    ))
except json.JSONDecodeError:
    print(response.text)

"""
Müşteri yorumu yazınız: restoran yemekleri kötüydü garsonlar saygısız ve yavaştı.
    "duygu": "olumsuz",
    "kategori": "diğer",
    "ozet": "Müşteri, restoran yemeklerinin kalitesinden ve garsonların saygısız ve yavaş hizmetinden memnun kalmamıştır.",
    "aksiyon_onerisi": "Restoran yönetimi, yemek kalitesini artırmalı ve garsonların hizmet içi eğitimlerini gözden geçirmelidir."
Müşteri yorumu yazınız: yemekler güzeldi ama fiyatlar pahalıydı
    "duygu": "nötr",
    "kategori": "fiyat",
    "ozet": "Müşteri yemeklerin lezzetinden memnun kalmış ancak fiyatların yüksek olduğunu belirtmiş.",
    "aksiyon_onerisi": "Fiyatlandırma stratejisi gözden geçirilmeli veya fiyat/performans oranı iyileştirilmelidir."
"""

# 9. Token kullanımını ve tahmini maliyeti hesapla
INPUT_PRICE_PER_1M_TOKEN = 0.1 # USD / 1M input token
OUTPUT_PRICE_PER_1M_TOKEN = 0.4 # USD / 1M output token

usage = response.usage_metadata

input_tokens = usage.prompt_token_count
output_tokens = usage.candidates_token_count
total_tokens = usage.total_token_count

input_cost = (input_tokens / 1000000) * INPUT_PRICE_PER_1M_TOKEN
output_cost = (output_tokens / 1000000) * OUTPUT_PRICE_PER_1M_TOKEN
total_cost = input_cost + output_cost

print("Token Kullanımı: ")
print(f"    Giriş Token: {input_tokens}")
print(f"    Çıkış Token: {output_tokens}")
print(f"    Toplam Token: {total_tokens}")

print("Token Maliyeti: ")
print(f"    Giriş Maliyeti: {input_cost}")
print(f"    Çıkış Maliyeti: {output_cost}")
print(f"    Toplam Maliyeti: {total_cost}")



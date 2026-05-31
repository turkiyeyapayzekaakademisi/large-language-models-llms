"""
Belge Özetleme Uygulaması

Amaç:
    1. PDF içindeki metni oku, vLLM e gönder, vLLM içerisinde çalışan gemma3:4b bu belgeyi özetlesin.

Adımlar:
    1. gerekli kütüphanelerin içeriye aktarılması
    2. env dosyasından ayarların okunması
    3. OpenAI uyumlu client nesnesi oluştur
    4. PDF oku
    5. PDF metnini tek bir prompt içinde modele gönder
    6. Modelden gelen özeti terminal de yazdır.

Kurulumlar:
pip install openai python-dotenv pypdf
"""

# 1. gerekli kütüphaneleri içeriye aktararak başla
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

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

# 4. PDF oku
def read_pdf_text(pdf_path: str) ->str:
    path = Path(pdf_path)

    reader = PdfReader(str(path))
    pdf_text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            pdf_text += f"\n\n--- Sayfa {page_number} ---\n"
            pdf_text += page_text
        
    pdf_text = pdf_text.strip()

    return pdf_text

# text = read_pdf_text("ornek_on_prem_llm_belgesi.pdf")
# print(text)

def summarize_document(pdf_text: str) -> str:
    
    prompt = f"""
    Aşağıdaki PDF belgesini türkçe olarak özetle.

    Özetleme kuralları:
    - Belgenin ana amacını açıkla.
    - En önemli noktaları madde madde yaz.
    - Gereksiz tekrarları çıkar.
    - Metinde olmayan bilgi ekleme.
    - Varsa önemli kararları, sonuçları veya önerileri belirt.
    - Profesyonel ve anlaşılır bir dil kullan 

    PDF belgesi:
    {pdf_text}
    """ 

    response = client.chat.completions.create(
        model = model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen PDF belgelerini dikkatli okuyan, metne sadık kalan "
                    "ve Türkçe profesyonel özetler üreten bir yapay zeka asistanısın."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=1500
    )

    return response.choices[0].message.content.strip()

pdf_path = "ornek_on_prem_llm_belgesi.pdf"
pdf_text = read_pdf_text(pdf_path)
summary = summarize_document(pdf_text)

print(summary)
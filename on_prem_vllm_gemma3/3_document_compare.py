"""
Belge Karşılaştırma Uygulaması

Amaç:
    1. İki farklı belge okunur
    2. Kullanıcı doğal dilde karşılaştırma soruları yazar ve vLLM modeli bu iki teklifi sadece kullanıcı sorusuna göre karşılaştırır

Adımlar:
    1. Gerekli kütüphanelerin içeriye aktarılması
    2. .env dosyasından vLLM bağlantı ayarlarının okunması
    3. OpenAI uyumlu client nesnesini oluşturma
    4. Firma A ve Firma B teklif pdf dosyalarının okunması
    5. Kullanıcıdan karşılaştırma sorusunun alınması
    6. PDF leri ve kullanıcı sorusunu modele gönder
    7. Modelden gelen cevabı terminalde göster

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

FIRMA_A_PDF = "firma_a_teklif.pdf"
FIRMA_B_PDF = "firma_b_teklif.pdf"

def compare_documents(document_a: str, document_b: str, user_question: str) -> str:

    prompt = f"""
    Aşağıda iki farklı firmaya ait teklif metni yer almaktadır.
    Kullanıcı bu iki teklif belgesiyle ilgili karşılaştırmalı bir soru soracaktır.

    Senin görevin:
    Kullanıcının sorusunu dikkatlice anlayarak Firma A ve Firma B tekliflerini
    yalnızca kullanıcının istediği açıdan karşılaştırmaktır.

    Cevaplama kuralları:
    - Cevabı sadece verilen iki teklif belgesine dayanarak üret.
    - Belge dışı bilgi, tahmin veya varsayım ekleme.
    - Kullanıcının sorduğu karşılaştırma başlığına odaklan.
    - Kullanıcı fiyat sorarsa fiyat ve ödeme koşullarını karşılaştır.
    - Kullanıcı süre sorarsa teslimat ve proje süresini karşılaştır.
    - Kullanıcı destek sorarsa bakım, destek, garanti ve SLA kapsamını karşılaştır.
    - Kullanıcı kapsam sorarsa dahil edilen ve hariç bırakılan işleri karşılaştır.
    - Kullanıcı risk sorarsa belirsizlikleri, eksikleri ve dikkat edilmesi gereken noktaları karşılaştır.
    - Kullanıcı genel karşılaştırma isterse fiyat, süre, kapsam, destek ve risk başlıklarını birlikte değerlendir.
    - Bir bilgi belgelerden birinde var, diğerinde yoksa bunu açıkça belirt.
    - Cevap formatını kullanıcının sorusuna göre belirle.
    - Gerekiyorsa madde madde, gerekiyorsa kısa tablo benzeri bir yapı, gerekiyorsa kısa sonuç paragrafı kullan.
    - Sonuç bölümünde hangi teklifin hangi açıdan daha avantajlı olduğunu belgeye dayalı gerekçeyle açıkla.

    Kullanıcının sorusu:
    {user_question}

    ====================
    BELGE A - Firma A Teklifi
    ====================
    {document_a}

    ====================
    BELGE B - Firma B Teklifi
    ====================
    {document_b}
    """

    response = client.chat.completions.create(
        model = model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sen teklif belgelerini dikkatli inceleyen, iki belgeyi karşılaştıran "
                    "ve Türkçe, açık, profesyonel cevaplar veren bir yapay zeka asistanısın. "
                    "Cevaplarını sadece verilen belgelere dayandırırsın."
                )
            },
            {
                "role":"user",
                "content": prompt
            }
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()

document_a = read_pdf_text(FIRMA_A_PDF)
document_b = read_pdf_text(FIRMA_B_PDF)

while True:

    # 5. Kullanıcıdan karşılaştırma sorusunun alınması
    user_question = input("Karşılaştırma sorusu: ").strip()
    
    # 6. PDF leri ve kullanıcı sorusunu modele gönder
    answer = compare_documents(document_a, document_b, user_question)
    
    # 7. Modelden gelen cevabı terminalde göster
    print(f"Model cevabı: \n{answer}")

"""
Tekliflerde bulunan Fiyatları karşılaştır.
Teklifleri destek ve bakım açısından karşılaştır
"""
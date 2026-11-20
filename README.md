# UzAssets — Product Limit Article Matching Service

> **Kompaniya haqida:** «UzAssets» investitsiya kompaniyasi AJ davlat ulushi mavjud yirik korxonalarni transformatsiya qilish, xususiylashtirish va kapital bozoriga tayyorlash jarayonlarini amalga oshiruvchi yetakchi investitsiya kompaniyasidir.

Ushbu servis korxonalarda xarid qilinayotgan tovar va xizmatlarni (xomashyo, materiallar, butlovchi qismlar va xizmatlar) tegishli **byudjet xarajat moddalariga (Limit Article)** avtomatik tarzda, sun'iy intellekt (AI) yordamida aniqlab beruvchi yuqori unumdor mikroxizmatdir.

---

## 🎯 Asosiy g'oya va afzalliklari

Korxonalarda har yili yuz minglab xarid qatorlari shakllanadi, lekin mahsulotlarning katta qismi takrorlanuvchidir:
* «Motor moyi» har doim yoqilg'i-moylash materiallari moddasiga tegishli.
* Servis mahsulotni faqat bir marta AI orqali tasniflaydi va mapping natijasini doimiy saqlash (caching) imkonini beradi:
  ```
  33 716 xarid qatori  →  15 140 noyob mahsulot  →  AI faqat 1 marta tahlil qiladi  →  Natija qayta ishlatiladi
  ```
* **Baza xavfsizligi:** Servis UzAssets bosh ma'lumotlar bazasiga hech qachon yozmaydi (`READ ONLY` + `REPEATABLE READ` snapshot).

---

## 🏗 Arxitektura va Texnologiyalar

* **Til va Framework:** Python 3.11+, FastAPI, Pydantic v2
* **AI Provayderlar:**
  * **vLLM / Gemma-31B** — Continuous batching orqali mahalliy modelda yuqori tezlikda tahlil (soatiga 10 000+ mahsulot).
  * **Google Gemini API** — Fallback va cloud inferensiya.
* **Deterministik Hesh (Fingerprint):** SHA-256 asosida mahsulot nomi va MXIK ma'lumotlari o'zgarganini avtomatik aniqlash.
* **Asinxron Navbat & Kesh:** SQLite (`run_store`) orqali asinxron runlarni boshqarish va kursorli paginatsiya.
* **Katalog saralash (Candidate Ranking):** Minglab byudjet moddalari orasidan dastlabki nomzodlarni shortlist qilish.

---

## 🚀 API Endpointlar

| Metod | Endpoint | Tavsif |
|---|---|---|
| `POST` | `/v1/product-limit-article-matching/runs` | Yangi tahlil run'ini navbatga qo'yish (Idempotent) |
| `GET` | `/v1/product-limit-article-matching/runs/{runId}` | Run holatini tekshirish (`QUEUED`, `RUNNING`, `COMPLETED`, `FAILED`) |
| `GET` | `/v1/product-limit-article-matching/runs/{runId}/results` | Sahifalangan (cursor pagination) natijalarni olish |
| `GET` | `/health` | Servis sog'lomligini tekshirish |

### Tahlil natijalari holatlari (`matchStatus`):
* `MATCHED` — Aniq mos kelgan xarajat moddasi topildi (doimiy mapping uchun saqlanadi).
* `AMBIGUOUS` — Bir nechta modda mos kelishi mumkin yoki kontekstga bog'liq (insonga ko'rsatiladi).
* `NO_MATCH` — Dalillar yetarli emas yoki katalogda mos modda mavjud emas.
* `ERROR` — Tahlil jarayonidagi xatolik (mahsulot hech qachon yo'qotilmaydi).

---

## 💻 Mahalliy muhitda ishga tushirish

1. Virtual muhitni faollashtiring va paketlarni o'rnating:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. `.env` faylini sozlang:
```bash
cp .env.example .env
# .env faylidagi API kalitlar va parametrlar qiymatini kiriting
```

3. Serverni ishga tushiring:
```bash
python main.py
# yoki
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```
Swagger UI: `http://localhost:8001/docs`

4. Testlarni ishga tushirish:
```bash
pytest limit_article_matching/tests.py -v
```

---

## 🔒 Xavfsizlik

Loyiha qat'iy xavfsizlik standartlariga javob beradi: barcha maxfiy kalitlar, parollar va shaxsiy ma'lumotlar `.env` faylida saqlanadi va `.gitignore` orqali repozitoriyga chiqib ketishi qat'iyan taqiqlangan.

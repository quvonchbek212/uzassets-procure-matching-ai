# Product Limit Article Matching

Mahsulot uchun **byudjet xarajat moddasini** aniqlaydigan servis. Backend natijani saqlaydi va o'sha mahsulot uchraydigan **barcha** xarid qatorlarida qayta ishlatadi.

## Asosiy g'oya

Mahsulotning moddasi u qaysi so'rovda turganiga bog'liq emas. «Motor moyi» har doim yoqilg'i-moylash materiali. Demak, uni bir marta tasniflab, javobni doimiy saqlash mumkin.

```
33 716 qator  →  15 140 noyob mahsulot  →  AI bir marta so'raydi  →  backend saqlaydi
```

Vaqt o'tgan sari AI ga murojaat kamayadi: backend faqat mapping'i yo'q mahsulotlarni yuboradi.

## Chegara

Servis **UzAssets bazasiga hech narsa yozmaydi**. `read-user` credential'i faqat o'qish uchun, va ekstraksiya sessiyasi buni transaction darajasida ham majburlaydi (`READ ONLY` + `default_transaction_read_only`).

Mapping'larni saqlash, foydalanuvchi tanlovi va qator update'i — **backend zimmasida**.

## Endpointlar

```
POST /v1/product-limit-article-matching/runs
GET  /v1/product-limit-article-matching/runs/{runId}
GET  /v1/product-limit-article-matching/runs/{runId}/results?limit=&cursor=
```

Boshqa endpoint yo'q. Schema tekshirish va nomzod ballashni ko'rish [`scripts/run_product_matching.py`](../scripts/run_product_matching.py) orqali qilinadi.

### Run boshlash

```http
POST /v1/product-limit-article-matching/runs
Idempotency-Key: product-matching-example-001
Content-Type: application/json

{"organizationId": 1, "productIds": [102, 103]}
```

```http
202 Accepted

{"runId": "product-run-3f9c2a1b8d4e",
 "status": "QUEUED",
 "organizationId": 1,
 "productIds": [102, 103]}
```

| Kirish | Kontrakt |
|---|---|
| `organizationId` | Majburiy, musbat butun son. Mahsulotlar ham, nomzod moddalar ham shu tashkilotdan. |
| `productIds` | Majburiy, bo'sh bo'lmagan massiv. **`organization_product.id`** — qator ID, corporate item ID, mahsulot kodi yoki MXIK ID emas. Maksimum 500 ta noyob ID. Saralanadi va takrorlar olib tashlanadi. |
| `Idempotency-Key` | **Majburiy sarlavha.** Bir xil kalit va normallashtirilgan body — o'sha run; o'zgargan body — `409`. |

`requestIds`, `budgetYear`, `limitArticleId` **qabul qilinmaydi** — ular qator tushunchalari. 8382 backend uchun manba qatorlarni tanlash vositasi, bu yerda esa na kirish, na chiqish cheklovi. Agar 8382 aktiv bo'lsa va mahsulotga mustaqil mos kelsa, u ham qaytishi mumkin.

### Natija

```json
{
  "productId": 183374,
  "productCode": "7c6ee16a-3e27-11f1-85f2-00155decd504",
  "productName": "Moy filtr XR12-740AS",
  "sourceFingerprint": "9f2c…",
  "matchStatus": "MATCHED",
  "matchedLimitArticleId": 8368,
  "matchedArticleCode": "000000079",
  "matchedArticleName": "Моддий-техник ресурслар (материаллар ва запчастлар)",
  "suggestedParentLimitArticleId": null,
  "confidence": 0.85,
  "explanation": "Moy filtr - bu moyning o'zi emas, balki uskunaning ehtiyot qismi.",
  "input": {
    "productId": 183374, "organizationId": 1,
    "productName": "Moy filtr XR12-740AS", "deleted": false,
    "mxikLinks": [{"mxikId": 1041142, "code": "INTP-3XHQ", "isPlaceholder": true, "source": "uz_goods_classification"}]
  },
  "evidence": [{"source": "product", "field": "product_name", "value": "Moy filtr XR12-740AS"}],
  "alternatives": [{"limitArticleId": 8367, "score": 0.3, "reason": "…"}],
  "warnings": ["MXIK_NOT_CLASSIFIED", "PRODUCT_NAME_ONLY_EVIDENCE"],
  "errorCode": null
}
```

| `matchStatus` | Ma'nosi | Backend nima qiladi |
|---|---|---|
| `MATCHED` | Bitta qayta ishlatiladigan mapping | **Saqlaydi** |
| `AMBIGUOUS` | Bir nechta modda, yoki kontekstga bog'liq | Saqlamaydi — odamga ko'rsatadi |
| `NO_MATCH` | Dalil yetarli emas yoki mos modda yo'q | Saqlamaydi |
| `ERROR` | Tahlil qilib bo'lmadi, `errorCode` bilan | Saqlamaydi. **Mahsulot hech qachon tashlab ketilmaydi** |

Faqat `MATCHED` saqlanadi. Qolganlari tahlil tarixi sifatida qolishi mumkin, lekin mapping sifatida emas — hal qilinmagan holatni hal bo'lgandek ko'rsatib bo'lmaydi.

### Paginatsiya

`limit` 1–1000 (default 200), `cursor` opaque, tartib `productId` bo'yicha. Oxirgi sahifada `nextCursor: null`.

Natijalar **faqat `COMPLETED` dan keyin** beriladi — tugallanmagan run'ga so'rov `409`. Backend har bir sahifani olishi shart.

### Xato kodlari

| HTTP | Qachon |
|---|---|
| 400 | Noto'g'ri yoki juda katta kirish, noto'g'ri `limit`/`cursor`, `Idempotency-Key` yo'q |
| 404 | Mahsulotlar shu tashkilotga ochiq emas; run topilmadi yoki muddati o'tgan |
| 409 | Idempotency konflikti, yoki natija hali tayyor emas |
| 503 | Baza yoki schema mos emas |

Javob: `{"code": "...", "message": "...", "traceId": "..."}`. Xabarlarda credential yoki SQL ichki tafsiloti bo'lmaydi.

**Tashkilot izolyatsiyasi:** boshqa tashkilotning mahsuloti va umuman mavjud bo'lmagan mahsulot **bir xil** `404` beradi. Farqini ko'rsatish boshqa tashkilot ma'lumotining mavjudligini tasdiqlagan bo'lardi.

## Fingerprint — ataylab tor

`sourceFingerprint` faqat mahsulotni qamraydi:

```
productId, organizationId, productName, productCode, deleted, updatedAt,
mxikLinks[]  (mxik_id bo'yicha saralangan, har birida iste'mol qilingan barcha maydon)
```

**So'rov ID, qator versiyasi, qatorning hozirgi moddasi — kiritilmaydi.** Brief buni to'g'ridan-to'g'ri talab qiladi, va sababi tuzilmaviy: mapping saqlanadi va barcha qatorlarda ishlatiladi. Qator o'zgarganda siljiydigan fingerprint har bir saqlangan mapping'ni bekordan eskirgan deb belgilagan bo'lardi.

Hisoblash qoidalari ([`fingerprint.py`](fingerprint.py)):

- kalitlar **rekursiv** saralanadi
- `null` saqlanadi — "mahsulot o'chirilgan" va "maydon yo'q" bir xil hash bermasligi kerak
- bo'shliqsiz separatorlar, `ensure_ascii=False` (kirill matni o'zi bo'lib hashlanadi)
- `Decimal` → normallashtirilgan string, `datetime` → UTC + `Z`
- MXIK massivi `mxik_id` bo'yicha saralanadi, ya'ni baza qanday tartibda qaytarsa ham hash bir xil

`catalogFingerprint` — nomzod moddalar to'plami va iste'mol qilingan ierarxiya maydonlari ustidan.

Backend takrorlay olishi uchun `fingerprintSchemaVersion` pin qilinadi. Test vektorlari: `canonical_json()` chiqishi — aynan shu string hashlanadi.

**Eskirishni aniqlash:** dalil o'zgarsa yoki modda inactive bo'lsa, backend saqlangan mapping'ni stale deb belgilab, qayta tahlil so'raydi.

## Ichki oqim

```
POST → mahsulot mavjudligi tekshiriladi (qisqa read-only sessiya) → 202 QUEUED
                                   ↓
              bitta REPEATABLE READ / READ ONLY snapshot
              (mahsulotlar + barcha MXIK bog'lanishlari + katalog)
                                   ↓  transaction yopiladi
              modelga yuborish (bittalab yoki batch)
                                   ↓
              validatsiya → natija → SQLite → COMPLETED
```

## MXIK — barcha bog'lanishlar

Brief: *"A product may have several MXIK links; gather and deduplicate all valid links rather than taking the first one."*

Mahsulot MXIK ga ikki yo'l bilan yetadi:

1. `uz_goods_classification` qatorlari orqali (o'chirilmaganlari)
2. `organization_product.mxik_id` orqali to'g'ridan-to'g'ri

Ikkalasi ham yig'iladi, `mxik_id` bo'yicha takrorlar olib tashlanadi, va har bog'lanish **qaysi jadvaldan kelgani** (`source`) yozib qo'yiladi — brief har bir atributning haqiqiy manbasini hujjatlashtirishni talab qiladi.

**Tasniflar ziddiyatli bo'lsa** (ikki bog'lanishning taksonomiyasi har xil) → `CONFLICTING_MXIK_LINKS` warning'i. Biri boshqasidan afzal deb hisoblanmaydi.

### Jonli ma'lumotdagi holat

2026 kogortasidagi **barcha** MXIK yozuvlari soxta: `type = 'INTERNAL_PRODUCT_CODE'`, kod `INTP-…`, taksonomiya bo'sh, nomi mahsulot nomining nusxasi.

Bunday yozuvni ikkinchi manba sifatida ko'rsatish bitta faktni ikki marta sanashga olib kelardi — model `product_name` va `mxik_name` da bir xil so'zlarni ko'rib, ikki mustaqil manba rozi deb o'qigan bo'lardi. Shuning uchun `MxikLink.is_placeholder()` uni aniqlaydi, prompt MXIK'ni yo'q deb e'lon qiladi, va natijaga `MXIK_NOT_CLASSIFIED` + `PRODUCT_NAME_ONLY_EVIDENCE` qo'shiladi.

## Model chiqishi validatsiyasi

Model taklif qiladi, [`matcher.py`](matcher.py) hal qiladi. Bu yerda xavf qator-darajadagidan yuqori: `MATCHED` saqlanadi va **hamma joyda** ishlatiladi.

- **Katalogda yo'q ID** — tashlab yuboriladi (`MODEL_PROPOSED_UNKNOWN_ARTICLE`)
- **Past confidence** (`< LAM_MIN_CONFIDENCE`, default 0.55) — `AMBIGUOUS` ga tushadi, tanlangan modda alternativda qoladi
- **Yaqin natijalar** (farq `< LAM_AMBIGUITY_MARGIN`, default 0.08) — `AMBIGUOUS`
- **Bir xil nomli moddalar** — majburan `AMBIGUOUS`. Nomlari aynan bir xil bo'lgani uchun model ulardan birini nom bo'yicha tanlagan bo'lishi mumkin emas
- **`contextDependent`** — boshqa hamma narsadan qat'i nazar `AMBIGUOUS`
- **Dalil elementlari** — `field` va `value` bo'sh bo'lsa tashlanadi
- **Parent** — faqat katalog bitta aniq yo'lni bersa

### `CONTEXT_DEPENDENT` — yangi va muhim

Ba'zi mahsulotlar so'rovga qarab turli moddalarga tegishli bo'ladi: bitta komponent bir kontekstda ehtiyot qism, boshqasida kapital xarid.

Model so'rovni ko'rmaydi, saqlangan mapping esa **hammasiga** qo'llaniladi. Shuning uchun bunday holatda `AMBIGUOUS` + `contextDependent: true` qaytariladi.

Noto'g'ri universal mapping — mapping yo'qligidan yomonroq: u jimgina, hamma joyda qo'llaniladi.

## Confidence kalibrlash

Birinchi jonli run'larda model **har bir javobga** 0.95–1.00 bergan edi. Bu ko'rib chiqish yo'lini jimgina o'chiradi: `LAM_MIN_CONFIDENCE` va `LAM_AMBIGUITY_MARGIN` faqat **o'zgaruvchan** raqamlarda ishlaydi.

Oddiy ko'rsatma yordam bermadi. Uning o'rniga:

- **Bandli rubrika** — har oraliq kuzatiladigan shartga bog'langan
- **Ishlangan misollar** — har band uchun, jumladan `AMBIGUOUS` bo'lishi kerak bo'lgan holat
- **Ikkinchi o'rin majburiy** — ballsiz yaqinlik qoidasi solishtiradigan narsa topmaydi
- **Aniq ogohlantirish** — *"agar ketma-ket har javobga 0.95 yozayotgan bo'lsangiz, siz kalibrlamayapsiz"*

Natija: jonli tarqalish 0.85–0.98, `1.0` yo'qoldi.

## Batching

Bitta mahsulot uchun prompt deyarli butunlay qat'iy yukdan iborat — qoidalar va katalog ~1 500 token, mahsulot ~100 token. `LAM_BATCH_SIZE=N` bilan qat'iy yuk bir marta to'lanadi.

Jonli o'lchov: 98 mahsulot 10 ta chaqiruvda, 98 ta emas.

**Xavfsizlik shartlari** (barchasi test bilan qoplangan):

- Batch faqat butun katalog bitta promptga sig'sa ishlaydi; aks holda avtomatik bittalab rejimga qaytadi
- Javob berilmagan `itemId` — **tashlab yuborilmaydi**, alohida so'raladi
- Bitta `itemId` ikki marta kelsa — ikkala nusxa ham rad etiladi
- Butun batch xato bersa — bittalab qayta so'raladi, qayta urinishsiz (bir xil prompt bir xil soatni yenga olmaydi)
- Validatsiya bittalab rejim bilan bir xil
- `itemId` lar 1..N, promptga lokal — modda ID'si bilan aralashmaydi

**Default `1`** — o'chiq. Yoqishdan oldin ikki rejimni bir xil mahsulotlarda solishtirish kerak.

## Schema mapping

Ustun nomlari qattiq yozilmagan. Har mantiqiy maydon o'zi uchraydigan nomlarni e'lon qiladi, `introspect()` ularni `information_schema` bo'yicha hal qiladi.

Bu nazariy ehtiyotkorlik emas: oldingi brief'ning **oltita** mapping'i jonli schema'dan farq qilgan, `treasury_limit_article_parent_link` esa umuman mavjud emas.

Jonli holat (2026-09-11 tasdiqlangan):

```
usable: True
  mxik                                12 maydon
  organization_product                 7 maydon
  treasury_limit_article               9 maydon  (is_deleted yo'q)
  uz_goods_classification              5 maydon
  treasury_limit_article_parent_link   JADVAL YO'Q
```

Tekshirish:

```bash
python scripts/run_product_matching.py --from-requests --sample 1 --dry-run
```

## CLI

```bash
# kogorta qanday mahsulotlarni ishlatadi
python scripts/run_product_matching.py --list --from-requests

# kichik namuna
python scripts/run_product_matching.py --from-requests --sample 20 --out var/pm/sample

# aniq mahsulotlar
python scripts/run_product_matching.py --products 183373,183374 --out var/pm/test

# hammasi, bo'laklab va davom ettirib
python scripts/run_product_matching.py --from-requests --all --out var/pm/full

# inference'siz
python scripts/run_product_matching.py --from-requests --sample 5 --dry-run
```

`--from-requests` PLAN qatorlaridan mahsulot ID'larini topadi. Bu **faqat CLI qulayligi** — servisning o'zi hech qachon so'rovlarni aylanmaydi, u faqat berilgan ID'larni tahlil qiladi.

Chiqish: `results.csv` (biznes ko'rib chiqishi uchun, `storable` ustuni bilan) va har bo'lak uchun JSON.

## Konfiguratsiya

Baza sozlamalari mavjud `.env` dan: `REAL_DB_*`, `DB_SSH_*`. Ixtiyoriy `LAM_*` — [`settings.py`](settings.py) ga qarang.

| O'zgaruvchi | Default | Ma'nosi |
|---|---|---|
| `LAM_AI_MODEL` | `DEFAULT_AI_MODEL` | `GEMINI` yoki `GEMMA` |
| `LAM_BATCH_SIZE` | 1 | Bitta chaqiruvdagi mahsulotlar |
| `LAM_MAX_PRODUCTS_PER_RUN` | 500 | Server tomonidagi ceiling |
| `LAM_MIN_CONFIDENCE` | 0.55 | Bundan past — ko'rib chiqish natijasi |
| `LAM_AMBIGUITY_MARGIN` | 0.08 | Bundan yaqin — `AMBIGUOUS` |
| `LAM_FULL_CATALOG_THRESHOLD` | 60 | Bundan kichik katalog to'liq yuboriladi |
| `LAM_RESULT_RETENTION_HOURS` | 168 | Natijalar saqlanish muddati |

⚠️ **`DB_LOCAL_BIND_PORT=0`** qo'ying. Qat'iy port bilan parallel tunnellar bir listener'ni bo'lishadi va ulanishlar buziladi.

## Testlar

```bash
python -m pytest limit_article_matching/tests.py -q
```

73 test, bazasiz va modelsiz. Qamrov: fingerprint determinizmi va uning tor doirasi, MXIK bog'lanishlarini yig'ish/dedupe/ziddiyat, placeholder aniqlash, leksik ballashning zaifligi, model chiqishi validatsiyasi, `CONTEXT_DEPENDENT`, bir xil nomli moddalar, idempotentlik, `productId` bo'yicha paginatsiya, tashkilot izolyatsiyasi, batching xavfsizligi.

## Ma'lum cheklovlar

- **Autentifikatsiya yo'q — bu qaror.** Uchala endpoint ochiq. Himoya to'liq **tarmoq darajasida**: servis ichki tarmoqda qolishi va tashqariga chiqarilmasligi kerak. Manzilni bilgan har kim run boshlab, mahsulot nomlari va byudjet katalogini o'qiy oladi.

  Bu `organizationId` query parametrining ma'nosini ham o'zgartiradi: u status va natija so'rovini tashkilotga cheklaydi, ya'ni bir tashkilotning run'i boshqasining UI'siga tasodifan tushmaydi — lekin u **o'zi e'lon qilingan** qiymat va hech narsani tasdiqlamaydi. Brief talab qilgan "caller'ga bog'lash" credential paydo bo'lmaguncha mumkin emas.
- **Precision o'lchanmagan.** Biznes hali natijalarni ko'rmagan, threshold'lar kelishilmagan.
- **`AMBIGUOUS` kam chiqadi.** Sinovlarda deyarli hamma narsa `MATCHED` bo'ldi. Bu to'g'ri bo'lishi ham mumkin (sinalgan mahsulotlar chinakam aniq edi), lekin kattaroq namuna kerak.
- **OpenAPI va test vektorlari topshirilmagan.** Brief ularni talab qiladi.

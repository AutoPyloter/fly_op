# Faz S1 Ön Tescili — Sahne-Tabanlı Duyusal-Motor Çerçeveleme

**Tarih:** 2026-09-16. PROTOCOL.md §4.5 kuralı: bu dosya deneyler
koşulmadan önce yazıldı ve koşulduktan sonra değiştirilmeyecek.

## Bağlam ve motivasyon

Faz 1/1b/1c'nin üçü de kapı testini geçemedi (EXPERIMENTS.md,
results/faz1_summary.md) — H1 (connectome kara-kutu optimizasyonda
avantaj sağlar) bu görev çerçevesinde reddedildi, bu nihai ve geçerli bir
sonuç.

Aynı gece, `fly_demos/malecns` projesiyle (MaleCNS connectome + F1 araba
sürüşü) yapılan gayrı-resmi bir keşifte, aynı prensiple (gerçek connectome
vs ER-null, aynı eğitim prosedürü) çok net bir **ters yönlü** sonuç
gözlemlendi: gerçek connectome, kalibrasyon sonrası her metrikte
(okuma R², tur tamamlama, hayatta kalma) ER-null'dan belirgin farkla
üstündü. İki gözlem arasındaki en olası ayrım:

1. **Eğitim:** Faz 1/1b'de okuma matrisi hiç eğitilmedi (rastgele sabit).
   Malecns'te iki aşamalı eğitildi (taklit + ES).
2. **Görev yapısı:** Malecns'in girdisi (lidar ışınları) uzamsal/yönsel
   bir yapıya sahip ve connectome'un gerçekten işlemek üzere evrimleştiği
   türden bir sinyal (çok sayıda duyusal kanaldan az sayıda motor
   kanalına süzme). Faz 1/1b/1c'nin girdisi (ham x enjeksiyonu) böyle bir
   yapıya sahip değil.

Faz S1 bu iki etkeni **aynı anda ama kontrollü şekilde** test eder: aynı
Rastrigin/Sphere kıyaslama fonksiyonları, aynı 30 paylaşılan tohum, aynı
4 substrate ailesi (flywire + 3 null), ama artık (a) girdi yönsel/uzamsal
olarak yapılandırılmış ("ışın" kodlaması) ve (b) okuma katmanı taklit
öğrenmesiyle bir kere kalibre ediliyor (sonra dondurulan).

## Mekanizma

`src/flyopt/variants/fly_proposer_scene.py::FlyProposerScene`.

**Kodlama ("ışınlar"):** Her eksende +/- yönde `ray_radius` kadar adım
atıp amaç fonksiyonunu örnekliyoruz: `r_k = f(x + step_k) - f(x)`,
`k=1..2*dim` (10 boyutlu problem için 20 ışın). Bu göreli-kötülük
sinyalleri, gerçek afferent nöron havuzundaki 20 nörona sabit akım olarak
enjekte ediliyor (Faz 1b'nin havuzu, `data/processed/afferent_indices.npy`).

**Süreklilik:** Faz 1/1b/1c'de substrate her `propose()` çağrısında
sıfırlanıyordu (hafızasız, tek atış). Faz S1'de substrate durumu **tüm
arama koşusu boyunca** bir kere sıfırlanıp korunuyor — malecns'in sürekli
kontrol döngüsüne benzer şekilde.

**Kod-çözme ve eğitim:** Okuma matrisi FlyProposer ile aynı yapıda
(efferent havuzundan 50 nöronun ateşlenme oranı → doğrusal okuma → Δx),
ama artık rastgele sabit değil: arama bütçesi başlamadan önce, ucuz
taklit bir öğretmene (aynı ışın örneklerinden türetilen kaba negatif
gradyan tahmini, `_teacher_delta`) karşı, sabit `CALIB_STEPS=100` adımlık
tek turluk ridge regresyonla kalibre ediliyor (`calibrate_readout`),
sonra **arama boyunca dondurulur** (`tell()` no-op). Faz 1c'nin küçük
bütçeyle çevrimiçi ince ayar yapıp tarama artefaktı ürettiği tuzağa
kasıtlı olarak girilmiyor (EXPERIMENTS.md 2026-09-15) — kalibrasyon tek
seferlik ve bütün substrate'ler için eşit büyüklükte.

**Işın/kalibrasyon örneklemeleri** ucuz sentetik fonksiyon (Rastrigin/
Sphere) üzerinde olduğu için ana arama bütçesine sayılmıyor — sadece
substrate'in kaç kere adım attığı (T=15 LIF adımı × iterasyon sayısı)
bütçe kısıtına tabi, ve bu her substrate için birebir eşit.

## Tasarım

Faz 1b/1c ile **aynı iskelet**: encode=afferent havuzu (n=19.261, 20'si
seçiliyor), decode=efferent havuzu (n=1.489, 50'si seçiliyor), 10-D
Rastrigin (birincil) + Sphere (ikincil/hata-ayıklama), T=15, 30 paylaşılan
tohum, 4 substrate (flywire, er_null, degree_preserving_null,
weight_shuffle_null), main_budget=1500, sphere_budget=200,
tuning_budget=100, max_workers=2, THREADS_PER_WORKER=4 — ölçülmüş güvenli
bütçe (Faz 1c ile birebir aynı iterasyon×T maliyeti, bkz. runner dosyası
başlığı).

**Yeni serbest parametreler:** `ray_radius=0.3` (sabit, taranmıyor —
kapsam kontrolü), `CALIB_STEPS=100` (sabit). Taranan: `ray_scale ∈
{0.5, 1.0, 2.0}` × `decode_scale ∈ {0.2, 0.5, 1.0}` — Faz 1'in
step_scale/decode_scale ızgarasıyla aynı değerler, süreklilik için.

**Koşum sırası:** Önce `--smoke` (2 tohum, budget=50) ile boru hattı
doğrulanacak — sayısal hata, NaN, bariz bug var mı kontrolü. Smoke
sonucu bu dosyayı etkilemez (tescil zaten yazıldı), sadece kod
doğruluğunu kontrol eder. Ana koşumun bütçesi/tohum sayısı smoke'tan
SONRA değiştirilmeyecek.

## Hipotez

**Beklenti (koşum başlamadan önce yazıldı):** Faz 1c'de olduğu gibi net
değilim, ama malecns gözleminden dolayı hafif pozitif yönde bir önsezi
var: eğer connectome'un avantajı gerçekten "eğitilmiş okuma + yapılandırılmış
duyusal girdi" kombinasyonuna bağlıysa, gate bu sefer geçebilir. Ama şu
ihtimalleri de ciddiye alıyorum: (a) Rastrigin'in 10-D soyut uzayı,
malecns'in gerçek 2-D/3-D fiziksel "arazi"sinden yeterince farklı olabilir
— ışın kodlaması "sahne" hissi versede, connectome'un beklediği türden
uzamsal/zamansal tutarlılığa (retinotopi, hareket) sahip olmayabilir; (b)
tek-turluk kalibrasyon malecns'in çok-turlu DAgger + ES'inden çok daha
zayıf bir eğitim, yetersiz kalabilir; (c) Faz 1/1b/1c'nin flywire'ın
er_null'dan *kötü* çıkma eğilimi (bilgi eksikliği değil, potansiyel bir
dezavantaj) burada da devam edebilir.

## Durma/yorum kriteri

- Gate geçerse (flywire vs degree_preserving_null, p<0.05 ve |δ|>0.33):
  "eğitilmiş okuma + yapılandırılmış sahne" kombinasyonunun soyut
  optimizasyon görevlerine de taşınabildiğine dair ilk pozitif kanıt —
  bir sonraki adım olarak daha büyük kalibrasyon bütçesi (çok-turlu
  DAgger) veya gerçek uzamsal bir problem (kullanıcının geoteknik
  alanından) denenebilir.
- Gate geçmezse: fark muhtemelen "eğitim" değil "görev-mimari eşleşmesi"
  etkeninden kaynaklanıyor demektir — soyut R^n optimizasyonu, connectome'un
  ne kadar iyi eğitilirse eğitilsin, doğal yeteneğinin dışında kalıyor
  olabilir. Bu durumda Faz S1 de negatif sonuç listesine eklenir, malecns
  gözlemiyle çelişmez (ikisi de doğru olabilir: mimari eşleşmesi > eğitim).
- Her iki durumda da flywire vs er_null ikincil karşılaştırması özellikle
  izlenecek — Faz 1/1b/1c'nin tutarlı "flywire er_null'dan kötü" yönü bu
  sefer tersine döner mi.

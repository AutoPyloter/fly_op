# PROTOCOL v2 — Görev-Mimari Eşleşmesi Hipotezi

**Tarih:** 2026-09-17/18. PROTOCOL.md'nin yerine geçmiyor, onu tamamlıyor.
PROTOCOL.md'nin orijinal H1'i (connectome genel-amaçlı optimizasyonda
avantaj sağlar) dört fazda (1, 1b, 1c, S1) ve beş mimari denemesinde
(PSO×3, TRM, sinaptik plastisite, türevlenebilir rate-brain) sistematik
olarak reddedildi — bu sonuçlar geçerli, değiştirilmiyor. Bu belge, o
reddin *neden* olduğunu açıklayan yeni, ayrı bir hipotezi (H2) formalize
ediyor ve rotamızı oraya çeviriyor.

## 1. Motivasyon: dört veri noktası, tek eğri

2026-09-16/17'de aynı istatistiksel disiplinle (Wilcoxon işaretli-sıra +
Cliff's δ, gerçek connectome vs derece/ağırlık-korumalı null modeller)
dört farklı görev tipi test edildi:

| Görev tipi | Cliff's δ | Kapı testi |
|---|---|---|
| Soyut optimizasyon (Faz 1/1b/1c/S1, PSO, TRM, sinaps, rate-brain) | ~0 ile -0.34 arası | FAIL (hepsi) |
| Fonksiyon değeri tahmini (Fly-Surrogate) | 0.188 | FAIL (eşik altı, ama yön 8/8 tutarlı) |
| Sürekli duyusal-motor kontrol (malecns, F1 araba) | **1.000** | **PASS** (mutlak) |

Örüntü tek yönlü ve monoton görünüyor: görev, connectome'un evrimleştiği
işe (yoğun duyusal girdiden az sayıda motor çıktısına sürekli süzme) ne
kadar yakınsa, gerçek bağlantı şemasının rastgele bir ağa üstünlüğü o
kadar büyüyor.

## 2. Literatür: kimlerle yarışıyoruz, katkımız ne

Bu soruyu soran ilk kişi biz değiliz — ama bizim spesifik çerçevemiz
(sistematik görev-spektrumu, aynı istatistikle tek bir connectome
üzerinde çoklu görev tipi) literatürde net bir karşılığı olmayan bir
boşluk dolduruyor.

**Doğrudan öncüller:**

- **conn2res** (Suárez ve ark., *Nature Communications* 2024) — insan
  MRI-connectome'unu 500 derece/yoğunluk-korumalı rewired null modeliyle
  kıyaslayan, açık kaynaklı bir araç kutusu. Kritiklik noktasında
  (α=1) gerçek ağ, hafıza kapasitesi görevinde null'lardan anlamlı
  üstün (p=0.002). **Bizim null-model + istatistik metodolojimizin
  neredeyse birebir aynısı, ama insan connectome'u ve tek görev
  ailesiyle (hafıza kapasitesi), ve "hangi görev tipinde işe yarar"
  sorusunu sistematik olarak sormuyor.**
  [nature.com/articles/s41467-024-44900-4](https://www.nature.com/articles/s41467-024-44900-4)

- **Costi, Hadjiivanov, Dold, Hale, Izzo** (ESA Advanced Concepts Team,
  *Biomimetics* 2025) — FlyWire connectome'unu kaotik zaman-serisi
  tahmininde (üç-cisim problemi) reservoir olarak kullanıyor, 2×2
  faktöriyel tasarım (gerçek/rastgele topoloji × gerçek/rastgele
  ağırlık), Mann-Whitney-Wilcoxon. Bulgu: gerçek topoloji özellikle
  **aşırı uyuma karşı dayanıklılıkta** avantaj sağlıyor, düşük
  düzenlileştirmede. **Bizim Fly-Surrogate'imize çok yakın bir görev
  ailesi (fonksiyon/zaman-serisi tahmini), farklı bir metrik
  (aşırı uyum direnci) üzerinden benzer "mütevazı ama gerçek" bir
  sonuca varıyor.**
  [doi.org/10.3390/biomimetics10050341](https://doi.org/10.3390/biomimetics10050341)

- **Multifunctionality in a Connectome-Based Reservoir Computer**
  (IEEE 2023) — rewiring'in çok-görevli kapasiteyi düşürdüğünü, yüksek
  ortalama derece merkeziliğinin çok-görevliliği desteklediğini
  gösteriyor. Bizim "görev tipi önemli" bulgumuzla aynı yönde ek kanıt.

**Bizim katkımız (literatürde net karşılığı olmayan kısım):**

1. **Aynı connectome'u, aynı istatistiksel çerçeveyle, sistematik olarak
   farklı görev tiplerinde test etmek** — conn2res tek bir görev
   ailesinde kalıyor, Costi ve ark. tek bir görev ailesinde kalıyor.
   Biz dört farklı görev tipini (soyut optimizasyon → fonksiyon tahmini
   → gömülü kontrol) AYNI FlyWire/MaleCNS connectome'unda, aynı null
   model ailesiyle, aynı istatistikle kıyasladık.
2. **Negatif sonucu pozitif sonuç kadar titizlikle raporlamak** — hem
   conn2res hem Costi ve ark. sadece "işe yaradığı" görevleri
   raporluyor; biz "işe yaramadığı" dört fazı da (Faz 1-S1, PSO, TRM,
   sinaps) aynı titizlikle, `results/negative/` altında saklıyoruz.
3. **Metodolojik bir uyarı literatüre eklenebilir**: bugün bulduğumuz
   "encode/decode nöronları bağlantı-doğrulanmadan seçilirse sinyal
   hiç ulaşmayabilir" bulgusu (rate-brain pilot v1→v2) —
   incelediğimiz hiçbir makale bunu açıkça tartışmıyor, muhtemelen
   çoğu çalışma görme/motor gibi zaten yoğun-bağlantılı popülasyonları
   kullandığı için hiç karşılaşmamışlar.

## 3. Hipotez (H2)

**Connectome-driven bir substrate'in null modellere göre ölçülebilir
avantajı, görevin connectome'un evrimleştiği fonksiyona (yoğun
duyusal girdiden az sayıda motor çıktısına sürekli, kapalı-döngü süzme)
ne kadar yakın olduğuyla monoton ilişkilidir — mimari zenginlik
(eğitim yöntemi, kodlama karmaşıklığı, sinaptik plastisite) bu ilişkiyi
değiştirmez, sadece görev tipi değiştirir.**

**Yanlışlanabilirlik kriteri:** Görev spektrumunda (soyut → gömülü)
sıralı olarak test edilen ek noktalar, Cliff's δ'da monoton bir artış
GÖSTERMEZSE (örn. ara bir görev, uç noktalardan daha yüksek veya daha
düşük δ verirse tutarsız şekilde), H2 zayıflar.

## 4. Metodoloji (değişmeyen kısım — bugün doğrulandı)

- Substrate: `flyopt.variants.rate_brain.RateBrain` — türevlenebilir
  leaky-rate, Dale yasası korunmuş (`w=sign×softplus(gain)`), gerçek
  backprop.
- Encode/decode seçimi: **`select_connected_encode_decode` ile ZORUNLU
  bağlantı doğrulaması** (2026-09-17'nin en önemli metodolojik dersi —
  rastgele seçim artık yasak).
- Null model: `er_null` (birincil), `degree_preserving_rewire`
  (ikincil, PROTOCOL.md'nin ana null'u) — aynı düğüm kümesi üzerinde.
- İstatistik: Wilcoxon işaretli-sıra + Cliff's δ, eşik |δ|>0.33,
  p<0.05 — PROTOCOL.md'yle birebir aynı kapı kriteri, süreklilik için.
- Minimum 8 tohum (bugünkü pilotların standardı); resmi bir faz için
  PROTOCOL.md'nin 30-tohum kuralına dönülmeli.

## 5. Planlanan spektrum noktaları — GÜNCELLEME (2026-09-18): H2'nin basit hali yanlışlandı

| # | Görev | δ (er_null) | δ (degree_preserving) | Kapı | Yapısal özellikler |
|---|---|---|---|---|---|
| 1 | Soyut optimizasyon (Rastrigin) | ~0 ile -0.34 | — | FAIL | ne uzamsal ne kapalı-döngü |
| 2 | Fonksiyon tahmini (Fly-Surrogate) | 0.188 | — | FAIL (eşik altı) | ne uzamsal ne kapalı-döngü (ama gerçek fonksiyon) |
| 3 | Uzamsal fonksiyon tahmini (şev stabilitesi, tek-atış) | er_null: 0.906 (n=8) | **degree_preserving: 0.884 (n=30, RESMİ Mann-Whitney p≈0.000000)** ✅✅✅ (weight_shuffle: 0.305, n=16, aynı yönde) | **RESMİ PASS — PROTOCOL.md'nin n=30 kapısını tam olarak geçen İLK bulgu** | uzamsal YAPI var, kapalı-döngü YOK — **etki TOPOLOJİden geliyor, ağırlık değerlerinden değil** |
| 3b | Aynı görev, SÜREKLİ/çok-adımlı versiyonu (v1-v3: bozuk, %43.6 geçersiz-başlangıç hatası; v4: DÜZELTİLDİ) | **-0.156** | — | **FAIL** | uzamsal YAPI var, kapalı-döngü VAR — ama δ negatif, temiz sonuç |
| 4 | Kesikli kapalı-döngü kontrol (cartpole, n=16, gradyan kırpmalı) | **0.133** | — | **FAIL** | kapalı-döngü var, uzamsal YAPI YOK |
| 5 | Sürekli gömülü kontrol (malecns) | 1.000 (degree: -0.156, weight_shuffle: -0.312 — İKİSİ DE KAYBOLDU) | **-0.156 (KAYBOLDU)** ⚠️⚠️ | **FAIL degree_preserving'e karşı** | uzamsal YAPI VE kapalı-döngü, ikisi de var — ama er_null bulgusu muhtemelen derece-dağılımı artefaktı
| 6 | CX-modulated tek-ajanlı arama (Rastrigin) | -0.562 | **+0.125 (KAYBOLDU)** ⚠️ | **FAIL degree_preserving'e karşı** | ne uzamsal ne kapalı-döngü — er_null bulgusu muhtemelen derece-dağılımı artefaktı |

**2026-09-18/19 gece yarısı ek dersi (GÜNCELLENDİ — malecns de test
edildi):** er_null'a karşı görülen bir sinyal otomatik olarak "gerçek
bağlantı özgüllüğü" anlamına gelmiyor — FlyWire'ın heterojen derece
dağılımı TEK BAŞINA (hangi nöronun kime bağlı olduğu bilgisi olmadan)
ER'den farklı davranış üretebilir.

- Satır 6 (CX-modulated): er_null δ=-0.562 → degree_preserving δ=+0.125 — **KAYBOLDU**
- **Satır 5 (malecns): er_null δ=1.000 (oturumun/projenin EN GÜÇLÜ bulgusu) → degree_preserving δ=-0.156 (p=0.375) — TAMAMEN KAYBOLDU, hatta yön hafifçe ters döndü.** flywire/degree_preserving_null medyanları neredeyse aynı (0.142 vs 0.143). ~4.7 saatlik 8-tohum GPU koşumuyla doğrulandı (`C:/projeler/fly_demos/malecns/multiseed_compare_degreenull.py`).
- Satır 3 (şev stabilitesi): er_null δ=0.906 → degree_preserving δ=1.000 — **HAYATTA KALDI, GÜÇLENDİ** — şu ana kadarki TEK sağlam pozitif bulgu.

**Bu, oturumun (ve muhtemelen tüm FlyOpt projesinin) en önemli
metodolojik dersi:** İKİ AYRI er_null-pozitif bulgu (biri "temiz" biri
"mükemmel/δ=1.000" olarak nitelendirilmişti) degree-preserving null'a
karşı TAMAMEN kayboldu; sadece BİR tanesi (uzamsal, tek-atış
toplu-regresyon) sağlam kaldı. **Bundan sonra herhangi bir er_null
sonucu, degree_preserving_rewire'a karşı doğrulanmadan "H1 lehine
kanıt" olarak sayılmamalı** — bu, PROTOCOL.md'nin zaten önceden
degree_preserving_rewire'ı birincil null olarak seçmiş olmasının ne
kadar isabetli olduğunu da gösteriyor; bu oturumun H2 spektrum
çalışması pratik nedenlerle er_null'a kaymıştı, bu bir hataydı.

**H2'nin ilk hali (tek boyutlu "soyut→gömülü" spektrum, monoton artış
iddiası) 2026-09-18'de cartpole sonucuyla YANLIŞLANDI**, ve H2'
("uzamsal yapı yeter, kapalı-döngü önemsiz") **3b'nin düzeltilmiş
sonucuyla da YANLIŞLANDI** — aynı uzamsal görev (şev stabilitesi),
sadece kapalı-döngü/iteratif hale getirilince δ=0.906'dan δ=-0.156'ya
düştü. Yani "uzamsal yapı" tek başına da yeterli değilmiş; tek-atış
toplu-regresyon çerçevesinin KENDİSİ önemli bir değişkenmiş.

**Durum (2026-09-18 gece sonu): iki basit hipotez de (H2, H2')
yanlışlandı. Üçüncü, daha mütevazı bir çalışma hipotezi (H2''):**
Connectome'un ölçülebilir avantajı şu ana kadar sadece **(a) tek-atış,
toplu-regresyon eğitimli** görevlerde (şev stabilitesi tek-atış) VEYA
**(b) çok büyük, zengin bir eğitim yatırımı almış gerçek-dünya
görevlerinde** (malecns — DAgger + ES, gerçek lidar) ortaya çıkıyor.
Ne "uzamsal olmak" ne "kapalı-döngü olmak" tek başına yeterli;
muhtemelen eğitim rejiminin zenginliği/ölçeği ayrı, bağımsız bir
değişken. Bu spektrum artık tek eksenli değil — en az üç boyutlu
(uzamsal yapı × kapalı-döngü × eğitim zenginliği) bir uzay gibi
görünüyor, ve elimizdeki 6 veri noktası bu uzayı yeterince
örneklemiyor. **Yeni deneyler tasarlarken bu üç boyutu ayrı ayrı
kontrol etmek gerekecek.**

## 6. Ek veri noktası (2026-09-18 gece): eğitimli CX-nüdge + gerçek PSO sürüsü

Kullanıcının "neden PSO kullanmıyorsun" sorusu üzerine tek-ajanlı
CX-modulated deneyi (δ=-0.562, 16/16 tohum, en temiz bulgu — bkz.
EXPERIMENTS.md) gerçek bir 12-parçacıklı PSO sürüsüne gömüldü. Yolda
kritik bir NaN/patlayan-gradyan bug'ı bulundu ve düzeltildi (bkz.
EXPERIMENTS.md 2026-09-18 girişi — eğitim ve sürü-koşumu artık ayrı
`episode_steps` kullanıyor). Düzeltme sonrası:

| # | Görev | δ (n=8) | Kapı | Not |
|---|---|---|---|---|
| 6 | Klasik gbest-PSO + eğitimli CX-nüdge (sürü, encode-seed=17000) | -0.406 | FAIL (p=0.31) | tek-ajanlı bulguyla AYNI yön |
| 6b | Aynı, bağımsız alt-graf (encode-seed=18000) | -0.375 | FAIL (p=0.38) | 6 ile TUTARLI, bağımsız çekilim |
| 6c | Chemotaxis (merkezi-olmayan, gbest'siz "koku" çekimi) | +0.250 | FAIL (p=0.95) | zayıf, TERS yönde |
| 6d | Lek (top-K en yakın çekici) | -0.156 | FAIL (p=0.64) | zayıf |
| 6e | Ring-lbest (literatür-standart halka topolojisi) | -0.062 | FAIL (p=0.95) | neredeyse sıfır |

**Ön yorum:** Tek-ajanlı bulgu sürüye taşınınca sinyal ZAYIFLIYOR ama
KAYBOLMUYOR (δ hâlâ null-lehine, iki bağımsız alt-grafta tutarlı) —
muhtemelen sürünün kendi PSO-gürültüsü (12 parçacık × rastgele
başlangıç × pbest/gbest dinamiği) sinyali seyreltiyor. Alternatif
sürü-organizasyonları (chemotaxis/lek/ring_lbest) bu görevde gbest'in
zayıf tutarlılığını YAKALAYAMADI — belki çünkü CX-nüdge sinyali klasik
gbest'in "hepsi aynı hedefe çekilme" yapısıyla daha uyumlu, merkezi
olmayan mekanizmalarda kayboluyor.

**n=16 sonucu (nihai):** 6/6b'nin birleşik n=16 testi δ=-0.008'e
düştü (p=0.940) — n=8'deki -0.406 tamamen sönmüş, oturumun tekrar
eden "küçük-n umut → büyük-n sönme" kalıbının bir örneği daha.
**Sonuç: tek-ajanlı CX-modulated bulgusu (δ=-0.562, 16/16 tohum,
en temiz bulgu) gerçek ama sürüye TAŞINMIYOR** — klasik PSO'ya
gömülünce kaybolan bir etki. Bu, sürünün kendi stokastik dinamiğinin
(12 bağımsız parçacık, rastgele başlangıç, pbest/gbest gürültüsü)
CX'in ince, tek-ajanlı sinyalini istatistiksel olarak boğduğunu
düşündürüyor — mimari uyumsuzluğun kendisi (H2'') burada da geçerli:
CX-modulated mekanizma "tek sürekli-durum ajanı" görev-mimarisiyle
eşleşiyor, "çoklu-bağımsız-parçacık sürüsü" mimarisiyle eşleşmiyor.

**Ek sağlamlık kontrolü — gain'i gerçekten eğit:** Orijinal tek-ajanlı
bulguda (δ=-0.562) `readout_gain` hiç gradyan almıyordu (ayrı bug,
EXPERIMENTS.md'de belgelendi). Uçtan-uca düzeltme sonrası (gain artık
gerçek görev-kaybından eğitiliyor): δ=-0.250, n=8, FAIL (eşik altı)
ama YÖN AYNI. İki farklı eğitim rejimi aynı yönde sinyal veriyor —
bulgunun eğitim-detayına aşırı duyarlı bir artefakt olmadığını
destekliyor.

**Bu araştırma hattının kapanışı:** CX-modulated tek-ajanlı sinyal
(δ=-0.562/−0.250, eğitim rejimine göre) gerçek ve tutarlı, ama (a)
SADECE tek-ajanlı mimaride ölçülüyor, (b) hiçbir sürü-organizasyonuna
(klasik PSO, chemotaxis, lek, ring_lbest) taşınmıyor. Kullanıcının
"alternatif PSO mimarisi" isteği tam olarak karşılandı — 4 farklı
sürü-organizasyonu test edildi, hiçbiri anlamlı sinyal göstermedi.

**Sıradaki en değerli iş (2026-09-18 gece güncellemesi — ikisi de
yapıldı):**
1. Cartpole gradyan kırpma + 16 tohum: yapıldı, δ=0.133'e düştü
   (v1'in -0.219'undan biraz düzeldi ama hâlâ FAIL/anlamsız).
2. Şev stabilitesini çok-adımlı/kapalı-döngü hale getirmek: yapıldı,
   3 turda düzeltildi. v1-v3 bir görev-tanımı hatası (%43.6 geçersiz
   başlangıç) yüzünden bozuktu; v4'te düzeltilip (`sample_valid_point`,
   `benchmarks_geo.py`) temiz bir sonuç alındı: **δ=-0.156, FAIL**,
   üstelik tek-atış versiyonun (δ=0.906) tam tersi yönde. Bu, H2/H2'nin
   ikisini de yanlışladı (§5'teki H2'' güncellemesine bakın).

**Şimdiki en değerli iş:** Neden tek-atış (toplu regresyon) ile
kapalı-döngü (iteratif arama) aynı görevde bu kadar farklı sonuç
veriyor sorusunu izole etmek — iki aday açıklama var: (a) eğitim
verisi hacmi/çeşitliliği (tek-atış çok daha fazla (x, hedef) örneği
görüyor), (b) toplu regresyonun "ne olursa olsun bir tahmin ver"
zorlaması ile iteratif aramanın "kendi hatasını biriktirme" riski
arasındaki fark. Bunu ayırt etmek için: kapalı-döngü versiyonunu
tek-atış'ınkiyle AYNI toplam örnek sayısını görecek şekilde eğitip
(episode sayısını çok artırıp) tekrar test etmek.

## 6. Verimlilik notları

- **rate_brain.py pipeline'ı GPU'ya taşınabilir** — şu an CPU'da 3000
  düğümlük alt-grafla çalışıyor (~dakikalar), tam 139k connectome'a
  GPU'da ölçeklenirse (fly_demos/malecns'in zaten kanıtladığı gibi)
  hem alt-graf örnekleme ihtiyacı ortadan kalkar hem de tohum başına
  süre kısalır.
- **Alt-graf örneklemesi bir konfor, zorunluluk değil** — BFS-tabanlı
  bağlantı garantisi tam grafta da çalışır, sadece hesap maliyeti artar.
- **Şev stabilitesi + malecns ortak bir ara nokta olabilir**: ikisi de
  "gerçek fiziksel/uzamsal yapı" taşıyor ama şev stabilitesi çok daha
  ucuz (CPU, saniyeler) — spektrumun orta noktalarını doldurmak için
  malecns'ten çok daha verimli bir araç.

## 7. SABAH ÖZETİ (2026-09-19) — gece boyunca en önemli gelişme

Bu gece, kullanıcının "alternatif PSO mimarisi" isteğini karşılarken
(4 sürü-topolojisi test edildi, hiçbiri sinyal vermedi — bkz. §6) yolda
**projenin şu ana kadarki en önemli metodolojik keşfi** yapıldı:

**er_null null modeli, TEK BAŞINA, "H1 lehine kanıt" için yeterli
değil.** İki ayrı er_null-pozitif bulgu — CX-modulated (δ=-0.562,
"en temiz bulgu" sanılıyordu) ve **malecns (δ=1.000, projenin EN GÜÇLÜ
bulgusu)** — PROTOCOL.md'nin asıl birincil null modeli olan
`degree_preserving_rewire`'a (derece dağılımını tam koruyan
Maslov-Sneppen karıştırması) karşı test edildiğinde **TAMAMEN
KAYBOLDU**. **malecns AYRICA üçüncü bir null'a (weight_shuffle — topoloji
birebir aynı, sadece ağırlık değerleri karışık) karşı da test edildi ve
YİNE kayboldu (δ=-0.312)** — iki farklı, birbirinden bağımsız null
modelinin TUTARLI biçimde aynı (zayıf-negatif) sonucu vermesi, bunun
tesadüf olmadığını gösteriyor: er_null'un kendisi anormal derecede
"kolay yenilebilir" bir taban çizgisi (tamamen düzensiz bağlantı hiçbir
gerçekçi ağ istatistiği taşımıyor, bu yüzden işlevsiz bir dinamiğe yol
açıyor) — ve BUNUNLA karşılaştırıldığında hemen her graf iyi görünüyor.

**Tek ayakta kalan, güçlenen bulgu: şev stabilitesi (tek-atış toplu
regresyon).** Üç bağımsız null modele karşı test edildi:
- er_null: δ=0.906, PASS
- degree_preserving_rewire: δ=1.000, PASS (mükemmel ayrım)
- weight_shuffle (topoloji sabit, sadece ağırlıklar karışık): δ=0.305 (n=16), FAIL ama AYNI YÖNDE

Üçü de aynı yönde, ikisi güçlü PASS — bu artık projenin EN SAĞLAM
bulgusu, ve etkinin **topolojiden (kim kime bağlı) geldiğini, spesifik
sinaptik ağırlık değerlerinden değil** düşündürüyor.

**Sonraki oturum için öncelik sırası:**
1. Bu üç-null tablosunu PROTOCOL.md'nin ana anlatısına (H1/H0 özet
   bölümü) yansıt — "connectome bazı görevlerde avantaj sağlar" iddiası
   artık SADECE şev-stabilitesi tek-atış görev ailesiyle sınırlı.
2. ~~malecns'i üçüncü null'a (weight_shuffle) karşı da test et~~ —
   **TAMAMLANDI (2026-09-19 gece):** δ=-0.312, FAIL, degree_preserving
   ile TUTARLI (ikisi de zayıf-negatif). malecns'in üç-null tablosu artık
   tam: er_null PASS (1.000), degree_preserving FAIL (-0.156), weight_shuffle
   FAIL (-0.312) — **iki bağımsız gerçekçi null'un ikisi de aynı yönde
   anlaşması, malecns'in δ=1.000 bulgusunun neredeyse kesinlikle bir
   er_null-taban-çizgisi artefaktı olduğunu gösteriyor.**
3. Bundan sonra hiçbir yeni er_null "pozitif" bulgu, degree_preserving_rewire'a
   karşı doğrulanmadan rapor edilmemeli — bu, artık standart protokol.
4. Şev stabilitesinin NEDEN diğer görevlerden farklı davrandığını
   (uzamsal + tek-atış + toplu-regresyon kombinasyonu mu, yoksa başka
   bir şey mi) izole edecek yeni bir görev tasarlanabilir.
5. **MEKANİZMA BULUNDU (2026-09-19 gece, `src/analyze_null_graphs.py`
   ile):** er_null'un neden bu kadar "kolay yenildiği" artık açık —
   FlyWire'ın heterojen derece dağılımı (hub nöronlar + %2.7-3.0
   neredeyse-izole nöron) er_null'da TAMAMEN kayboluyor (homojen,
   max derece 66-69 vs gerçek 6660-7570), degree_preserving VE
   weight_shuffle ise bunu koruyor. Resiprosite (%15.65 flywire,
   %0.02 er_null, %0.22 degree_preserving, %15.65 weight_shuffle)
   AYRIŞTIRICI DEĞİL — çünkü weight_shuffle resiprositeyi tam koruyor
   ama etkiyi GERİ GETİRMİYOR. Sonuç: **er_null'un zayıflığı hub-nöron
   /heterojen-derece eksikliğinden geliyor, resiprositeden değil** —
   bu, connectome'a özgü bir avantajdan çok, "her heterojen-dereceli
   ağ er_null'u yener" türünden yapısal bir olgu.
6. **Proje-seviyesinde büyük resim:** Sabah bu özeti okuyunca, FlyOpt'un
   PROTOCOL.md'deki orijinal negatif tezinin (soyut optimizasyonda
   connectome avantajı yok) yanına şimdi çok daha güçlü bir ikinci ders
   eklenmiş oluyor: **"gömülü/fiziksel görevlerde bile görülen
   'avantaj', çoğunlukla gerçek bağlantı özgüllüğü değil, er_null'un
   zayıf bir taban çizgisi olmasından kaynaklanıyor."** Tek istisna
   (şev stabilitesi) bile sadece iki testte güçlü, üçüncüsünde zayıf —
   yani "connectome'un ölçülebilir katkısı" iddiası bu oturum sonunda
   oturumun başındakinden çok daha dar ve çok daha şüpheli bir alana
   sıkıştı.

## 8. RESMİ SONUÇ (2026-09-20) — şev stabilitesi PROTOCOL.md'nin n=30 kapısını GEÇTİ

Kullanıcının "test et" talimatıyla, tek ayakta kalan bulgu PROTOCOL.md'nin
TAM ön-kayıtlı kriterine göre test edildi: **30 tohum, degree_preserving_rewire,
Mann-Whitney U testi** (oturumun geri kalanında kullanılan Wilcoxon
işaretli-sıra değil — protokolün gerçek harfi).

```
real medyan = 0.1173   null medyan = 0.1731
Mann-Whitney U (ön-kayıtlı resmi test): p = 0.000000
Wilcoxon işaretli-sıra (oturum konvansiyonu): p = 0.000001
Cliff's delta = 0.884
RESMİ KAPI: PASS
```

**Bu, FlyOpt'un kuruluşundan beri aradığı, PROTOCOL.md'nin kendi
kurallarına göre resmi olarak geçerli İLK VE TEK pozitif sonuç.**
İki farklı istatistiksel test (eşleşmemiş Mann-Whitney, eşleşmiş
Wilcoxon) neredeyse aynı p-değerini veriyor — test seçimi sonucu
değiştirmiyor, bulgu sağlam. Cliff's delta=0.884, "büyük etki"
eşiğinin de üzerinde.

**Sonuç:** Gecenin başında sorulan "sinek başardı mı?" sorusunun nihai
cevabı — genel-amaçlı optimizasyonda hayır, ama **şev-stabilitesi gibi
uzamsal, tek-atış toplu-regresyon görevlerinde EVET, ve bu artık
protokolün kendi en katı standardına göre resmen doğrulanmış bir evet.**

## 9. DÜZELTME (2026-09-20) — "herhangi bir heterojen graf yeter" hipotezi YANLIŞLANDI

§8'deki yorumda "etki muhtemelen fly'a özgü değil, herhangi bir
heterojen-dereceli ağ işe yarar" denmişti. Bunu doğrudan test ettim:
`scale_free_null` (fly'ın derece dizisini KULLANMAYAN, sıfırdan
Pareto-örneklenmiş, sadece düğüm/kenar sayısı eşleşen sentetik
hub-baskın graf) ile karşılaştırma yapıldı.

**Sonuç: gerçek connectome bu sentetik grafı da açık arayla yeniyor**
(Mann-Whitney p=0.00295, Cliff's delta=0.844, n=8, PASS).

**Üç null'un birlikte anlattığı düzeltilmiş hikaye:**
- degree_preserving_rewire (fly'ın TAM derece dizisini korur, kim-kime-bağlıyı bozar) → gerçek KAZANIYOR (δ=0.884, n=30)
- weight_shuffle (kim-kime-bağlıyı korur, ağırlıkları bozar) → gerçek ZAR ZOR kazanıyor (δ=0.258, n=30, FAIL)
- scale_free_null (SADECE düğüm/kenar sayısını korur, fly'ın derece dizisinin kendisini bozar) → gerçek AÇIKÇA kazanıyor (δ=0.844, n=8)

**Düzeltilmiş sonuç: yük taşıyan şey "herhangi bir heterojen derece
dağılımı" değil, FLY'IN KENDİ SPESİFİK DERECE DİZİSİ.** Bu, önceki
"sadece ağ-teorisi genel bir olgu" yorumundan çok daha güçlü ve daha
ilginç bir bulgu — fly connectome'unun derece yapısı, jenerik bir
güç-yasası örneklemesinden bile ayırt edilebilir bir düzenlilik
taşıyor.

**n=30'a çıkarıldı, DOĞRULANDI:** Mann-Whitney p=0.000001, Wilcoxon
p=0.000003, Cliff's delta=0.738 (n=8'deki 0.844'ten hafifçe düştü ama
hâlâ ezici anlamlı) — **PASS, tam protokol gücünde.** FlyOpt'un artık
İKİ bağımsız, n=30-doğrulanmış pozitif bulgusu var: degree_preserving_rewire'a
karşı (δ=0.884) ve fly'a özgü olmayan sentetik hub-baskın grafa karşı
(δ=0.738). İkisi birlikte, "fly'ın kendi spesifik derece dizisi özel"
hipotezini tam protokol gücünde destekliyor.

**Mekanizma denemesi (giriş/çıkış derece korelasyonu) YANLIŞLANDI:**
Gerçek grafta giriş/çıkış derecesi güçlü korelasyonlu (Spearman r=0.70);
bağımsız-örneklenmiş scale_free_null'da bu yok (r≈-0.03). Bu
korelasyonu düzelten yeni bir sentetik model (`scale_free_correlated_null`,
r=0.67 elde edildi) test edildiğinde **gerçek hâlâ açık farkla
kazandı** (δ=0.719, n=8, PASS) — bağımsız modelden neredeyse hiç
farklı değil. Yani "sadece giriş/çıkış korelasyonu" hipotezi
yanlışlandı; fly'ın derece dizisinde iki basit parametreyle
yakalanamayan başka bir düzenlilik var, henüz bulunamadı. "Fly'ın
kendi derece dizisi özel" sonucu sağlam kalıyor, ama NEDEN sorusu açık.

**Tam parite için weight_shuffle de n=30'a çıkarıldı:** δ trend'i net
biçimde zayıflıyor (n=8: 0.375 → n=16: 0.305 → n=30: 0.258, p=0.088,
FAIL) — degree_preserving'in aksine (n=8'den n=30'a neredeyse hiç
zayıflamadı: 1.000→0.884, hâlâ ezici anlamlı). Bu, etkinin ağırlık
değerlerinden değil GERÇEK TOPOLOJİDEN geldiği yorumunu güçlendiriyor,
ama resmi sonucu değiştirmiyor: **PROTOCOL.md'nin talep ettiği testte
(degree_preserving_rewire, n=30) bulgu kesin ve sağlam.**

## 10. Mekanizma bulundu (kısmen): modülerlik boşluğun yarısını açıklıyor (2026-09-20/21)

"Fly'ın derece dizisi neden özel" sorusuna üç mekanistik hipotez
test edildi:
1. **Giriş/çıkış derecesi korelasyonu** (r=0.70 gerçekte) — bağımsız
   örneklenmiş sentetik graf (r≈-0.03) kaybetti; korelasyonu düzelten
   sentetik graf (r=0.67) de AYNI ÖLÇÜDE kaybetti (δ=0.719) — **hipotez
   YANLIŞLANDI**.
2. **Yapısal ölçüm:** gerçek grafın modülerliği (Louvain Q=0.357) HER
   null modelde (degree_preserving dahil, Q=0.069) neredeyse sıfıra
   iniyor.
3. **`community_preserving_rewire`** (TAM derece dizisi + TAM
   modülerlik, Q=0.365 elde edildi) test edildi: **n=30'da δ=0.429,
   p=0.004, ANLAMLI** — n=8'deki (δ=0.375, FAIL) sonuçtan güçlenerek
   geldi.

**Nihai tablo (2026-09-21 güncellemesi: artık HER satır n=30):**
| null | ne koruyor | δ (n=30) | p (Mann-Whitney) | anlamlı mı |
|---|---|---|---|---|
| degree_preserving | TAM derece dizisi | 0.884 | ≈0.000000 | EVET |
| scale_free | jenerik heterojenlik | 0.738 | ≈0.000001 | EVET |
| scale_free_correlated | + giriş/çıkış korelasyonu | 0.604 | 0.00006 | EVET |
| **community_preserving** | **TAM derece + TAM modülerlik** | **0.429** | **0.00443** | **EVET** |
| weight_shuffle | TAM topoloji, ağırlık hariç | 0.258 | 0.0877 | HAYIR |

**Sonuç: modülerlik etkiyi ~yarıya indiriyor (0.88→0.43) ama tam
kapatmıyor.** Geriye anlamlı bir artık boşluk kalıyor — muhtemelen
motif dağılımı, kümeler-arası bağlantı deseni ya da fly'ın tam
kendine özgü wiring'inde saklı, henüz izole edilemedi. "Neden özel"
sorusu artık kısmen (yaklaşık yarı yarıya) cevaplı.

Tüm bulgular interaktif bir özet sayfasında toplandı:
https://claude.ai/artifact/2pE791tBWxsfVPLRL8CpSb

## 11. Tür-ötesi genelleme testi: TEKRARLANMADI (2026-09-21)

Kullanıcının Gemini beyin fırtınasından gelen "türler arası
kıyaslama" önerisi (gerçekçi olarak C. elegans'a daraltılmış hali)
test edildi: Cook ve ark. 2019 C. elegans hermafrodit somatik sinir
sistemi (272 nöron) — yapısal olarak FlyWire'a çarpıcı derecede
benzer (Q=0.381 vs 0.357, kümeleme=0.329 vs 0.276) — AYNI görev/mimari
ile 3 null'a (er_null, degree_preserving, community_preserving) karşı
test edildi.

**Sonuç: HİÇBİRİNDE anlamlı fark yok** (δ=0.156/0.094/-0.250, n=8,
hepsi FAIL). FlyWire bulgusu bu ilk denemede C. elegans'a genellenmedi.
Parametreler C. elegans ölçeğine göre kasıtlı olarak yeniden
ayarlanmadı (PROTOCOL.md kuralı: negatif sonucu hiperparametre
kovalayarak değiştirmeye çalışma) — bu yüzden sonuç kesin değil ama
FlyWire bulgusunun kapsamı şimdilik türe/görev-kombinasyonuna özgü
kabul edilmeli, genel bir "modüler biyolojik ağ" iddiası yapılamaz.

## 12. Gecenin genel durumu (2026-09-21 sabahı)

Bu gece üç ayrı araştırma hattı ilerledi:
1. **Alternatif PSO mimarileri** (kullanıcı isteği) — 4 sürü-topolojisi
   test edildi, hiçbiri sinyal vermedi; yolda kritik bir NaN eğitim
   bug'ı bulunup düzeltildi.
2. **er_null yetersizliği keşfi** — projenin iki en güçlü bulgusu
   (malecns δ=1.000, CX-modulated δ=-0.562) daha sıkı nulllara karşı
   tamamen çöktü; sadece şev-stabilitesi ayakta kaldı ve PROTOCOL.md'nin
   TAM resmi kriterine göre (n=30, degree_preserving, Mann-Whitney)
   doğrulandı (δ=0.884, p≈0).
3. **Mekanizma avı** — giriş/çıkış korelasyonu yanlışlandı; modülerlik
   boşluğun yarısını açıklıyor (n=30 doğrulandı, δ=0.429); C. elegans'a
   genelleme bu ilk denemede başarısız oldu.

**Proje artık şurada duruyor:** FlyOpt'un tek resmi, protokole uygun
pozitif bulgusu — şev stabilitesi tek-atış toplu regresyonu, FlyWire
connectome'una özgü (henüz başka türe genellenmedi), kısmen (yaklaşık
yarısı) modülerlikle açıklanan bir avantaj. Genel-amaçlı optimizasyonda
H0 (connectome avantajı yok) hâlâ geçerli ve bu gece daha da
güçlendi (malecns/CX-modulated'in çöküşüyle).

## 13. İKİNCİ RESMİ PASS + iki ayrı eksende genelleme testi (2026-09-21, gün içi)

**Görev-genellenebilirlik (PASS):** §12'nin "henüz başka göreve/türe
genellenmedi" notu kısmen değişti — konsol kiriş tasarımı (yapısal
mühendislik, şev stabilitesinden bağımsız bir fiziksel alan) AYNI dişi
FlyWire FAFB alt-grafında, aynı mimari/null ile n=30'a çıkarıldı:

```
real medyan=0.1183  degree_preserving medyan=0.1824
Mann-Whitney p=0.000046   Wilcoxon p=0.000003   Cliff's δ=0.613
gate: PASS
```

Bu, PROTOCOL.md'nin tam n=30 kapısını geçen **ikinci** bağımsız bulgu.
Sonuç: avantaj tek bir görevin tuhaflığı değil, en azından bu iki
tek-atış vektör-regresyonu görevinde tutarlı.

**Tür/birey-genellenebilirlik (iki ayrı test, ikisi de NEGATİF):**
| Eksen | Test edilen | δ (n=8) | Gate |
|---|---|---|---|
| Tür | C. elegans hermafrodit somatik sinir sistemi (§11) | 0.156 (er_null) / 0.094 (degree_preserving) / -0.250 (community_preserving) | FAIL (üçü de) |
| Cinsiyet/birey | Erkek Drosophila CNS connectome (fly_demos/malecns'in ham verisi, bizim resmi metodolojimizle) | 0.250 (degree_preserving) | FAIL |

**Genel resim:** Avantaj **dişi FlyWire FAFB'nin bu spesifik
3000-düğümlük alt-grafına özgü** görünüyor — ama o alt-graf üzerinde
birden fazla göreve (şev stabilitesi + kiriş tasarımı) genelliyor. Hem
farklı tür (C. elegans) hem aynı türün farklı bireyi/cinsiyeti (erkek
CNS) bu avantajı göstermiyor. Bu, "genel olarak biyolojik/modüler ağlar
avantajlıdır" iddiasını değil, "bu spesifik connectome örneğinin bu tip
görevlerde ölçülebilir bir avantajı var, ama bu her connectome'a
genellenen bir özellik değil" iddiasını destekliyor — kapsamı daraltan
ama daha kesin, daha savunulabilir bir sonuç.

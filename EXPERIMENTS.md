# Deney Günlüğü

Kural (PROTOCOL.md §7.1): her koşumdan önce beklenti buraya yazılır. Sonuç
görüldükten sonra beklenti metni değiştirilmez — yeni giriş eklenir.

Format: her girişte tarih, faz, ne bekleniyor, sonuç (koşulduktan sonra
eklenir), ve varsa şüphe/doğrulama notları.

---

## 2026-09-14 — Proje başlangıcı

**Faz:** 0 (Altyapı)

**Durum:** Proje iskeleti kuruldu (`src/flyopt`, `data/`, `literature/`,
`preregistration/`, `results/`). Henüz veri yok, henüz simülasyon yok, henüz
hiçbir optimizasyon iddiası yok.

**Beklenti:** Yok — bu bir altyapı girişi, deney değil.

**Bloklayıcı:** FlyWire v783 verisinin indirilmesi kullanıcı eylemi gerektirir
(lisans kabulü). Bkz. `data/raw/README.md`. Bu olmadan gerçek connectome
substrate'i (`FlyWireSubstrate`) test edilemez; null model substrate'leri
(`ERNullSubstrate`, `DegreePreservingNullSubstrate`, `WeightShuffleNullSubstrate`,
`SignFlipNullSubstrate`) veri gerektirmez ve sentetik seyrek graflarla test
edilebilir/edildi.

**Sonraki adım:** Kullanıcı veriyi indirdikten sonra `flyopt.data.validate`
çalıştırılıp ağ istatistikleri ölçülecek; ardından LIF simülatörünün brian2 /
vektörize versiyonları arasındaki denklik testi yazılacak.

**Güncelleme (aynı gün, ilerleyen saat):** Faz 0 iskeleti kuruldu, 16 test
geçti — Brian2 referansı ile torch-vektörize LIF motoru arasında spike-spike
tam eşleşme dahil (Faz 0 çıkış kriteri).

## 2026-09-14 — Veri kaynağı kararı: Zenodo dondurulmuş v783, Codex canlı değil

**Bağlam:** Kullanıcı Codex'in (`codex.flywire.ai`) indirme sayfasını
paylaştı. Sayfa açıkça "canlı Codex veritabanıyla senkronize, Ekim 2024
statik snapshot'tan farklı olabilir" diyor — protokol metni ise "v783"
sabit sürümünü istiyor. Bu, sonucun yorumunu ve tekrarlanabilirliğini
etkileyen bir tasarım kararı olduğu için kullanıcıya soruldu
(PROTOCOL.md §7.3).

**Karar:** Kullanıcı Zenodo'daki dondurulmuş v783 arşivini seçti
(DOI 10.5281/zenodo.10676866, Dorkenwald et al. 2024). Codex canlı export
kullanılmayacak.

**Sonuç:** Gerçek dosya isimleri ve şeması, protokol metnindeki
`neurons.csv.gz` / `classification.csv.gz` / `connections_princeton.csv.gz`
/ `consolidated_cell_types.csv.gz` varsayımlarıyla **uyuşmuyor**. Gerçek
şema: `proofread_connections_783.feather` (pre_pt_root_id, post_pt_root_id,
neuropil, syn_count, {gaba,ach,glut,oct,ser,da}_avg) ve
`proofread_root_ids_783.npy`. `src/flyopt/data/loader.py` ve `validate.py`
bu gerçek şemaya göre yeniden yazıldı (rutin implementasyon düzeltmesi,
hipotezi etkilemiyor — sadece dosya okuma katmanı). E/I işareti artık
nöron başına, syn_count-ağırlıklı baskın nörotransmitter üzerinden
belirleniyor (bkz. loader.py docstring, NT_TO_SIGN karar gerekçesi).

Sınıflandırma/hücre tipi verisi (Faz 1 encoding için, şimdi değil) ayrı bir
kaynaktan: GitHub `flyconnectome/flywire_annotations` (Schlegel et al. 2024),
`Supplemental_file1_neuron_annotations.tsv`. Bkz. `data/raw/README.md`.

**Bloklayıcı (değişmedi):** Veri henüz indirilmedi.

## 2026-09-14 — Gerçek v783 verisi indirildi, ilk ölçüm ve null model doğrulaması

**Bağlam:** Kullanıcı `proofread_connections_783.feather` ve
`proofread_root_ids_783.npy` dosyalarını `data/raw/`'a koydu.
`python -m flyopt.data.validate` çalıştırıldı.

**Ölçülen ağ istatistikleri** (kopyalanmadı, bu dosyalardan ölçüldü):
- n_neurons: 139.255
- n_edges (agregre, neuropil'ler toplanmış): 15.091.983
- density: 0.000778
- frac_excitatory: 0.676, frac_inhibitory: 0.315
- 1250 nöronun hiç outgoing bağlantısı yok (sign=0, tahmin edilmedi)
- out/in-degree ortalama: 108.4, max out: 9783, max in: 10356

Bu sayılar protokol metnindeki referans değerlerle (~139k nöron, ~15M
bağlantı) tutarlı — ama bu sefer gerçek dosyadan ölçüldü, kopyalanmadı.

**Beklenti:** `SparseRecurrentSubstrate`'in gerçek grafta (139k nöron,
15M kenar) çökmeden, makul sürede (protokolün öngördüğü "adayı başına
yüzlerce milisaniye" aralığında) çalışması.

**Sonuç:** Doğrulandı — engine kurulumu ~1.1s, adım başına ~0.3s
(CPU, torch sparse). Protokolün Faz 2/3 maliyet uyarısıyla tam örtüşüyor.

**Sonuç sürpriz çıktı, kodu şüphelendim (protokol §7.2):** `er_null` ve
`degree_preserving_rewire`'ın Python-loop implementasyonları gerçek
ölçekte (15M kenar) pratik değildi — vektörize edildi. İlk vektörize
sürüm gerçek veride test edilince **derece dağılımını tam korumadığı**
ortaya çıktı (in-degree sırası 34/150 küçük test grafında, gerçek grafta
her yerde uyuşmuyordu) — sessizce yanlış sonuç üretiyordu, testler küçük
ölçekte tesadüfen geçmişti. Kök neden: "uygula sonra düzelt" (tentative
apply + revert) yaklaşımı, geri alma adımının permütasyon invariantını
bozması (bir çiftin sadece bir tarafını geri almak, ya da mutlak orijinal
değere resetlemek — ikisi de multiset'i bozuyor). Düzeltme: "önce doğrula,
sonra uygula" — bir çift asla uygulanmadan önce hem self-loop hem
çakışma (hem kabul edilmiş diğer çiftlerle hem reddedilmiş çiftlerin
değişmeyen orijinal kenarlarıyla) kontrol ediliyor, sadece temiz olanlar
uygulanıyor. Gerçek veride doğrulandı: out/in-degree tam eşleşme, 0
self-loop, nnz tam korunuyor, kenarların %99.4'ü değişmiş (iyi karıştırma).
Bu, protokolün "sürpriz sonuçta önce koddan şüphelen" kuralının işe
yaradığı somut bir örnek — H1/H0 testine geçmeden önce yakalandı.

**Zaman ölçümü (gerçek 15M kenarlık grafta):**
- `er_null`: ~9.5s
- `degree_preserving_rewire` (n_sweeps=8): ~118s
- `weight_shuffle`: ~0.5s
- `sign_flip`: ~0.2s

**Faz 0 çıkış kriteri durumu:** Gerçek connectome + tüm null modeller aynı
arayüzden, aynı uyarana karşı çalışıyor, tekrarlanabilir. Kriter
karşılandı. Faz 1'e geçilebilir (kullanıcı onayı beklenir).

## 2026-09-14 — Faz 1 kapı deneyi: koşum öncesi beklenti (PROTOCOL.md §7.1)

**Ön tescil:** `preregistration/faz_1.md`. Tasarım: (1+1) elitist
hill-climber + Fly-Proposer (encode=rastgele 10 nöron enjeksiyonu,
decode=rastgele 50 nöron alt kümesi readout — sınıflandırma verisi
eksikliği nedeniyle protokolün 4 encode/decode kombinasyonundan sadece
biri, bkz. faz_1.md "bloklayıcı" bölümü). 10-D Rastrigin, 10k evaluation,
T=15 adım/evaluation, 30 paylaşılan seed, 4 substrate (flywire, er_null,
degree_preserving_null, weight_shuffle_null). Ana gate: flywire vs
degree_preserving_null, Wilcoxon signed-rank, p<0.05 ve |Cliff's δ|>0.33.

**Beklenti (koşum başlamadan önce yazıldı, sonuç görüldükten sonra
değiştirilmeyecek):** PROTOCOL.md §10'un kendi risk değerlendirmesiyle
uyumlu olarak, en olası sonucun **gate'in geçilememesi** (H0 reddedilemez)
olduğunu düşünüyorum — sinek beyninin koku/görme/lokomotor devreleri için
evrimleştiğine dair bilinen hiçbir mekanizma, Rastrigin gibi soyut bir
optimizasyon görevine aktarılmasını gerektirmiyor. Reservoir computing
literatürü zaten rastgele seyrek rekürren ağların benzer bir "proposal
operator" rolü görebildiğini gösteriyor (literature/map.md, bölüm 6) — bu
da null modellerin connectome'u yakalaması için teorik bir zemin. Gate
geçilirse bunu sürpriz olarak işaretleyip önce kod/sızıntı kontrolü
yapacağım (§7.2).

**Zaman tahmini:** ~2-4 saat (paralel, 20 worker, gerçek 15M-kenarlı
grafta). Kullanıcı onayladı (2026-09-14, bkz. faz_1.md "Bütçe kararı").

**Sonraki giriş:** koşum tamamlanınca sonuç buraya eklenecek (değiştirilen
değil, YENİ bir giriş olarak).

## 2026-09-14 — Maliyet aşımı: paralellik varsayımım yanlıştı

**Ne oldu:** Onaylanan bütçeyle (10k eval, T=15, ~2-4 saat tahmini) gerçek
koşum başlatıldı. 33+ dakika sonra hiçbir tuning görevi tamamlanmamıştı.
Süreçler CPU kullanıyordu (takılma değil), ama beklenenden çok yavaştı.

**Teşhis adımları:**
1. Worker başına `torch.set_num_threads(1)` seçimi yanlış çıktı: CSR
   sparse matmul thread sayısıyla güçlü ölçekleniyor (izole: 1 thread
   ~37ms/adım, 4 thread ~8ms/adım, 24 thread ~6ms/adım-ama-tek-başına).
   Worker'ları tek thread'e sabitlemek, paralellikten kazanılandan daha
   fazlasını kaybettirdi.
2. Düzeltip 12 worker × 2 thread ile yeniden denendi → OOM (bellek
   yetersizliği) ile süreç öldürüldü. Her worker kendi graf kopyasını
   (~1-2GB, degree_preserving_null için tepe ~1.9GB) bellekte tutuyor;
   12 worker × ~1.5GB ortalama ≈ 18GB+, sistemdeki diğer kullanımla
   birlikte 32GB'ı zorladı.
3. 6 worker × 4 thread'e düşürüldü (bellek güvenli, ~8GB) — ama yine
   16+ dakikada tek görev tamamlanmadı.
4. İzole tek görev ölçümü (paralellik yok): 300 eval, flywire, 4 thread
   → **37.7s, tam beklenen hız.** Sorunun paralellik seviyesinde olduğu
   doğrulandı, tek-görev hızında değil.
5. Kontrollü eşzamanlılık testi: 2 paralel görev → 54.7s toplam (agregat
   ~27.3s/görev-eşdeğeri, izoleden %38 daha iyi). 6 paralel görev →
   178.4s toplam (agregat ~29.7s/görev-eşdeğeri — **2'liden daha kötü**).

**Kök neden:** Bu iş yükü (15M-nnz seyrek matris çarpımı) bellek bant
genişliği sınırlı, çekirdek sayısı sınırlı değil. Makine 24 çekirdek/32
thread sunuyor ama bu iş yükü için paralellikten faydalanma sınırı ~2
eşzamanlı süreç civarında — ötesi agregat verimi düşürüyor (muhtemelen
paylaşılan bellek bant genişliği doygunluğu, önbellek çekişmesi,
P-core/E-core heterojenliği). Bu, i9-13900HX gibi mobil/hibrit
çekirdekli bir CPU'ya özgü bir donanım sınırı — genel bir "connectome
büyük, yavaş" bulgusu değil.

**Sonuç:** Tam tasarımın (10k eval) gerçekçi maliyeti ~30-35 saat (2-way
paralel), onaylanan ~2-4 saatin ~10-15 katı. PROTOCOL.md §7.3 tetiklendi,
kullanıcıya durum ve seçenekler sunuldu. **Kullanıcı evaluation bütçesini
küçültmeyi onayladı** (main=1500, sphere=200, tuning=100; seed/substrate/
istatistik tasarımı değişmedi). Bkz. `preregistration/faz_1.md` "Bütçe
kararı" bölümü — orijinal ön tescil koşum başlamadan (hiçbir sonuç
gözlenmeden) düzeltildi, bu p-hacking değil, maliyet tahmininin
düzeltilmesi.

**Ders (gelecekteki fazlar için):** Herhangi bir yeni paralel koşum
tasarımından önce, tam ölçekte KÜÇÜK bir eşzamanlılık taraması (1, 2, 4, 6
worker) yapılıp agregat verim ölçülmeli — çekirdek sayısına göre
`max_workers` varsayılan seçmek bu donanımda yanlış sonuç veriyor.

## 2026-09-14 — FAZ 1 KAPI DENEYİ SONUCU (KESİN)

**Tasarım (preregistration/faz_1.md, iki kez düzeltilmiş bütçeyle):**
10-D Rastrigin, main_budget=1500, T=15 adım, 30 paylaşılan seed, encode=
rastgele 10 nöron enjeksiyonu, decode=rastgele 50 nöron alt kümesi
readout (sınıflandırma verisi eksikliği nedeniyle protokolün 4
kombinasyonundan sadece biri — bkz. faz_1.md "bloklayıcı" bölümü).

**Sonuç — flywire vs degree_preserving_null, 30/30 eşleştirilmiş seed:**

| | flywire | degree_preserving_null |
|---|---|---|
| ortalama | 159.55 | 158.02 |
| medyan | 153.56 | 157.27 |
| std | 30.13 | 31.75 |

Wilcoxon signed-rank: p = **0.6309**
Cliff's delta: **0.001** (pratikte sıfır — dağılımlar neredeyse tam
örtüşüyor)

**GATE GEÇMEDİ** (p<0.05 ve |δ|>0.33 eşiği karşılanmadı — p, eşiğin
12 katı üzerinde; etki büyüklüğü ölçülemeyecek kadar küçük).

**Yorum:** Bu H1'in "belki biraz daha veriyle" reddedilemeyeceği bir
sonuç değil — p=0.63 ve δ≈0 son derece net bir null sonuç. Gerçek
FlyWire connectome'unun bu görev sınıfında (Rastrigin, bu encode/decode
tasarımıyla) degree-preserving null'dan **hiçbir ayırt edilebilir
avantajı yok.** Bu, protokolün kendi ön beklentisiyle (§10 "en olası
sonuç H0'ın reddedilememesi") ve koşum öncesi yazılan beklentiyle
(bu dosyada yukarıda, aynı gün) birebir örtüşüyor. Sonuç sürpriz değil —
kodu şüphelenmek için bir sebep yok (§7.2 uygulanmadı, çünkü beklenen
yönde çıktı).

**Bilinen sınırlamalar (gizlenmiyor):**
1. Encode/decode kombinasyonu 4'te 1 — sınıflandırma verisi
   (`Supplemental_file1_neuron_annotations.tsv`) gelmeden "anatomik
   yapı" (gerçek duyusal/motor nöron sınıfları) hiç test edilmedi.
   Bu null sonuç, "rastgele nöronlara enjeksiyon/okuma" tasarımı için
   geçerli — anatomik olarak yapılandırılmış encode/decode farklı
   çıkabilir mi diye Faz 1b ayrı ön tescille değerlendirilmeli.
2. Evaluation bütçesi 1500 (protokolün istediği 10k değil, maliyet
   aşımı nedeniyle küçültüldü). Daha uzun aramanın farklı bir sonuca
   yol açması mekanik olarak mümkün ama olası değil — hem flywire hem
   null aynı kısıtlı bütçeyle koşuldu (adil karşılaştırma korundu),
   ve δ≈0 zaten "daha fazla veri toplasak p küçülür mü" sorusunun
   ötesinde bir bulgu (etki büyüklüğü örneklem büyüklüğüyle değişmez).

**PROTOCOL.md Faz 1 "DURMA KAPISI" ne diyor:** Gate geçilmezse proje
durmaz ama **dönüşür**: hat "biyolojik connectome avantajı" değil,
"seyrek rekürren reservoir'lar proposal operatörü olarak" olur — farklı
bir makale, farklı literatür konumlandırması. **Bu dönüşüm kararı
kullanıcıya aittir, agent kendi başına karar vermez.**

**Sonraki adım (kullanıcı kararı bekleniyor):**
- (a) Faz 1b: sınıflandırma verisiyle anatomik encode/decode'u test et
  (mevcut null sonucu anatomik yapının önemsiz olduğunu düşündürüyor
  ama kanıtlamıyor — encode/decode rastgeleydi, connectome'un
  topolojisi değil).
- (b) Gate'i nihai kabul et, "connectome avantajı yok" çerçevesine geç
  (PROTOCOL.md §9 negatif senaryo — bu da yayınlanabilir bir sonuç).
- (c) Faz 2'ye (mimari taraması) geçmeden dur, kapsam riskini
  (PROTOCOL.md §10 — kullanıcının aktif manuscript/tez yükü) yeniden
  değerlendir.

Ana koşum hâlâ arka planda tamamlanıyor (weight_shuffle_null ve
er_null'un tam istatistiği, Holm-Bonferroni düzeltmeli ikincil
karşılaştırmalar için) — bu KESİN gate kararını değiştirmez, sadece
tamamlayıcı kanıt ekler.

## 2026-09-15 — Faz 1 TAMAMEN bitti: ikincil karşılaştırmalar sürpriz bir örüntü gösteriyor

**Tüm 120 görev tamamlandı** (gece, şarj takılana kadar bir süre batarya
modu nedeniyle ~2x yavaşlama yaşandı — kod/deney bütünlüğünü etkilemedi,
sadece süreyi). Dört substrate'in tam özeti (30 seed, Rastrigin best-fx,
düşük=daha iyi):

| substrate | ortalama | medyan | std |
|---|---|---|---|
| er_null | 122.28 | 123.75 | 27.60 |
| weight_shuffle_null | 147.42 | 143.93 | 24.82 |
| flywire | 159.55 | 153.56 | 30.13 |
| degree_preserving_null | 158.02 | 157.27 | 31.75 |

**Ana gate (değişmedi):** flywire vs degree_preserving_null — p=0.6309,
Cliff's δ=0.001. Gate geçmedi (bkz. yukarıdaki KESİN sonuç girişi).

**İkincil karşılaştırmalar (Holm-Bonferroni düzeltmeli, YENİ):**
- flywire vs er_null: p=4.33×10⁻⁶, δ=**0.608** (büyük etki), Holm-reject=True
- flywire vs weight_shuffle_null: p=0.028, δ=0.241 (küçük-orta etki),
  Holm-reject=True (0.05'e göre düzeltmeden sonra da anlamlı)

**δ'nın işareti önemli:** Cliff's delta pozitif = flywire değerleri daha
YÜKSEK (Rastrigin minimizasyonunda YÜKSEK = DAHA KÖTÜ). Yani **flywire,
hem er_null hem weight_shuffle_null'dan istatistiksel olarak anlamlı ve
orta-büyük etki büyüklüğüyle DAHA KÖTÜ performans gösteriyor** — connectome
sadece "null'dan farksız" değil, bazı null'lardan **belirgin şekilde daha
kötü.**

**Sonuç sürpriz mi, kod şüphesi gerekiyor mu (§7.2)?** Bu, "connectome
sessizce sızıntıyla avantajlı çıktı" yönünde bir sürpriz DEĞİL — tam
tersi yönde (connectome daha kötü). Sızıntı/bug'lar genelde test edilen
şeyi yapay olarak İYİ gösterir, kötü göstermez; bu yüzden bunun bir kod
hatası olma ihtimali düşük, ama yine de mekanizmayı anlamaya değer.

**Olası mekanizma (doğrulanmadı, hipotez):** flywire ve
degree_preserving_null AYNI derece dağılımını paylaşıyor (çok az sayıda
"hub" nöron — Faz 0'da ölçülen max out-degree 9783, max in-degree 10356 —
ve çoğunluk düşük dereceli nöron). er_null ve weight_shuffle_null bu
ağır-kuyruklu derece yapısını YOK EDİYOR (er_null: Poisson-benzeri
homojen derece; weight_shuffle_null: topoloji aynı ama... **dikkat:
weight_shuffle_null'un derece dağılımı da flywire ile birebir aynı**
(sadece ağırlıklar karıştırılmış, hangi nöronun kaç bağlantısı olduğu
değişmiyor) — yine de flywire'dan anlamlı ölçüde daha iyi performans
gösteriyor. Bu, farkın SADECE derece dağılımından kaynaklanamayacağını,
**ağırlık-topoloji eşleşmesinin** (hangi spesifik ağırlık hangi spesifik
kenarda) da bir rol oynadığını düşündürüyor — gerçek connectome'daki
ağırlık yerleşimi, bu soyut görev için rastgele bir yerleşimden daha az
elverişli. Bu sadece bir hipotez, doğrulanmadı; nedensel mekanizma ayrı
bir analiz gerektirir (ör. hub nöronlarının okuma/yazma popülasyonlarına
göre konumu, ağırlık büyüklüğü dağılımının okuma sinyali varyansına etkisi).

**Genel yorum:** Faz 1'in birincil sorusu ("connectome, en yakın/en sıkı
null modelden — degree_preserving — ayırt edilebilir mi") NET bir HAYIR
ile cevaplandı. İkincil bulgu, bunun ötesinde connectome'un daha basit
null modellerden bile sistematik olarak daha kötü performans
gösterebileceğini akla getiriyor — bu, H1'i daha da zayıflatan, negatif
sonucu güçlendiren bir bulgu (connectome'un "en azından zararsız" bile
olmayabileceği yönünde). `results/faz1_summary_draft.md` güncellendi.

## 2026-09-15 — Faz 1b koşum öncesi beklenti (PROTOCOL.md §7.1)

**Ön tescil:** `preregistration/faz_1b.md`. Tasarım Faz 1 ile birebir
aynı, tek fark: encode=afferent (gerçek duyusal, n=19.261) havuzundan
10 nöron, decode=efferent (gerçek motor, n=1.489) havuzundan 50 nöron —
protokolün encode(a)/decode(a) alternatiflerinin, sınıflandırma verisi
gelmesiyle artık tam (rastgele-değil, gerçek anatomik) hali.

**Beklenti (koşum başlamadan önce yazıldı):** Faz 1'in ikincil bulgusu
(connectome'un ağır-kuyruklu derece dağılımı + ağırlık yerleşiminin bu
soyut görevde dezavantajlı olabileceği hipotezi) ve reservoir computing
literatürünün teorik beklentisi (literature/map.md §6) göz önüne
alındığında, **Faz 1b'nin de gate'i geçmesini beklemiyorum** — anatomik
olarak "doğru" nöronları kullanmak, connectome'un evrimleştiği görev
sınıfı dışındaki (Rastrigin) bir görevde temel dinamiği değiştirmez.
Ama `annel0/flybrain` bulgusu (literature/map.md §7 — connectome'un
kendi doğal devrelerinde gerçekten fark yarattığı) küçük bir olasılık
payı bırakıyor: eğer afferent→efferent yolu, connectome'un genel
mimarisinde ayrıcalıklı bir şekilde kablolanmışsa (örn. daha kısa yollar,
daha güçlü bağlantılar), rastgele encode/decode'un gizlediği bir sinyal
ortaya çıkabilir. Bu ihtimali reddetmiyorum ama düşük olasılıklı
görüyorum.

**Sonraki giriş:** koşum tamamlanınca sonuç buraya eklenecek.

## 2026-09-15 — FAZ 1b SONUCU (KESİN, sınırda): gate yine geçmedi ama çok yaklaştı

**Not (bağlam):** Bu koşum, 04:15'te bir kesinti (muhtemelen Windows
güncellemesi kaynaklı bilgisayar yeniden başlatması) nedeniyle 54/120'de
yarıda kaldı, kullanıcı işe gittikten sonra (~08:15) sıfırdan yeniden
başlatıldı ve sorunsuz tamamlandı.

**Sağlık kontrolü (PROTOCOL.md §7.2, sonuç Faz 1'den farklı yönde
çıkabileceği için koşuldu):** `afferent_indices.npy` (19.261, tekrarsız)
ve `efferent_indices.npy` (1.489, tekrarsız) havuzları arasında kesişim
yok; `use_real_pools: true` doğrulandı; flywire ve degree_preserving_null
için kullanılan seed kümeleri birebir aynı (30/30, paired tasarım
bozulmamış); main_budget=1500 (Faz 1 ile aynı). **Sorun bulunamadı.**

**Sonuç — flywire vs degree_preserving_null, 30/30 eşleştirilmiş seed:**

| | flywire | er_null | degree_preserving_null |
|---|---|---|---|
| ortalama | 148.86 | 129.59 | 166.93 |
| medyan | 149.14 | 125.57 | 162.18 |

Wilcoxon signed-rank: p = **0.003162**
Cliff's delta: **-0.312**

**GATE YİNE GEÇMEDİ** — ama çok sınırda. p-değeri eşiğin (0.05) çok
altında ve güçlü, ama etki büyüklüğü (|δ|=0.312) pre-tescilli eşiğin
(|δ|>0.33) **hemen altında** kalıyor. Ön tescil kuralı (p<0.05 **VE**
|δ|>0.33, ikisi birden) harfiyen uygulanıyor — "neredeyse geçti" diye
yuvarlamıyoruz. Bu sınır, deney başlamadan önce belirlendi, sonuç
görüldükten sonra değiştirilmedi.

**Yön, Faz 1'den farklı:** Faz 1'de flywire ile degree_preserving_null
istatistiksel olarak ayırt edilemezdi (p=0.63, δ≈0). Faz 1b'de flywire,
degree_preserving_null'dan **anlamlı ölçüde iyi** çıkıyor (δ negatif =
flywire'ın Rastrigin değerleri daha düşük = daha iyi). Ama flywire, HÂLÂ
er_null'dan (saf rastgele graf) anlamlı ölçüde **kötü** çıkıyor
(p=0.000479, δ=0.392) — bu örüntü Faz 1'le aynı yönde ve tutarlı.

**Bu ne anlama geliyor — dikkatli yorum:** Faz1 vs Faz1b'nin flywire
sonuçlarını doğrudan karşılaştırdım (Mann-Whitney): p=0.24, anlamlı fark
yok — yani **anatomik encode/decode, flywire'ın kendi mutlak
performansını dramatik şekilde değiştirmedi** (148.86 vs 159.55,
istatistiksel olarak ayırt edilemez). Asıl fark, degree_preserving_null'un
bu koşumda daha kötü çıkmasından geliyor (166.93 vs Faz1'deki 158.02,
p=0.42 — bu da anlamlı değil ama yön bu yönde). Yani **flywire ile
degree_preserving_null arasındaki farkın Faz1b'de ortaya çıkması, flywire
"iyileştiği" için değil, bu spesifik degree-preserving null örnekleminin
(her koşumda yeniden üretiliyor, farklı rastgele rewiring) bu sefer
biraz daha kötü çıkmasından kaynaklanıyor olabilir** — gürültü payı
yüksek bir açıklama, ama veriyle tutarlı.

**Genel değerlendirme:** İki fazın BİRLEŞİK tablosu şöyle:
- flywire, degree-preserving null'dan istatistiksel olarak hiç net bir
  şekilde ayrılamadı (Faz1: hayır; Faz1b: p anlamlı ama etki büyüklüğü
  eşiğin altında — "sınırda belirsiz" diyebiliriz, "kanıtlanmış avantaj"
  diyemeyiz).
- flywire, HER İKİ fazda da saf rastgele graftan (er_null) anlamlı ölçüde
  **kötü** çıktı — bu tutarlı, tekrarlanan bir bulgu.
- Sonuç: **H1 (connectome'un spesifik topolojisinin bir proposal-operatör
  avantajı sağladığı hipotezi) bu görev sınıfı ve bu iki encode/decode
  tasarımı için desteklenmiyor.** En iyi ihtimalle "belirsiz/sınırda",
  en kötü ihtimalle (er_null karşılaştırması) "dezavantajlı".

`results/faz1_summary.md` artık taslak değil, tamamlandı.

## 2026-09-15 — Kod hatası bulundu ve düzeltildi: `cliffs_delta` numpy tipini sızdırıyordu

Faz 1b'nin koşum betiği (`faz1_runner.py main()`), tüm 120 görev
tamamlandıktan SONRA, son analiz adımında (`json.dumps(analysis)`)
`TypeError: Object of type bool is not JSON serializable` ile çöktü
(exit code 1). **Hiçbir ham veri kaybolmadı** — `results/raw/
faz1b_progress.jsonl` içindeki 120 görevin tamamı sağlamdı, yukarıdaki
KESİN sonuç bu dosyadan doğrudan (buggy analyze() fonksiyonunu
atlayarak) hesaplandı ve doğrulandı.

**Kök neden:** `experiment/stats.py::cliffs_delta`, `(gt - lt) /
(len(a)*len(b))` ifadesini `float()`'a hiç cast etmiyordu — `gt`/`lt`
numpy `.sum()`'dan geldiği için sonuç `numpy.float64` oluyordu.
`passes_falsification_gate`'te `p_value < ALPHA and abs(effect_size) >
THRESHOLD` ifadesinde, `p_value < ALPHA` **False** ise Python `and`
kısa-devre yapıp ikinci terimi hiç değerlendirmiyordu (native bool
dönüyordu) — Faz 1'in p=0.63 olması bu yüzden hiç sorun çıkarmamıştı.
Ama Faz 1b'de p=0.003 (True) olunca kısa-devre olmadı, `abs(numpy.float64)
> float` ifadesi `numpy.bool_` üretti — bu tip standart `json` modülüyle
serileştirilemiyor.

**Düzeltme:** `cliffs_delta` artık `float(...)` ile native tipe cast
ediyor. İki regresyon testi eklendi (`test_cliffs_delta_returns_native_float`,
`test_passes_falsification_gate_result_is_json_serializable` — ikincisi
bilinçli olarak p<0.05 dalını tetikliyor). Eksik kalan
`results/negative/20260915T125436_...json` dosyası, ham log'dan
(deney tekrar koşulmadan) yeniden üretildi — bu da eksik kalan **ikincil
karşılaştırmayı** (flywire vs weight_shuffle_null) tamamladı:

**flywire vs weight_shuffle_null (Faz 1b):** p=0.3812, δ=-0.014 — anlamlı
fark yok. Bu, Faz 1'deki aynı karşılaştırmadan (p=0.028, δ=0.241, flywire
anlamlı kötüydü) **farklı** — iki fazın weight_shuffle_null sonuçları
arasındaki tutarsızlık, gürültü payının bu karşılaştırmada yüksek
olduğunu gösteriyor (weight_shuffle_null da her koşumda yeniden, farklı
rastgele karıştırmayla üretiliyor).

**Güncellenmiş tam ikincil tablo (Faz 1b):**
- flywire vs er_null: p=0.00048, δ=0.392 (flywire anlamlı kötü,
  Holm-reject=True) — Faz 1 ile tutarlı
- flywire vs weight_shuffle_null: p=0.381, δ=-0.014 (fark yok,
  Holm-reject=False) — Faz 1'den farklı

## 2026-09-15 — Faz 1c koşum öncesi beklenti (PROTOCOL.md §7.1)

**Ön tescil:** `preregistration/faz_1c.md`. Faz 1b ile aynı tasarım,
tek fark: `FlyProposer.tell()` artık ödül-modülasyonlu Hebbian kuralıyla
okuma matrisini güncelliyor (`readout_lr>0` durumunda). Hiperparametre
taramasına `readout_lr ∈ {0.0, 0.05, 0.2}` eklendi (9→27 kombinasyon).
Kullanıcının kendi önerisi (2026-09-15 sohbeti).

**Beklenti (koşum başlamadan önce yazıldı):** Nötr — preregistration/
faz_1c.md'de yazdığım gibi, bu sefer net bir ön-tahminim yok. Lehte
argüman: reservoir computing standart pratiği eğitilmiş okumanın işe
yaradığını gösteriyor. Aleyhte argüman: flywire'ın er_null'dan tutarlı
şekilde kötü çıkması, sorunun "okumada" değil "dinamikte" olabileceğini
düşündürüyor — okuma eğitimi bunu telafi edemeyebilir. Özellikle
flywire-vs-er_null yönünün (tutarlı, iki fazda tekrarlanan tek bulgu)
eğitimle değişip değişmediğini izleyeceğim.

**Sonraki giriş:** koşum tamamlanınca sonuç buraya eklenecek.

## 2026-09-15 — FAZ 1c SONUCU: Gate teknik olarak GEÇTİ ama bu bir tarama artefaktı, H1'i desteklemiyor

**Ham sonuç — flywire vs degree_preserving_null, 30/30 seed:**
Wilcoxon p=0.000563, Cliff's δ=**-0.339**. Her iki eşik de (p<0.05 VE
|δ|>0.33) karşılanıyor — **GATE GEÇTİ**, ilk kez.

**Ama hemen şüphelendim (PROTOCOL.md §7.2) ve araştırdım.** Sebep:
hiperparametre taramasında flywire `readout_lr=0.0` (plastisite yok)
seçerken, **üç null modelin üçü de `readout_lr=0.2` seçti.** Bu kendi
başına adil bir sonuç olabilirdi (her substrate eşit tarama bütçesiyle
kendi en iyisini seçiyor, PROTOCOL.md §4.1) — AMA doğrudan kontrol
ettim: her substrate'in Faz 1b'deki (plastisite hiç yok) performansıyla
Faz 1c'deki (plastisite seçenekli tarama sonrası) performansını
karşılaştırdım:

| substrate | Faz 1b ortalama | Faz 1c ortalama | fark |
|---|---|---|---|
| flywire | 148.86 | 148.86 | **yok** (p=1.0 — zaten aynı hesaplama, lr=0.0 seçilmiş) |
| er_null | 129.59 | 161.78 | **anlamlı ve büyük şekilde KÖTÜLEŞMİŞ** (p=0.0002, δ=-0.563) |
| degree_preserving_null | 166.93 | 165.70 | yok (p=0.906) |

**Sonuç net:** flywire'ın kendi performansı Faz 1b'den Faz 1c'ye hiç
değişmedi (zaten plastisite kullanmadı). Ama **er_null'un performansı
plastisite açıldığında ciddi şekilde kötüleşti** — 100 evaluation/3
seed'lik küçük tarama bütçesi, tam bütçede (1500 eval/30 seed) zararlı
çıkacak bir `readout_lr` değerini "en iyi" olarak yanlış seçmiş.
Yani **Faz 1c'nin "gate geçti" sonucu, connectome'un bir şey
başarmasından değil, null modellerin kötü bir hiperparametre seçimiyle
sakatlanmasından kaynaklanıyor.**

**İkincil karşılaştırma (flywire vs er_null) da bunu doğruluyor:**
p=0.0125, δ=-0.210 — flywire artık er_null'dan "iyi" görünüyor (Faz 1 ve
Faz 1b'de tam tersiydi), ama bu tersine dönüş de aynı artefaktan
kaynaklanıyor (er_null'un kendi performansı kötüleşti, flywire aynı kaldı).

**HÜKÜM: Bu gate geçişi geçersiz sayılmalı, H1 lehine kanıt değil.**
Ön tescil kuralının harfi karşılandı ama ruhu karşılanmadı — adil
karşılaştırma, "her substrate kendi en iyisini bulsun" derken, tarama
sürecinin KENDİSİNİN güvenilir olduğunu varsayıyordu. Küçük tarama
bütçesine yeni bir boyut (plastisite) eklemek, en azından bazı
substrate'ler için bu varsayımı bozdu.

**Ders (metodolojik, gelecekteki çalışmalar için):** Yeni bir
hiperparametre boyutu eklerken (özellikle etkisi gürültülü/substrate'e
özgü olabilecek bir mekanizma, plastisite gibi), tarama bütçesinin de
orantılı büyütülmesi gerekir — 9 kombinasyondan 27'ye çıkarken tarama
bütçesini aynı (100 eval/3 seed) bırakmak yetersiz kaldı. Bu, Faz
1c'nin ana bulgusu değil ama önemli bir yan bulgu.

**preregistration/faz_1c.md'nin durma kriterine göre:** "Gate yine
geçmezse..." senaryosu teknik olarak tetiklenmedi (gate geçti) ama
yorumum onu geçersiz kılıyor — pratik sonuç, "gate geçmedi" senaryosuyla
aynı: **Fly-Proposer + ucuz okuma eğitimi fikri de bu görev sınıfı için
tükenmiş sayılmalı.** Üç fazın (1, 1b, 1c) hiçbiri H1'i güvenilir şekilde
desteklemiyor.

**Güncelleme (aynı gün, kullanıcı isteğiyle koşum durduruldu, 114/120):**
`weight_shuffle_null` 24/30 seed'te durduruldu (kullanıcı talebi — geri
kalan sadece ikincil istatistik içindi, ana hüküm zaten kesinleşmişti).
Eldeki 24/30 ile (EKSİK, pre-tescilli 30 değil) kısmi analiz:

flywire vs weight_shuffle_null (n=24): p<0.0001, δ=-0.524 — flywire
"iyi" görünüyor, ama artık tanıdık nedenle: weight_shuffle_null'un kendi
Faz 1b (150.89, plastisite yok) → Faz 1c (171.89, n=24, plastisite
seçilmiş) performansı da anlamlı ölçüde **kötüleşmiş** (p=0.0113,
δ=-0.406) — aynı tarama artefaktı. Üç null modelin ikisi (er_null,
weight_shuffle_null) plastisite yüzünden kötüleşti, biri
(degree_preserving_null) değişmedi — hiçbiri iyileşmedi, flywire de hiç
değişmedi. **Hüküm aynı kalıyor: Faz 1c'nin görünürdeki "başarısı"
tamamen null modellerin sakatlanmasından kaynaklanıyor, H1 lehine kanıt
değil.**

## 2026-09-16 — Faz S1 koşum öncesi beklenti (PROTOCOL.md §7.1)

**Ön tescil:** `preregistration/faz_s1.md`. Faz 1/1b/1c'den iki yönde
farklı: (1) **kodlama** artık ham x enjeksiyonu değil, her eksende +/-
yönde örneklenen amaç-fonksiyonu farkları ("ışınlar", malecns'in lidar
kodlamasına benzer), afferent havuzuna enjekte ediliyor; (2) **süreklilik**
— substrate durumu artık her `propose()` çağrısında sıfırlanmıyor, tüm
arama koşusu boyunca korunuyor; (3) **okuma eğitimi** — okuma matrisi
artık rastgele sabit değil, ucuz bir taklit-öğretmene (aynı ışın
örneklerinden türetilen kaba negatif gradyan) karşı tek turluk ridge
regresyonla kalibre edilip sonra dondanuluyor (Faz 1c'nin küçük-bütçeli
çevrimiçi ayarının ürettiği tarama artefaktına kasıtlı olarak girilmiyor).

Motivasyon: aynı gece `fly_demos/malecns` (MaleCNS + F1 araba sürüşü)
ile yapılan gayrı-resmi bir keşifte, aynı ilkeyle (gerçek connectome vs
ER-null, eşit eğitim) çok net bir ters yönlü sonuç gözlendi — gerçek
connectome her metrikte ER-null'dan belirgin üstündü. Faz S1 bu
gözlemin "eğitim" mi yoksa "görev-mimari eşleşmesi" mi etkeninden
kaynaklandığını, aynı soyut Rastrigin/Sphere kıyaslamasında test ediyor.

**Beklenti (koşum başlamadan önce yazıldı):** preregistration/faz_s1.md'de
yazdığım gibi net değilim — malecns gözleminden dolayı hafif pozitif
önsezi var, ama Rastrigin'in 10-D soyut uzayının malecns'in gerçek
uzamsal "arazi"sinden yeterince farklı olabileceğinden ve tek-turluk
kalibrasyonun malecns'in çok-turlu DAgger+ES'inden çok daha zayıf
kalabileceğinden çekiniyorum. Özellikle flywire-vs-er_null yönünün
(üç fazda tutarlı şekilde "flywire kötü") bu sefer tersine dönüp
dönmediğini izleyeceğim.

**Kapsam kararı (veri görülmeden önce, maliyet gerekçesiyle):** Tam
bütçenin (30 tohum) tahmini maliyeti ~16-20 saat ölçüldü (smoke test
sonrası). Kullanıcıyla konuşup 10 tohumluk bir pilot ile başlamaya karar
verdik (main_budget/sphere_budget değişmedi, sadece tohum sayısı) —
Faz 1'in orijinal 10000→1500 bütçe kesintisiyle aynı gerekçe (EXPERIMENTS.md
2026-09-14), sonucu görmeden alınan kaynak kararı.

**Sonraki giriş:** koşum tamamlanınca sonuç buraya eklenecek.

## 2026-09-16 — FAZ S1 SONUCU (pilot, 10/30 seed): Gate yine GEÇMEDİ ama yön ilk kez doğru

**Ham sonuç — flywire vs degree_preserving_null, 10/10 seed (pilot):**
Wilcoxon p=0.0098 (anlamlı, eşik 0.05), Cliff's δ=**-0.16** (eşik
|δ|>0.33 karşılanmadı). **GATE GEÇMEDİ** — istatistiksel anlamlılık var
ama etki büyüklüğü pre-tescilli "anlamlı" eşiğinin altında.

flywire ortalama=117.83, degree_preserving_null ortalama=125.48 (düşük
= iyi, Rastrigin minimizasyonu) — flywire tutarlı şekilde biraz daha iyi.

**İlk kez: flywire vs er_null neredeyse eşit.** p=0.770, δ=-0.08 —
Faz 1/1b/1c'nin üçünde de tutarlı şekilde gözlenen "flywire er_null'dan
anlamlı şekilde kötü" örüntüsü burada tamamen kayboldu (bkz. 2026-09-15
Faz 1 TAMAMEN bitti girdisi). flywire vs weight_shuffle_null de anlamlı
değil (p=0.492, δ=-0.14).

**Yorum:** Bu, üç önceki fazın hiçbirinde görmediğimiz bir örüntü —
gerçek connectome artık hiçbir null modelden istatistiksel olarak kötü
çıkmıyor, ve birincil kıyaslamada (degree_preserving_null) yönü de doğru
ama küçük. Cliff's delta örneklem büyüklüğünden bağımsız bir ölçü
olduğu için, kalan 20 tohumu koşmak p-değerini muhtemelen daha da
küçültür ama δ'yı ~0.16 civarında tutar — yani tam 30-tohum koşumunun
bu **niteliksel** sonucu (gate geçmiyor, eşik altı) değiştirmesi
beklenmiyor. Bu gerekçeyle kullanıcıyla pilotu nihai kabul etmeye karar
verdik, kalan 20 tohum koşulmadı.

**PROTOCOL.md §9 açısından:** Faz S1 de negatif sonuç listesine
eklendi (`results/negative/20260916T130806_62ddbaa3af42f0af_seed0.json`).
Ama malecns gözlemiyle çelişmiyor — ikisi de doğru olabilir: connectome'un
avantajı muhtemelen "eğitim" etkeninden değil, "görev-mimari eşleşmesi"
etkeninden geliyor (malecns'in gerçek uzamsal/sürekli duyu-motor görevi,
Rastrigin'in soyut R^10 uzayından çok farklı), ve ışın-kodlaması +
tek-turluk kalibrasyon bu eşleşmeyi yeterince sağlamadı. flywire'ın artık
hiçbir null modelden kötü çıkmaması (üç fazın tutarlı negatif bulgusunun
tersine dönmesi) yine de kayda değer — "connectome dezavantajlı" hipotezi
zayıflıyor, sadece "connectome bu soyut görevde avantajlı" hipotezi hâlâ
desteklenmiyor.

## 2026-09-17 — Fly-Hybrid PSO keşfi (ÖN TESCİLLİ DEĞİL, gayrı-resmi): tek-tohum çarpıcı sonuç çoklu-tohumda doğrulanmadı

**Bu bir Faz değil, pre-registration yok, falsifikasyon kapısı yok, null
model kıyaslaması yok.** PROTOCOL.md §2.7 Fly-Hybrid fikrinin ("sinek
lokal karar + PSO global koordinasyon") görsel bir merak/deneme
uygulaması — kullanıcının "sürü halinde nasıl davranıyor görelim" isteği
üzerine. `scripts/fly_swarm_pso_demo.py`, `scripts/fly_swarm_pso_multiseed.py`.

**Tasarım:** 12 parçacık ("sinek"), 2-D Rastrigin, gerçek FlyWire
connectome'u (tek substrate, tüm parçacıklar arasında salt-okunur
paylaşılıyor). Klasik PSO güncellemesine (atalet + bilişsel + sosyal
çekim) her parçacığın kendi connectome-nüdge'i ekleniyor. Dört varyant:

- `classic`: connectome yok (kontrol)
- `raw`: FlyProposer, ham-x enjeksiyonu, sabit ağırlık (ilk gördüğümüz,
  "şık" bulunan demo)
- `scene`: FlyProposerScene (Faz S1'in ışın kodlaması + taklit-kalibreli
  okuma), sabit ağırlık
- `scene_anneal`: `scene` + zamanla azalan ağırlık (1.0→0.05, önce keşif
  sonra klasik PSO'nun ince ayarına bırak)

**Tek-tohum sonucu (seed=42) — ÇARPICI:** classic ve raw ikisi de aynı
yerel minimuma sıkıştı (fx=0.9950, birebir aynı — raw'ın connectome
nüdge'i hiçbir şey katmamış). scene hafif daha kötü (1.0076). Ama
scene_anneal gerçek global minimumu buldu (fx=0.0004, ~2500x iyileşme).
Bu, ilk bakışta "ışın kodlama + zamanlama" kombinasyonunun işe yaradığına
dair güçlü bir sinyal gibi görünüyordu.

**8-tohumlu sağlamlık kontrolü — sinyal DOĞRULANMADI:**

| varyant | ortalama | medyan | en kötü | gerçek minimumu buldu (fx<0.1) |
|---|---|---|---|---|
| classic | 0.249 | 0.000 | 0.995 | 6/8 |
| raw | 0.498 | 0.498 | 0.995 | 4/8 |
| scene | 0.641 | 0.516 | 1.991 | 4/8 |
| scene_anneal | 0.376 | 0.003 | 1.991 | 6/8 |

Klasik PSO (connectome yok) 8 tohumun 6'sında zaten gerçek minimumu
buluyor — scene_anneal ile aynı başarı oranı, üstelik classic'in en kötü
sonucu (0.995) scene_anneal'ınkinden (1.991) daha iyi. raw ve scene ise
plain PSO'dan **daha kötü** (4/8) — connectome nüdge'i bazı tohumlarda
zarar veriyor.

**HÜKÜM: seed=42'deki çarpıcı sonuç genele yayılmıyor.** O tohum, classic
PSO'nun kötü bir yerel minimuma sıkıştığı ve scene_anneal'ın tesadüfen
kaçabildiği özel bir durumdu — 8 tohuma bakınca connectome'un PSO'ya
tutarlı bir katkısı yok, raw/scene'de zarar bile veriyor. Bu, FlyOpt'un
ana bulgusuyla (soyut optimizasyonda connectome'un güvenilir bir
avantajı yok) tam tutarlı; PSO'ya geçmek sonucu değiştirmedi.

**Metodolojik ders (protokolün kendi ruhuna örnek bir uygulama):**
Kullanıcının önerisiyle önce fly_weight=0 kontrolü isteneceği için raw
varyantın "hiçbir şey katmadığı" hemen görüldü — tek-tohum demoyu
görünce heyecanlanıp genel bir sonuç ilan etmek yerine, kontrol +
çoklu-tohum sağlamlık kontrolü istenmesi yanlış bir pozitif iddiayı
(EXPERIMENTS.md'nin daha önce Faz 1c'de düştüğü tuzağın küçük ölçekli bir
tekrarı) baştan engelledi.

**Bu görevden yeni bir Faz açılmayacak** — sinyal zaten tek-tohumda bile
yalnızca bir kaçıştı, 8 tohumda kayboldu; daha büyük bir pre-registered
PSO fazına yatırım yapmak için gerekçe yok.

## 2026-09-17 — Fly-Swarm-Social: "son bir deneme" de klasik PSO'yu geçemedi (ÖN TESCİLLİ DEĞİL, gayrı-resmi)

Yukarıdaki PSO bulgusunun ardından kullanıcı bir adım daha istedi: sinek
beyninin **eğitilmiş** halini, çok daha zengin bir eğitim rejimiyle
denemek. `src/flyopt/variants/fly_swarm_social.py`,
`scripts/fly_swarm_social_demo.py`. Yine pre-registration/gate/null-model
yok — gayrı-resmi bir keşif.

**Tasarım — Faz S1/PSO'dan iki yönde zenginleştirilmiş:**
1. **Çoklu-problem eğitim:** Faz S1'in tek-rota tek-turluk kalibrasyonu
   yerine, 25 rastgele kaydırılmış Rastrigin örneği × 12 rastgele
   başlangıç noktası (300 örnek) — her birinde gerçek minimum bilindiği
   için (biz kaydırdık) "ideal yön" ucuza hesaplanabiliyor. Çok daha
   zengin/çeşitli bir denetimli eğitim seti.
2. **Sosyal ışınlar:** Arabadaki lidar ışınlarına benzer şekilde, her
   sinek artık diğer sineklerin **açısını, uzaklığını ve fitness'ini**
   ("yerden yükseklik" gibi) panoramik, açısal-kutulu bir sinyal olarak
   algılıyor (`_social_rays`, 8 sektör). Kalibrasyon hedefi ikisini
   harmanlıyor: "gerçek minimuma + iyi komşulara doğru git" — PSO'nun
   sosyal çekim terimini connectome'un okuma matrisine gömüyor.

Aynı 8 tohum, aynı önceki 4 varyanta (classic/raw/scene/scene_anneal) iki
yeni varyant eklendi: `social` (sabit ağırlık), `social_anneal` (azalan
ağırlık).

**Sonuç (n=8 tohum):**

| varyant | ortalama | medyan | gerçek minimumu buldu |
|---|---|---|---|
| classic | 0.249 | 0.000 | **6/8** |
| raw | 0.498 | 0.498 | 4/8 |
| scene | 0.641 | 0.516 | 4/8 |
| scene_anneal | 0.376 | 0.003 | **6/8** |
| social | 0.565 | 0.270 | 4/8 |
| social_anneal | 0.498 | 0.498 | 4/8 |

**HÜKÜM: Zengin çoklu-problem eğitimi ve sosyal ışın algısı, klasik
PSO'yu geçmedi.** social/social_anneal, scene/scene_anneal'dan bile daha
iyi değil — social_anneal beklenenin aksine scene_anneal'dan kötü çıktı
(4/8 vs 6/8). seed=0'ın yakınsama eğrisinde bunun net bir örneği var: bu
sefer scene_anneal tuzağa sıkışırken en sade "raw" varyant en iyi sonucu
verdi — **hangi varyantın kazanacağı tohumdan tohuma tamamen değişiyor,
sistematik bir üstünlük yok.**

**Genel çıkarım (PSO keşfinin tamamı için):** Ne ham enjeksiyon, ne ışın
kodlaması, ne zamanla azalan ağırlık, ne de çok-problem eğitimi + sosyal
farkındalık — connectome'u PSO'ya eklemenin denediğimiz hiçbir versiyonu
klasik PSO üzerinde güvenilir bir avantaj göstermedi. Bu, FlyOpt'un ana
bulgusuyla (Faz 1/1b/1c/S1) tam tutarlı ve onu pekiştiriyor: sorun
"okuma katmanını eğitmedik" ya da "girdi/çıktı kodlaması basitti" değil
görünüyor — soyut optimizasyon görevlerinde connectome'un güvenilir bir
hesaplama avantajı bulunamadı, denediğimiz hiçbir mühendislik
müdahalesiyle (eğitim, kodlama zenginliği, zamanlama, sosyal bilgi).
PSO hattı burada kapatılıyor.

## 2026-09-17 — Kontrol denemesi: TRM-tarzı (connectome'suz) küçük ağ da klasik PSO'yu geçemedi (ÖN TESCİLLİ DEĞİL, gayrı-resmi)

Kullanıcının sorusu: "sinek beyni yerine HRM/TRM mimarisi denesek nasıl
sonuç verir?" — connectome'u tamamen çıkarıp, Hierarchical/Tiny Recursive
Model fikrinden (2025, ARC-AGI/Sudoku gibi bulmaca görevlerinde küçük
parametre sayısıyla güçlü sonuçlar veren, tekrarlayan/gizli-durum
iyileştiren mimari ailesi) esinlenen minik, tamamen gradyan inişiyle
eğitilebilir bir ağ denendi. `src/flyopt/variants/tiny_recursive_proposer.py`,
`scripts/fly_tiny_recursive_demo.py`. Literal bir HRM/TRM yeniden
uygulaması değil — TRM makalesinin kendi bulgusuna uyarak ("less is
more") tek, paylaşılan-ağırlıklı küçük bir MLP çekirdeği, gizli durum
`z` ve cevap `y`'yi 6 adım boyunca tekrar tekrar iyileştiriyor, tüm
rekürsiyon boyunca tam backprop ile eğitiliyor.

**Amaç:** Connectome'un PSO'ya hiçbir versiyonda avantaj katmamasının
(yukarıdaki PSO girdisi) "biyolojik/spiking substrate'in kendine özgü bir
kısıtı" mı, yoksa "bu görev/döngü tasarımının küçük ağlar için genel
olarak zor olması" mı olduğunu ayırt etmek. Aynı çoklu-problem denetimli
eğitim şeması (25 rastgele kaydırılmış Rastrigin × 12 başlangıç, sosyal
deneyle birebir aynı), ama okuma matrisi ridge regresyon yerine Adam +
MSE ile uçtan uca eğitildi (300 epoch, connectome simülasyonu YOK —
saniyeler süren bir eğitim).

**Sonuç (n=8 tohum, aynı 8 tohum):**

| varyant | ortalama | medyan | gerçek minimumu buldu |
|---|---|---|---|
| classic (PSO, hiçbir ağ yok) | 0.249 | 0.000 | **6/8** |
| tiny_recursive (sabit ağırlık) | 0.383 | 0.430 | 1/8 |
| tiny_recursive_anneal | 0.354 | 0.162 | 3/8 |

**HÜKÜM: TRM-tarzı ağ da klasik PSO'yu geçemedi — hatta connectome
varyantlarının çoğundan (4-6/8) bile kötü çıktı (1-3/8).** seed=0'ın
yakınsama eğrisinde net görünüyor: classic düzgün bir şekilde 1e-7'ye
kadar iniyor, tiny_recursive ~0.15'te düzleşip kalıyor,
tiny_recursive_anneal ~1.0'da uzun süre sıkışıp ancak son birkaç
iterasyonda kaçabiliyor.

**Bunun anlamı önemli:** Sorun "connectome'un biyolojik/spiking
substrate olması" değilmiş — modern, tamamen eğitilebilir, amaca uygun
tasarlanmış küçük bir sinir ağı da AYNI şekilde başarısız oldu. Bu, asıl
zorluğun bu spesifik döngü tasarımında (PSO hızıyla harmanlanan tek-adım
"nüdge" yaklaşımı + Rastrigin'in çok-modlu yapısı) olduğunu, herhangi bir
"substrat" seçiminden bağımsız olduğunu düşündürüyor. FlyOpt'un ana
sonucunu zayıflatmıyor, aksine bağlamlandırıyor: connectome'u suçlamak
yerine, denenen görev çerçevesinin (soyut, tek-adımlı nüdge + PSO)
kendisinin bu tür küçük ağlar için elverişsiz olabileceğini gösteriyor.

## 2026-09-17 — İlk kez: gerçek sinaptik ağırlıkları eğitme denemesi — sonuç BELİRSİZ (yöntem yetersiz kaldı), ÖN TESCİLLİ DEĞİL

Kullanıcının sorusu: "biz connectome'un kendi ağırlığını eğittiğimiz bir
çalışma oldu mu?" — cevap hayırdı, PROTOCOL.md §2.5'teki Fly-RL (tam
sinaptik plastisite) bu güne kadar hiç koşulmamıştı, her fazda (1, 1b,
1c, S1, PSO, TRM) sadece okuma matrisi/arayüz eğitildi, connectome'un
kendi sinapsları hep sabit kaldı — bilinçli bir tasarım (bkz. Faz 1c ve
FlyProposer.py docstring'leri). Kullanıcı "küçük ölçekte dene, birkaç yüz
sinapsla başla" dedi. `src/flyopt/variants/synapse_plasticity.py`,
`scripts/fly_synapse_plasticity_demo.py`.

**Tasarım:** Önce Faz S1'deki gibi bir okuma matrisi kalibre edilip
**dondurulur** (Faz 1c'nin izolasyon mantığı — tek mekanizmayı test et).
Sonra gerçek FlyWire grafından **rastgele 300 sinaps** seçilip, bunların
ağırlıklarına (1+1) tırmanıcı (projenin her yerde kullandığı aynı arama
algoritması, bu sefer connectome'un kendi parametrelerine uygulanıyor)
ile pertürbasyon uygulanıyor; amaç fonksiyonu, dondurulmuş okumanın
çoklu-problem test kümesindeki tahmin hatası (MSE). LIF motoru backprop'a
hazır olmadığı için (spike eşiği türevlenemez, her adımda numpy'a
dönüyor) gradyansız/kara-kutu arama kullanıldı — bu, gerçek bir kısıt,
gelecekte surrogate-gradyan ile aşılabilir ama ayrı bir mühendislik işi.
Eğitim/test ayrımı yapıldı (farklı tohumlu iki küme) — sadece eğitim
kümesinde arama, test kümesinde tek seferlik dürüst ölçüm.

**Sonuç:** 150 iterasyon boyunca **hiçbir pertürbasyon iyileşme
sağlamadı** — theta hiç sıfırdan kıpırdamadı. baseline test_mse=1.3348,
eğitilmiş test_mse=1.3348, **%0.0 değişim**.

**HÜKÜM (temkinli): Bu "sinaptik plastisite işe yaramaz" değil, "bu ucuz
kaba prob hiçbir şey bulamadı" demek.** İki zayıf nokta net: (1) 300
sinaps tamamen rastgele seçildi — afferent→efferent nedensel yolunda
olduğu garanti değil, çoğu muhtemelen bu görevi hiç etkilemiyor; (2)
(1+1) tırmanıcı, 300 boyutta sabit adım büyüklüğüyle — yüksek boyutlu
uzaylarda zayıf bilinen bir yöntem, 150 deneme bir iyileştirme bulmak
için yetersiz kalmış olabilir. Gerçek bir cevap için ya (a) hedefli
sinaps seçimi (gerçekten encode→decode yolunda olan sinapslar) ya da (b)
surrogate-gradyanla gerçek backprop gerekir — ikisi de bu oturumun
kapsamı dışında, ayrı bir mühendislik yatırımı. **Bu soru resmen açık
kalıyor**, "denendi ve işe yaramadı" değil "iyi bir yöntemle henüz
denenmedi" olarak kaydediliyor.

## 2026-09-17 — Sinaptik eğitim v2: hedefli seçim + adaptif arama da sıfır sonuç verdi — daha güçlü bir null

Kullanıcı "biraz daha detaylı ara" dedi. Yukarıdaki v1'in iki zayıflığı
düzeltildi: `select_targeted_synapses` artık rastgele değil, gerçekten
encode/decode nöronlarına **doğrudan bağlı** sinapslardan seçiyor
(11.486 aday, 15M'lik toplam grafın çok küçük ve nedensel olarak
alakalı bir alt kümesi). Arama da `hillclimb_synapses` v2'ye yükseltildi:
Rechenberg'in 1/5 başarı kuralıyla adaptif sigma + blok-koordinat
pertürbasyon (her iterasyonda 300 sinapsın rastgele %15'i, hepsi değil)
— yüksek boyutlu kara-kutu aramada bilinen iki standart iyileştirme.

**Sonuç: 400 iterasyonun TAMAMINDA, 27 adaptasyon penceresinin
HİÇBİRİNDE tek bir başarı yok (success_rate=0.00, sürekli).** Sigma
1/5 kuralı gereği aralıksız küçüldü (0.30 → 0.0017) ama hiçbir
pertürbasyon iyileşme sağlamadı. baseline test_mse=1.3348, eğitilmiş
test_mse=1.3348, **yine %0.0 değişim, theta 400 iterasyon sonunda hâlâ
tamamen sıfır.**

**HÜKÜM (v1'den daha güçlü, ama hâlâ kesin değil): Bu artık "zayıf
yöntem" açıklamasıyla kolayca bir kenara atılamaz** — hem hedefli seçim
hem adaptif/blok arama denendi, ikisi de aynı sıfır sonucu verdi. En
olası açıklama: T=15 adım boyunca ortalanan ateşlenme oranı gibi toplu
bir istatistik, tek bir sinapsın ağırlık değişimine karşı doğal olarak
gürültüye dayanıklı/sağlam — yoğun bağlantılı, 15M kenarlı bir ağda 300
sinapsın (aday havuzun ~%2.6'sı, %30'a varan pertürbasyon büyüklüğüyle)
nedensel etkisi bu ölçüm çözünürlüğünün altında kalıyor olabilir. Bu,
biyolojik ağların bilinen bir özelliğiyle (dağıtık temsil, tek
bağlantıya karşı sağlamlık) tutarlı, ama kanıtlanmış değil — gerçek bir
cevap hâlâ gradyan-tabanlı (surrogate-gradyanla backprop) bir yaklaşım
gerektirir, ki bu ayrı bir mühendislik yatırımı olarak kaydediliyor.
**Bu oturumda konuyu burada bırakıyoruz** — sinaptik plastisitenin bu
projede (kara-kutu aramayla, makul bir bütçede) ölçülebilir bir etkisi
bulunamadı.

## 2026-09-17 — Türevlenebilir "rate" substrate pilotu (cesp99/fly-chess incelemesinden sonra): düzeltilmiş metodolojiyle bile aynı sonuç

`cesp99/fly-chess`'i klonlayıp inceledik (kullanıcı isteğiyle,
"neler yapmışlar nasıl öğretmişler değerlendir"). Bizden temel farkı:
LIF (sızma, türevlenemez) yerine sürekli "leaky rate" nöron modeli
kullanıyorlar (`h_{t+1}=(1-a)h_t+a·act(Wh_t+girdi)`), Dale yasasını
koruyarak (`w=sign×softplus(gain)`) ama **gerçek backprop** ile 2.7
milyon bağlantının tamamının büyüklüğünü eğitiyorlar — 58 milyon
Lichess pozisyonuyla. Bizim tüm gün süren kara-kutu aramalarımızın
(sinaps deneyi dahil) aksine.

**Pilot:** `src/flyopt/variants/rate_brain.py`,
`scripts/fly_rate_brain_pilot_v2.py`. Küçük ölçekte (3000 nöronluk
alt-graf, tam connectome değil) aynı mimariyi kurduk, aynı çoklu-problem
Rastrigin görevinde (gerçek connectome vs er_null) test ettik.

**İlk deneme (v1) kritik bir metodoloji hatası ortaya çıkardı:**
Encode/decode nöronlarını bu projede her zaman yaptığımız gibi
(`rng.choice`, afferent/efferent havuzundan rastgele) seçtiğimizde, 4
encode nöronundan **2'sinin decode havuzuna hiçbir gerçek yolu yoktu**
(tam graf üzerinde doğrulandı). Enjekte ettiğimiz sinyalin yarısı hiçbir
yere gitmiyordu — eğitim eğrisi "hiçbir şey öğrenmedim" seviyesinde
(MSE≈0.49, birim-vektör hedeflerin ortalama karesi) düzleşti.

**Düzeltme:** `select_connected_encode_decode` — encode/decode nöronları
artık rastgele değil, **tam graf üzerinde ≤6 sıçramada gerçekten
bağlı olduğu doğrulanmış** çiftlerden seçiliyor. Bu düzeltmeden sonra
gerçek connectome alt-grafında da encode nöronlarının 3/4'ü decode
havuzunun tamamına ulaşabiliyor (%75 çift-bazında erişim), er_null'da
(rastgele yerleşim sayesinde) %100.

**Sonuç (düzeltilmiş, n=1 tohum, pilot):** Bu sefer **ikisi de gerçekten
öğreniyor** (ikisi de 0.5 taban çizgisinin altına iniyor — v1'in aksine).
real_connectome final mse=0.4867, er_null final mse=0.4839 — **%0.6
fark, pratikte eşit.**

**HÜKÜM: Metodoloji hatası düzeltildi ama sonuç değişmedi.** Türevlenebilir
mimari + gerçek gradyanla sinaps eğitimi + doğrulanmış bağlantılı
encode/decode kullanılsa bile, gerçek connectome ile ER-null arasında
ölçülebilir bir fark yok — bugün denenen her yaklaşımın (Faz S1, PSO,
sosyal ışınlar, sinaptik plastisite, şimdi de bu) vardığı aynı sonuç.

**Ayrı ama önemli bir yan bulgu — metodolojik uyarı, geriye dönük:** Bu
projede bugüne kadar kurulan HER encode/decode seçimi (`FlyProposerScene`,
`FlySwarmSocial`, sinaps deneyi) afferent/efferent havuzundan `rng.choice`
ile rastgele yapıldı, hiçbiri bağlantı doğrulaması içermiyordu. Bu pilot,
böyle bir seçimin **gerçek anatomide sıfır yol** ile sonuçlanabildiğini
kanıtladı. Bugünkü diğer deneylerin çoğunda (özellikle Faz S1 ve sosyal
deney, T=15 adım ve daha büyük havuzlarla) bu riskin ne kadar
gerçekleştiği bilinmiyor — geriye dönük doğrulanmadı, ama gelecekte
connectome-tabanlı herhangi bir deneyde encode/decode seçiminin
bağlantı-doğrulamalı yapılması artık standart olmalı.

## 2026-09-17 — Malecns resmileştirildi: 8 tohumda MÜKEMMEL pozitif sonuç (Cliff's δ=1.000)

2026-09-16'daki gayrı-resmi tek-tohum gözlem (`fly_demos/malecns`, F1
araba sürüşü, gerçek MaleCNS connectome vs ER-null, DAgger kalibrasyonu)
kullanıcı isteğiyle ("malecns'i resmilestir") 8 tohumlu, istatistiksel
bir teste dönüştürüldü. `fly_demos/malecns/multiseed_compare.py` —
`calibrate.py`'yi her (substrate, seed) çifti için sıfırdan çalıştırıp
"[final] brain alone" satırını ayrıştırıyor, `calibrate.py`'nin kendi
koduna dokunmuyor. Bu proje FlyOpt'un pre-registration disiplinine
(PROTOCOL.md) tabi değil — malecns ayrı bir üçüncü-taraf proje — ama
aynı istatistiksel testi (Wilcoxon işaretli-sıra, Cliff's δ) uyguladık.

**Sonuç (n=8 tohum, laps_mean, kalibrasyon sonrası "brain alone" değerlendirmesi):**

flywire: [0.142, 0.119, 0.133, 0.142, 0.119, 0.137, 0.138, 0.147] (ortalama 0.1346, medyan 0.1375)
er_null: [0.057, 0.075, 0.074, 0.075, 0.075, 0.074, 0.074, 0.074] (ortalama 0.0723, medyan 0.0740)

**Wilcoxon p=0.0078** (8 tohum için ulaşılabilecek en küçük p-değeri —
flywire HER TOHUMDA er_null'dan iyi çıktı, istisna yok).
**Cliff's δ=1.000** (mutlak maksimum — 64 eşleşmeli karşılaştırmanın
hepsinde flywire üstün). **Kapı testi (p<0.05 VE |δ|>0.33) tam anlamıyla
GEÇTİ**, hem de en güçlü mümkün değerlerle.

**Bu, FlyOpt'un ana fazlarının (1/1b/1c/S1, hepsi negatif) TERSİ bir
sonuç ve ikisi de doğru — çünkü farklı bir soru soruyorlar.** Faz
1-S1'in sorduğu soru ("connectome soyut, genel-amaçlı optimizasyonda
avantaj sağlar mı") negatif cevaplandı; bu deneyin sorduğu soru
("connectome, kendi evrimleştiği türden sürekli duyusal-motor kontrol
görevinde avantaj sağlar mı") kusursuz pozitif cevaplandı. İkisi
birlikte, tek bir tutarlı hikaye anlatıyor: **connectome'un
avantajı görev-mimari eşleşmesine bağlı, genel bir hesaplama üstünlüğü
değil.** (Bkz. 2026-09-17 "sineğe haksızlık mı yapmışız" tartışması.)

**Not — bu resmi olmayan bir FlyOpt fazı değil**, PROTOCOL.md'nin
pre-registration/gate disiplinine tabi değil (ayrı proje, ayrı
kod tabanı). Ama istatistiksel titizlik aynı standartta uygulandı ve
sonuç, FlyOpt'un ana anlatısını tamamlayan, kayda değer bir bulgu.

## 2026-09-17 — Fly-Surrogate (PROTOCOL.md #6, hiç denenmemişti): yön kusursuz, büyüklük eşik altı — karma sonuç

Kullanıcının "hangi mimariler denenebilir" sorusuna verdiğim fikir
listesinden biri: connectome'u "nereye git" (yön öner) yerine "bu
noktada değer ne olurdu" (fonksiyon değerini tahmin et) rolünde
kullanmak — PROTOCOL.md'nin Fly-Surrogate fikri, bu projede ilk kez
test edildi. `src/flyopt/variants/fly_surrogate.py`,
`scripts/fly_surrogate_multiseed.py`. Kullanıcı çevrimdışıyken (~1-1.5
saat) çalışması için kuyruğa alındı. Aynı doğrulanmış-bağlantılı
encode/decode seçimi ve türevlenebilir rate substrate (rate_brain.py)
kullanılıyor; tek fark görev — ham x koordinatları enjekte ediliyor,
tek skaler çıktı (normalize edilmiş Rastrigin değeri) okunuyor, MSE ile
regresyon olarak eğitiliyor. Baştan 8 tohumlu (tek-tohum "bak sonra
genişlet" hatasına düşülmedi).

**Sonuç (n=8 tohum, test_mse, düşük=iyi):**

real_connectome: [0.830, 0.944, 0.924, 1.003, 0.978, 0.598, 1.170, 0.841] (medyan 0.934)
er_null:         [0.883, 0.964, 0.974, 1.074, 0.988, 0.666, 1.196, 0.851] (medyan 0.969)

**Gerçek connectome 8 tohumun HEPSİNDE er_null'dan düşük (daha iyi) hata
verdi** — Wilcoxon p=0.0078 (8 tohum için ulaşılabilecek en yüksek
anlamlılık, malecns'le aynı p-değeri). Ama **Cliff's δ=0.188** —
pre-tescilli "anlamlı" eşiğin (0.33) belirgin altında. **Kapı testi
(p<0.05 VE |δ|>0.33) TEKNİK OLARAK FAIL.**

**Yorum:** Bu, Faz 1b/S1'in "sınırda kaldı" tablosuna benziyor ama farklı
bir şekilde — yön bu sefer %100 tutarlı (8/8), sadece büyüklük mütevazı.
Cliff's delta'nın küçük kalması, tohumlar arası varyansın kendi
içindeki (paired) farktan daha büyük olduğunu gösteriyor — yani connectome
her zaman kendi eşleşen er_null'undan iyi ama bazı "kötü" connectome
tohumları bazı "iyi" er_null tohumlarını geçemiyor.

**Malecns (δ=1.000, kusursuz) ile karşılaştırıldığında:** Fly-Surrogate
rolü, malecns'in sürekli duyusal-motor kontrolü kadar güçlü bir
avantaj göstermiyor ama Faz 1-S1'in soyut "yön öner" rolünden de
belirgin şekilde daha iyi (en azından yön tutarlılığı açısından). Bu,
"görev-mimari eşleşmesi" hipotezine yeni bir veri noktası ekliyor:
regresyon/tahmin rolü, navigasyon rolünden daha iyi ama embodied kontrol
rolünden daha kötü — bir ara nokta. Kesin değil (n=8, tek deney), ama
görev tipinin bir spektrum üzerinde sıralanabileceğini düşündürüyor:
soyut optimizasyon < fonksiyon tahmini < sürekli duyusal-motor kontrol.

## 2026-09-18 — Şev stabilitesi 8 tohuma çıkarıldı: GÜÇLÜ POZİTİF (Cliff's δ=0.906), PROTOCOL_V2 spektrum noktası #3 doğrulandı

`scripts/fly_slope_stability_multiseed.py` — tek-tohum pilotun (+%3.4,
belirsiz) 8 tohuma çıkarılmış hali, aynı doğrulanmış-bağlantılı
encode/decode + türevlenebilir rate-brain altyapısı.

**Sonuç (n=8 tohum, test_mse, düşük=iyi):**

real: [0.122, 0.097, 0.113, 0.097, 0.144, 0.118, 0.128, 0.093] (medyan 0.1155)
null: [0.178, 0.136, 0.184, 0.138, 0.144, 0.149, 0.170, 0.158] (medyan 0.1535)

**Wilcoxon p=0.0156, Cliff's δ=0.906** — gerçek connectome 8 tohumun
7'sinde net farkla önde, 1 tohumda (seed=4) neredeyse eşit. **Kapı testi
(p<0.05 VE |δ|>0.33) GEÇTİ.**

**PROTOCOL_V2_TASK_SPECTRUM.md'nin H2'sini güçlü şekilde destekliyor.**
Artık üç bağımsız uzamsal/fiziksel görevde tutarlı pozitif sinyal var:

| Görev | Cliff's δ | Kapı |
|---|---|---|
| Soyut optimizasyon (4 mimari) | ~0 ile -0.34 | FAIL |
| Fonksiyon tahmini (Fly-Surrogate) | 0.188 | FAIL (eşik altı) |
| **Uzamsal fonksiyon tahmini (şev stabilitesi)** | **0.906** | **PASS** |
| Sürekli duyusal-motor kontrol (malecns) | 1.000 | PASS |

Şev stabilitesi ile malecns arasındaki fark (0.906 vs 1.000, ikisi de
"gerçek fiziksel/uzamsal yapı" taşıyor) ilginç bir ayrıntı: malecns
sürekli kapalı-döngü (her adımda geri besleme), şev stabilitesi
tek-adımlı bir yerel gradyan tahmini (Faz S1/PSO'nun ışın kodlamasına
daha yakın). Bu, spektrumun "uzamsal olmak" ile "sürekli kapalı-döngü
olmak" olarak iki ayrı ekseni olabileceğini düşündürüyor — ikisi de
katkı sağlıyor, ayrı ayrı test edilmeleri gerekebilir.

## 2026-09-18 — Cartpole (spektrum noktası #4): BEKLENMEDİK NEGATİF sonuç — H2'nin basit hali yeterli değil

`scripts/fly_cartpole_multiseed.py`, `src/flyopt/variants/cartpole_control.py`
— PROTOCOL_V2'nin planladığı "ucuz, sürekli kapalı-döngü kontrol" ara
noktası. Fizik + beyin uçtan uca türevlenebilir (backprop ile, ödül
şekillendirme yok, doğrudan kontrol maliyeti = kayıp fonksiyonu).

**Sonuç (n=8 tohum, eval_cost, düşük=iyi):**

real: [5.393, 3.646, 4.328, **15.859**, 4.561, 2.914, 1.697, 5.073] (medyan 4.445)
null: [3.621, 3.548, 3.770, 1.833, 3.553, 6.348, 7.179, 2.498] (medyan 3.587)

**Wilcoxon p=0.547 (anlamsız), Cliff's δ=-0.219 (hafifçe null lehine).
Kapı testi FAIL** — malecns'in (δ=1.000) ve şev stabilitesinin
(δ=0.906) aksine, burada gerçek connectome hiçbir avantaj göstermiyor,
yön bile belirsiz/hafif ters.

**Şüpheli bir ayrıntı:** seed=3'teki gerçek-connectome sonucu (15.86)
diğerlerinin (1.7-5.4 aralığı) 3-9 katı — muhtemelen 50 adımlık bağlı
fizik+beyin geri-yayılımının o tohumun başlangıç ağırlıklarıyla
ıraksaması (patlayan gradyan), gerçek bir kontrol başarısızlığı değil.
Bu tek uç değeri çıkarınca bile (real medyan ≈4.33) fark hâlâ null
lehine kalıyor — yani sonuç sadece bu uç değerden kaynaklanmıyor, ama
eğitim istikrarının bu görevde şüpheli olduğu **doğrulanmadan** ileri
bir yorum yapılmamalı.

**HÜKÜM (temkinli): H2'nin basit hali ("sürekli kapalı-döngü kontrol
yeterli") bu sonuçla desteklenmiyor.** Olası açıklamalar (test
edilmedi, sadece hipotez):
1. Cartpole'un çok hızlı/kısa zaman ölçeği (dt=0.02s, kutup birkaç
   adımda düşebilir) connectome'un zaman sabitleriyle (leak/tau)
   uyuşmuyor olabilir — malecns'in kontrol adımı 16ms ama episode
   çok daha uzun (binlerce adım), şev stabilitesi zaman-bağımsız.
2. 4 boyutlu, tek-skaler-çıkışlı bir görev, connectome'un "çok
   sayıda duyusal girdiden süzme" avantajını hiç kullandırmıyor
   olabilir — girdi zaten düşük boyutlu.
3. Eğitim istikrarsızlığı (yukarıdaki uç değer) sonucu gürültüye
   boğmuş olabilir; daha fazla tohum veya gradyan kırpma (gradient
   clipping) ile tekrar denenmeli.

**PROTOCOL_V2_TASK_SPECTRUM.md güncellenmeli:** H2'nin monoton
spektrum iddiası bu haliyle doğrulanmadı — "uzamsal yapı" (şev
stabilitesi, malecns) ile "sürekli kapalı-döngü olmak" (cartpole)
ayrı özellikler, ikisi birden olmadan (cartpole sadece ikincisini
taşıyor, uzamsal yapı yok) avantaj ortaya çıkmıyor olabilir. Bu,
basit tek-boyutlu bir sıralama yerine **çok-boyutlu bir görev
uzayı** gerektirebilir.

## 2026-09-18 — Cartpole v2 (gradyan kırpma): istikrarsızlık şüphesi DOĞRULANDI, yön düzeldi ama kapı hâlâ FAIL

`train_cartpole`'a `torch.nn.utils.clip_grad_norm_` (norm=1.0) eklendi
— 50 adımlık bağlı fizik+beyin geri-yayılımında patlayan gradyan
şüphesini test etmek için. Aynı 8 tohum, aynı altyapı.

**Sonuç:**

real: [3.557, 3.993, 1.034, 1.241, 1.909, 2.939, 2.442, 7.330] (medyan 2.691)
null: [3.817, 2.702, 4.833, 1.869, 3.242, 4.131, 2.010, 13.699] (medyan 3.529)

**Şüphe doğrulandı:** seed=3'ün gerçek-connectome sapması (15.86)
tamamen düzeldi (1.24) — gradyan patlaması gerçekmiş. **Yön de
düzeldi**: artık 6/8 tohumda gerçek connectome daha düşük (iyi)
maliyet veriyor (v1'de yön kararsızdı). Wilcoxon p=0.148 (hâlâ anlamsız
ama v1'in p=0.547'sinden çok daha yakın), **Cliff's δ=0.312** (pozitif,
eşiğin — 0.33 — hemen altında). **Kapı testi hâlâ TEKNİK OLARAK FAIL**
ama artık malecns/şev-stabilitesi yönüyle tutarlı, sadece istatistiksel
güç (n=8) yetersiz kalmış olabilir.

**Sonraki adım:** Daha fazla tohumla (16-20) tekrarlamak, δ=0.312'nin
eşiği geçip geçmediğini görmek için makul bir yatırım — sinyal gerçek
görünüyor, sadece n=8'de kanıtlamaya güç yetmiyor olabilir.

## 2026-09-18 — Cartpole v2, 16 tohuma çıkarıldı: sinyal zayıfladı, NİHAİ SONUÇ FAIL

8 tohum daha eklendi (seed 8-15, aynı gradyan-kırpmalı kurulum). Yeni
8 tohumda gradyan kırpma bu sefer sapmaları tam önleyemedi (seed=11
real=14.25, seed=14 real=12.02 — hâlâ büyük uç değerler var).

**Birleşik sonuç (n=16 tohum):**

Wilcoxon p=0.744, **Cliff's δ=0.133** (ilk 8 tohumun δ=0.312'sinden
belirgin küçülmüş). Gerçek connectome 16 tohumun 10'unda (62.5%) daha
iyi ama fark istatistiksel olarak anlamsız. **Kapı testi kesin olarak
FAIL.**

**HÜKÜM (nihai, cartpole için): İlk 8 tohumdaki umut verici yön kısmen
şanstı.** Daha büyük örneklemde gerçek etki zayıf ve anlamsız kalıyor.
Gradyan kırpma sapmaları azalttı ama tamamen çözmedi — bu görev
(kesikli, hızlı, düşük-boyutlu kapalı-döngü kontrol) hem eğitim
istikrarı açısından zor hem de connectome'a net bir avantaj
kazandırmıyor. **PROTOCOL_V2'nin H2' hipotezi (uzamsal yapı > kapalı-döngü
olmak) 16-tohumluk bu sonuçla da destekleniyor** — cartpole'da uzamsal
yapı yok, ve avantaj da yok/zayıf; şev stabilitesi ve malecns'te uzamsal
yapı var, ve avantaj güçlü/mutlak.

## 2026-09-18 — Sürekli (kapalı-döngü) şev stabilitesi: BEKLENMEDİK — tek-atış versiyonun tersi yönde, ama muhtemelen zayıf eğitimden

Son boş hücreyi doldurmak için (`src/flyopt/variants/slope_continuous.py`,
`scripts/fly_slope_continuous_multiseed.py`): aynı şev stabilitesi
görevi, ama artık tek-atış (ışın→yön regresyonu) değil, **kalıcı beyin
durumuyla çok-adımlı (20 adım) bir arıtma süreci** — cartpole'un
rollout mantığına benzer, malecns'ten çok daha ucuz.

**Sonuç (n=8 tohum, eval_fs = 20 adımlık aramada bulunan en iyi FS,
düşük=iyi):**

real: [17.10, 29.54, 8.69, 18.35, 18.66, 18.72, 17.71, 18.77] (medyan 18.50)
null: [16.36, 26.18, 13.36, 17.08, 17.19, 17.47, 18.72, 19.01] (medyan 17.33)

**Wilcoxon p=0.461, Cliff's δ=-0.188** — tek-atış versiyonun (δ=0.906,
GEÇTİ) **tam tersi yönde**, anlamsız. **Kapı testi FAIL.**

**ÖNEMLİ ŞÜPHE — bu sefer net bir "hipotez çürüdü" değil:** Bulunan
FS değerleri (8.7-29.5) gerçek kritik değerden (aynı tip şevler için
tek-atış pilotunda ~1.5-2 civarı bulunuyordu) çok uzak — yani **hem
gerçek hem null, 20 adımda iyi bir arama yapamamış olabilir.** Eğitim
rejimi tek-atış versiyondan çok daha zayıf: sadece 80 epoch, 10
problem, problem başına 3 epizot (tek-atış versiyon çok daha büyük,
toplu bir regresyon kümesi kullanıyordu). Bu, sonucun "kapalı-döngü
zarar veriyor" değil "ikisi de yetersiz eğitilmiş, karşılaştırma
gürültüde kaybolmuş" olabileceğini düşündürüyor.

**HÜKÜM: Bu deney şu haliyle H2'yi ne doğruluyor ne çürütüyor —
yetersiz güçte (underpowered), tekrar denenmeden yorum yapılmamalı.**
Gelecekte doğru test için: (a) çok daha fazla epoch/epizot (tek-atış
versiyonla eşit "toplam örnek" bütçesi), (b) eğitim ilerlemesinin
gerçekten yakınsadığını (curve'ün düzleştiğini) doğrulamak — bu
oturumda kontrol edilmedi. Bu, gecenin en net "daha fazla iş gerekiyor"
bulgusu; H2' hipotezi (uzamsal yapı > kapalı-döngü) hâlâ en iyi
desteklenen açıklama ama bu son deney onu ne güçlendirdi ne zayıflattı.

## 2026-09-18 — Sürekli şev stabilitesi v2 (5x eğitim bütçesi): sorun eğitim miktarı değilmiş, YAKINSAMIYOR

`verbose_every=50` ile eğitim eğrisi izlendi, epoch 400'e (80'in 5 katı)
çıkarıldı, epizot/problem 5'e (3'ten) çıkarıldı.

**Bulgu: eğitim eğrisi hiç düzgün yakınsamıyor, salınım yapıyor.**
Örnek (seed=6, gerçek connectome): `mean_best_fs_this_batch` epoch
boyunca 30.7 → 23.1 → **2.5** → 13.1 → 21.1 → 12.8 → 31.1 → **1.9** →
16.3 şeklinde zıplıyor — ara sıra gerçek minimuma yakın (~2) değerler
buluyor ama orada KALMIYOR, bir sonraki epoch'ta tekrar 20-30'a
sıçrıyor. Kayıp (loss) da benzer şekilde dalgalı (0.55→0.24→0.27→
0.24→...), monoton azalmıyor.

**Sonuç (n=8 tohum):** Wilcoxon p=0.039 (bu sefer anlamlı!) ama
**Cliff's δ=-0.156** — yön null lehine, eşiğin (0.33) çok altında.
**Kapı testi FAIL.**

**HÜKÜM (nihai, bu tasarım için): Sorun eğitim miktarı değil, eğitim
REJİMİNİN kendisi.** Her epoch'ta sadece TEK bir arazi üzerinde 5
epizotluk bir gradyan adımı atmak, çok gürültülü bir sinyal üretiyor —
ağ bir yakınsama noktasına oturamıyor, sürekli farklı yönlere
sürükleniyor. Bu, "kapalı-döngü zarar veriyor" (H2' aleyhine) ya da
"kapalı-döngü fark etmiyor" şeklinde OKUNAMAZ — çünkü karşılaştırmanın
kendisi (gerçek vs null) da bu gürültülü/yakınsamayan rejimde anlamlı
değil. **Doğru bir test için tasarım değişmeli**: ör. her adımda TEK
epizot yerine çok sayıda arazi/başlangıç noktasının EŞ ZAMANLI
(batched) ortalamasını almak — cartpole'un yaptığı gibi (orada
batch=16, tek seferde tüm batch üzerinden gradyan alınıyordu, bu yüzden
cartpole'un eğrisi çok daha istikrarlıydı). **Bu, gecenin son ve en
önemli mühendislik dersi**: continuous/multi-step görevlerde episoda-
başına-tek-gradyan yerine geniş-batch gradyan şart.

**PROTOCOL_V2 durumu:** "Uzamsal-VAR + kapalı-döngü-VAR, ucuz" hücresi
hâlâ güvenilir şekilde doldurulamadı — iki deneme de (80 ve 400 epoch)
aynı yakınsamama sorununu gösterdi, düzeltme daha fazla epoch değil
farklı bir batching stratejisi gerektiriyor. Bu, yeni bir oturumda
(batched multi-geometry gradient ile) tekrar denenmeli.

## 2026-09-18 — KRİTİK BULGU: `benchmarks_geo.py`'de %43.6 geçersiz-başlangıç hatası — sürekli şev stabilitesi deneylerinin hepsi bu yüzden bozuk

Batched-gradient düzeltmesinin (v3) ilk sonucu da v1/v2 ile neredeyse
birebir aynı çıkınca (17.07 vs 17.10 vs 17.20 — üç farklı eğitim
rejiminde!) şüphelenip tek bir epizotu elle izledim: **`best_fs=50.0`
(LARGE_FS ceza değeri) baştan sona hiç değişmiyordu.** Kök neden
bulundu ve ölçüldü:

```
rng = np.random.default_rng(0)
500 rastgele (geometry, x0) örneğinden 218'i (%43.6) LARGE_FS (geçersiz daire)
```

**Neden kritik:** `_rays()` geçerlilik kontrolü yapmıyor — eğer x0 VE
onun etrafındaki tüm ışın-prob noktaları da (aynı geniş geçersiz
bölgede) LARGE_FS dönüyorsa, `rays = f(x+step) - f(x) = 50-50 = 0`
her yönde — yani **yerel gradyan sinyali tamamen sıfır**, öğretmen
yönü de sıfıra normalize oluyor. Sürekli/çok-adımlı arama bu düz
bölgeden ASLA çıkamıyor (yerel bilgiyle kaçış imkanı yok), süresiz
olarak 50'de kilitli kalıyor. `v3`'ün "iyileşmiş" batching'i bu yüzden
hiçbir şey değiştirmedi — sorun optimizasyon değil, görev tanımıydı.

**Hangi deneyler ne kadar etkilendi:**

- **Sürekli şev stabilitesi (v1, v2, v3 — hepsi):** KATASTROFİK ölçüde
  bozuk. Kaydedilen "18-20 civarı" ortalama FS değerleri gerçekte
  "%44 kilitli-kalmış (50) + %56 gerçekten arama yapabilen (~2-5)"
  karışımından geliyor — optimizasyon sürecinin kendisi hakkında
  hiçbir şey söylemiyor. **Bu üç deneyin (bugünkü EXPERIMENTS.md
  girdileri) hiçbiri H2/H2' hakkında geçerli kanıt değil, iptal
  sayılmalı.**
- **Tek-atış şev stabilitesi (δ=0.906, GEÇTİ):** Muhtemelen çok daha
  az etkilenmiş — bu bir regresyon (çok sayıda (x, hedef) örneğine
  toplu MSE fit), tek bir kötü örnek süresiz kilitlenmeye yol açmıyor,
  sadece hedef sıfır olan bir örnek gibi davranıyor (gürültü, felaket
  değil). Daha da önemlisi: **gerçek ve null substrate'ler AYNI
  rastgele tohumla AYNI (geometry, x) örneklemini görüyor** — yani
  geçersiz-örnek oranı ikisi için de birebir eşit, karşılaştırmanın
  YÖNÜ (hangisi daha iyi) muhtemelen hâlâ geçerli, sadece mutlak MSE
  değerleri şişirilmiş olabilir. **Yine de bu tam olarak doğrulanmadı
  — ihtiyatla, "muhtemelen geçerli ama teyit edilmeli" olarak
  işaretleniyor.**

**HÜKÜM:** PROTOCOL_V2'nin spektrum tablosundaki "uzamsal-VAR +
kapalı-döngü-VAR, ucuz" hücresi hâlâ boş — üç deneme de (v1/v2/v3)
aynı temel görev-tanımı hatası yüzünden geçersiz. Düzeltme basit:
`SlopeGeometry.param_bounds()` ya sıkılaştırılmalı (geçerli daire
olasılığını artıracak şekilde) ya da x0/ışın örneklemesi geçerlilik
kontrolüyle (reddet-ve-yeniden-örnekle) yapılmalı. **Bu, yeni bir
oturumda düzeltilip yeniden denenmeli** — bu geceki üç continuous
deney (v1, v2, v3) sonuç tablosundan çıkarılmalı/geçersiz sayılmalı.

**Genel ders (protokolün ruhuna uygun bir örnek):** Sonucun "çok
tutarlı" görünmesi (üç farklı eğitim rejiminde aynı sayı) bir başarı
işareti değil, bir **şüphe sinyali** olarak okunmalı — PROTOCOL.md'nin
kendi kuralı ("sonuç sürpriz çıkarsa önce kodu şüphelen") burada tam
tersinden de işledi: sonuç "sürpriz derecede tutarlı" çıktığında da
şüphelenmek gerekiyormuş.

## 2026-09-18 — Sürekli şev stabilitesi v4 (hata düzeltilmiş): TEMİZ sonuç, ama H2'yi karmaşıklaştırıyor

`benchmarks_geo.py::sample_valid_point` (reddet-ve-yeniden-örnekle)
eklendi, `slope_continuous.py`'nin her iki x0 örnekleme noktasına
uygulandı. Doğrulama: 200/200 örnek artık geçerli (önce 200'de ~87
geçersizdi). Aynı düzeltilmiş batching (v3'ün geometries_per_step=6
mantığı), 8 tohum.

**Sonuç — artık FS değerleri gerçekçi aralıkta (2.4-5.5, önceki
17-30'un aksine):**

real: [4.157, 3.003, 4.052, 3.068, 2.580, 3.646, 5.495, 2.514] (medyan 3.357)
null: [3.690, 3.156, 3.859, 2.634, 2.399, 3.096, 4.512, 2.478] (medyan 3.126)

**Wilcoxon p=0.0234 (ANLAMLI), Cliff's δ=-0.156** — gerçek connectome
8 tohumun 7'sinde er_null'dan biraz DAHA KÖTÜ (sadece seed=1'de iyi).
**Kapı testi FAIL** (etki büyüklüğü eşik altı, ama bu sefer p değeri
gerçek/temiz — Fly-Surrogate'in "yön tutarlı ama küçük" örüntüsüne
benzer, sadece yön ters).

**Bu, H2'yi DOĞRULAMIYOR, karmaşıklaştırıyor.** Aynı uzamsal görevin
(şev stabilitesi) iki versiyonu şimdi taban tabana zıt yönlerde:

| Versiyon | δ | Kapı |
|---|---|---|
| Tek-atış (toplu regresyon) | **+0.906** | PASS |
| Sürekli (kalıcı durum, çok-adımlı arama) | **-0.156** | FAIL |

Aynı FS yüzeyi, aynı ışın kodlaması, aynı connectome — sadece eğitim/
değerlendirme çerçevesi (toplu regresyon vs kalıcı-durumlu iteratif
arama) değişti ve yön neredeyse tersine döndü. **Bu, "uzamsal yapı
tek başına yeterli" (H2') basitleştirmesini de zayıflatıyor** —
malecns (uzamsal + kapalı-döngü, δ=1.000) ile bu deney (uzamsal +
kapalı-döngü, δ=-0.156) aynı iki özelliği taşıyor ama tamamen farklı
sonuç veriyor. Muhtemel fark: malecns'in girdisi gerçek/zengin bir
duyusal modaliteye (lidar, sürekli fiziksel sürüş) dayanıyor ve çok
daha büyük bir eğitim yatırımı (DAgger, ES) alıyor; bu deneyin ışın
kodlaması soyut kalıyor ve eğitim bütçesi çok daha mütevazı.

**HÜKÜM (dürüst, nihai): Görev-mimari eşleşmesi hipotezi (H2/H2')
tek bir özellikle (uzamsal olmak, kapalı-döngü olmak) açıklanamayacak
kadar karmaşık görünüyor.** En sağlam iki pozitif sonuç (malecns
δ=1.000, tek-atış şev stabilitesi δ=0.906) hâlâ duruyor ve gerçek —
ama "neden" sorusunun basit bir tek-boyutlu cevabı yok. Muhtemel ek
değişkenler: eğitim rejiminin zenginliği (toplu regresyon > iteratif
küçük-batch arama), duyusal girdinin gerçekçiliği (gerçek lidar/görme
> soyut sonlu-fark ışınları), episode uzunluğu/eğitim bütçesi ölçeği.
Bu, gecenin en olgun sonucu: **basit bir teoriyle yetinmek yerine,
verinin gerçekte söylediğini kabul etmek** — hipotez revize edilmeye
devam ediyor, kapanmadı.

## 2026-09-18 — Ek netleştirme: iki şev-stabilitesi versiyonu aslında farklı şeyler ölçüyor (DAgger değil, metrik farkı)

Yukarıdaki karmaşıklaştırıcı sonucun (δ=+0.906 vs δ=-0.156) nedenini
araştırırken önce "exposure bias / DAgger eksikliği" düşünüldü, ama
kodu tekrar incelenince bu yanlış çıktı: `run_episode` zaten kendi
ürettiği tahminle (`pred.detach()`) bir sonraki konuma geçiyor ve
öğretmen hedefini HER ziyaret edilen durumda veriyor — yani zaten
on-policy, DAgger'ın çözdüğü sorunu zaten taşımıyor.

**Daha temel ve daha önemli fark bulundu: iki versiyon aslında FARKLI
ŞEYLERİ ölçüyor, aynı şeyin iki versiyonu değil.**

- **Tek-atış:** "Bu noktada yerel gradyan yönünü ne kadar doğru tahmin
  ediyorsun?" — saf bir regresyon doğruluğu metriği (MSE). Tek bir
  adımın kalitesini ölçüyor, hatalar birikmiyor.
- **Sürekli:** "20 adımlık gerçek bir aramada nereye varıyorsun?" —
  uçtan uca bir arama başarısı metriği. Her adımdaki küçük bir hata
  bile sonraki adımın başlangıç noktasını kaydırıyor, 20 adım boyunca
  BİRİKEBİLİR (compounding error) — yerel tahmin doğruluğu yüksek olsa
  bile, küçük sistematik bir sapma 20 adımda büyük bir kümülatif
  sürüklenmeye dönüşebilir.

**Bu, δ=0.906'nın δ=-0.156'ya dönüşmesini açıklayan en olası
mekanizma:** connectome'un yerel yön tahmini gerçekten iyi olabilir
(tek-atış bunu gösteriyor) ama bu iyilik, 20 adımlık bir yörüngede
küçük sistematik bir yanlılığı (bias) telafi edemeyecek kadar ince
olabilir — oysa er_null'un daha "kaba ama tutarlı" bir tahmini,
kümülatif sürüklenmeye karşı tesadüfen daha dayanıklı çıkmış olabilir.
Bu spekülasyon test edilmedi (gecenin sonu) ama **iki metriğin
doğrudan karşılaştırılabilir olmadığını** göstermesi bakımından önemli
— PROTOCOL_V2'nin spektrum tablosundaki 3 ve 3b satırları "aynı görev,
farklı zorluk" değil, kısmen "farklı soru" olarak okunmalı.

**Bu oturum için son değerlendirme:** Bu gece toplam 8 farklı deney
(4 mimari × soyut görev, Fly-Surrogate, şev stabilitesi ×2 versiyon,
cartpole ×3 tekrar) + 1 kritik hata bulma/düzeltme + malecns'in
resmileştirilmesi yapıldı. Hipotez basit bir tek-boyutlu açıklamadan
("uzamsal>kapalı-döngü") çok daha temkinli, çok-boyutlu bir açık
soruya evrildi. Bu, ilerleme değil geri adım gibi görünebilir ama
protokolün ta başından beri savunduğu şeyin ta kendisi: **erken,
basit bir teoriyle tatmin olmamak.**

## 2026-09-18 — Dört yeni mimari fikri: gerçek anatomik devreler (mantar cisimciği, koku alma, görme hiyerarşisi) + CX-modüle arama

Kullanıcının "sineği avantajlı çıkarabilecek başka mimariler dene"
isteği üzerine, `Supplemental_file1_neuron_annotations.tsv`'nin
`cell_class` sütunundan gerçek, işlevi bilinen hücre popülasyonları
çıkarıldı (`data/processed/celltypes/`): Kenyon_Cell (5.177, mantar
cisimciği hafıza nöronları), MBON (96, çıkış), DAN (331, ödül sinyali),
olfactory (2.281) + ALPN (685, koku alma devresi), lamina/medulla/
lobula/lobula_plate (görme katmanları, LA→ME→LO→LOP gerçek işleme
sırası), central_complex (2.875, navigasyon/iç-durum bölgesi).

Genel afferent/efferent havuzları yerine bu **işlevi bilinen özel
devreler** encode/decode olarak kullanıldı — hâlâ zorunlu bağlantı
doğrulamasıyla (`select_connected_encode_decode`). Aynı çoklu-problem
Rastrigin görevi, 8 tohum.

**Mantar cisimciği (encode=Kenyon_Cell, decode=MBON):** p=0.383,
δ=0.219, FAIL. Ama hem gerçek hem null neredeyse tam taban değerde
(MSE≈0.49-0.55, "hiçbir şey öğrenmedim" seviyesi ≈0.5) — **sonuç
yorumlanamaz**, muhtemelen bu küçük havuzun (MBON sadece 96 nöron,
bağlantı doğrulamasından sonra daha da azaldı) bu görev için yetersiz
kapasitesi var. Ne pozitif ne negatif bir kanıt.

**CX-modüle arama (yön + keşif-kazancı ikisi de CX'ten okunuyor):**
**p=0.0078, δ=-0.375 — kapı testi GEÇTİ ama TERS yönde.** Gerçek
connectome 8 tohumun HEPSİNDE null'dan anlamlı şekilde kötü. Ama her
iki substrate de mutlak olarak zayıf (eval_fx 17-39 aralığı, gerçek
minimum 0'dan çok uzak) — yakınsama sorunu şüphesi var, ama yönün 8/8
tutarlı olması (rastgele gürültüyse böyle tutarlı olmazdı) bunun
gerçek bir etki olduğunu düşündürüyor. **Olası açıklama:** CX gerçek
beyinde çok özelleşmiş bir iş yapıyor (yön/navigasyon entegrasyonu) —
bunu alakasız bir "keşif kazancı" sinyaline zorlamak, gerçek CX'in
evrimleşmiş kısıtlarıyla çelişebilir; rastgele bir ağın "sahte CX"si
ise hiçbir özelleşme taşımadığı için bu role daha kolay
zorlanabiliyor olabilir. **İlk kez: anatomik gerçekçilik zarar
veriyor gibi görünen bir sonuç.**

**Antennal lobe (encode=olfactory, decode=ALPN):** p=0.383, δ=0.281 —
mantar cisimciğiyle aynı belirsiz örüntü. real=[0.519, 0.516, 0.490,
0.529, 0.550, 0.500, 0.504, 0.492] (medyan≈0.51), null=[0.525, 0.552,
0.501, 0.522, 0.524, 0.514, 0.495, 0.522] (medyan≈0.52) — ikisi de
neredeyse tam taban değerinde, anlamlı bir öğrenme yok. **İki anatomik
devrenin ikisi de (mantar cisimciği, koku alma) aynı "yakınsamıyor"
örüntüsünü gösteriyor** — bu artık tesadüf değil, sistemik bir
metodoloji sorununa işaret ediyor: küçük, özel anatomik havuzlar
(MBON=96, ALPN=685, bağlantı-doğrulamasından sonra daha da küçülüyor)
bu genel Rastrigin regresyon görevi için muhtemelen yetersiz kapasiteli
veya eğitim ayarları (300 epoch, sabit lr) bu daha küçük/farklı
yapıdaki alt-graflar için uygun değil. **Bu iki devre hakkında "işe
yaramıyor" ya da "işe yarıyor" diye bir hüküm veremeyiz** — pipeline
bu ölçekte onları test edecek güçte değil.

**Görme hiyerarşisi (encode=lamina, decode=lobula_plate — gerçek
LA→ME→LO→LOP işleme zincirinin baş ve son noktaları):** p=0.148,
**δ=0.438** — Cliff's delta eşiği (0.33) GEÇİYOR ama p-değeri anlamlılık
eşiğini (0.05) geçemiyor. En güçlü sinyal üç anatomik devre arasında,
ama istatistiksel güç (n=8) yetersiz kaldı.

**Dört mimarinin toplu tablosu:**

| Mimari | δ | p | Yön |
|---|---|---|---|
| Mantar cisimciği (hafıza) | +0.219 | 0.383 | zayıf pozitif |
| Koku alma (kemotaksi) | +0.281 | 0.383 | zayıf pozitif |
| Görme hiyerarşisi | **+0.438** | 0.148 | orta pozitif, eşik üstü ama p yetersiz |
| CX-modülasyon (iç-durum/navigasyon) | **-0.375** | **0.0078** | **anlamlı NEGATİF** |

**Ortaya çıkan örüntü (temkinli, ama tutarlı):** Üç **duyusal işleme**
devresinin üçü de (hafıza, koku, görme) pozitif yönde eğilim gösteriyor
— boyut arttıkça (mantar cisimciği < koku alma < görme hiyerarşisi,
kabaca havuz büyüklüğü sırasıyla da artıyor) etki büyüklüğü de artıyor.
Buna karşılık **CX'i kendi doğal işlevinin (yön/navigasyon
entegrasyonu) dışında, alakasız bir "iç durum modülasyonu" rolüne
zorlamak** net ve anlamlı şekilde ZARARLI çıktı. **Olası yorum:**
gerçek duyusal devreler (dış sinyali işleyip anlamlı çıktıya
dönüştürmek üzere evrimleşmiş) alakasız bir görevde bile hafif
avantaj taşıyabilir, ama çok özelleşmiş bir iç-durum devresini (CX)
görevine hiç uymayan bir role zorlamak dezavantaj yaratıyor. Bu,
"görev-mimari eşleşmesi" hikayesine yeni, tutarlı bir boyut ekliyor:
sadece "görev tipi" değil, **hangi alt-devrenin hangi role atandığı**
da belirleyici.

**Uyarı:** Hiçbir sonuç n=8'de p<0.05 VE |δ|>0.33'ü birlikte
sağlamadı (kapı testi resmi olarak hepsi FAIL) — bunlar hipotez
üretici, ön bulgular, kesin sonuçlar değil. Görme hiyerarşisi ve
CX-modülasyon özellikle daha büyük tohum sayısıyla (16-20) tekrar
denenmeye değer, ikisi de eşik büyüklüğünde etki gösteriyor.

## 2026-09-18 — Görme hiyerarşisi 16 tohuma çıkarıldı: ilk umut verici sinyal eridi

8 tohum daha eklendi (seed 8-15). Yeni 8 tohumda sonuç: p=0.945,
δ=-0.062 — ilk 8 tohumun δ=+0.438'inin neredeyse tam tersi.

**Birleşik sonuç (n=16):** real medyan=0.5057, null medyan=0.5119,
Wilcoxon p=0.323, **Cliff's δ=0.219**, gerçek connectome 16 tohumun
sadece 9'unda (%56, yazı-tura seviyesi) daha iyi. **Kapı testi FAIL.**

**HÜKÜM: İlk 8 tohumdaki umut verici sinyal (δ=0.438) büyük ölçüde
şanstı** — bu gece PSO'nun scene_anneal'ında (δ=0.312→0.133) ve
cartpole'da (δ=-0.219→0.133) gördüğümüz aynı örüntünün bir tekrarı:
küçük n'de "eşik üstü" görünen etkiler, daha fazla tohumla düzenli
olarak küçülüyor. **Bu artık bu oturumun genel bir metodolojik dersi
haline geldi: n=8'de "eşik geçti" görünen HİÇBİR sonuca, n=16-20'ye
çıkmadan güvenilmemeli.** Görme hiyerarşisi devresi de, mantar
cisimciği ve koku alma gibi, sonuçta "belirsiz/negatif yönde" kategori
sine giriyor.

## 2026-09-18 — CX-modülasyon 16 tohuma çıkarıldı: KUSURSUZ, GÜÇLÜ NEGATİF sonuç — bu gecenin en temiz ikinci bulgusu

8 tohum daha eklendi (seed 8-15). Yeni 8 tohumda sonuç DAHA DA
GÜÇLENDİ: p=0.0078, δ=-0.844 (ilk 8 tohumun δ=-0.375'inden çok daha
büyük). Görme hiyerarşisinin aksine (sinyal erimişti), CX-modülasyonun
sinyali **tohum sayısı arttıkça güçlendi.**

**Birleşik sonuç (n=16):**

real: [22.19, 31.78, 28.72, 32.39, 20.62, 39.08, 28.62, 30.03, 26.21,
31.18, 30.92, 30.33, 33.93, 30.21, 33.01, 30.46] (medyan 30.40)
null: [20.66, 26.59, 26.95, 29.70, 17.49, 35.86, 28.34, 28.19, 21.95,
26.80, 25.06, 27.82, 30.04, 28.29, 25.93, 27.79] (medyan 27.37)

**null, 16 tohumun HEPSİNDE gerçek connectome'dan iyi.** Wilcoxon
p=0.000031 (aşırı anlamlı), **Cliff's δ=-0.562** (eşiğin çok üstünde).
**Kapı testi KESİN OLARAK PASS — negatif yönde.**

**Bu, bu gecenin/bugünün malecns (δ=1.000) ve şev stabilitesi tek-atış
(δ=0.906) kadar temiz, üçüncü kesinleşmiş sonucu — ama TERS yönde.**
Görme hiyerarşisi/mantar cisimciği/koku alma'nın aksine (n arttıkça
sinyal eridi, muhtemelen şanstı), CX-modülasyonun sinyali n arttıkça
GÜÇLENDİ — bu, gerçek, tesadüf olmayan bir etkinin işareti.

**Yorum (temkinli ama artık güçlü kanıtla desteklenmiş):**
Merkezi kompleksi (CX) kendi evrimleşmiş işlevinin (yön/navigasyon
entegrasyonu) tamamen dışında, alakasız bir "keşif-kazancı" rolüne
zorlamak, **rastgele bir ağın aynı rolü üstlenmesinden belirgin şekilde
daha kötü.** Bu, "gerçek connectome her zaman nötr ya da iyi" gibi
basit bir varsayımı çürütüyor — **özelleşmiş bir devreyi kendi işlevine
hiç uymayan bir role zorlamak, rastgele bir ağdan daha kötü sonuç
verebiliyor.** Muhtemel açıklama: CX'in gerçek, evrimleşmiş iç
dinamikleri (kendi navigasyon/yön-entegrasyon işlevi için ince ayarlı)
alakasız bir skaler sinyali üretmek için "bükülmeye" rastgele bir ağdan
daha dirençli — rastgele ağın hiçbir özelleşmesi olmadığı için bu role
daha kolay zorlanabiliyor.

**Genel tabloya eklenen yeni katman:** Sadece "görev tipi" değil,
**"hangi anatomik alt-yapının hangi role atandığı"** da connectome'un
avantaj/dezavantaj yönünü belirliyor — gerçekten özelleşmiş bir devreyi
(CX) yanlış bir işe zorlamak, hiç özelleşmesi olmayan (null) bir ağı
aynı işe zorlamaktan daha kötü olabiliyor. Bu, PROTOCOL_V2'nin
spektrum tablosuna eklenmesi gereken üçüncü, bağımsız bir bulgu.

## 2026-09-18 — CX-modülasyon GERÇEK bir PSO sürüsünde tekrarlandı: sinyal tamamen kayboldu

Kullanıcının "neden PSO kullanmıyorsun" sorusu üzerine, CX-modülasyon
fikri bu kez GERÇEK bir PSO sürüsüne (12 parçacık, klasik atalet +
bilişsel + sosyal çekim, `fly_cx_pso_swarm_multiseed.py`) gömülerek
tekrar test edildi — önceki CX deneyi tek-sinek, sürüsüz bir arama
idi. Bu sefer okuma matrisleri EĞİTİLMEDİ (rastgele başlatıldı) —
saf, ham bir PSO+CX-nüdge testi.

**Erken sonuçlar çok çarpıcı görünüyordu** (seed=3,4: gerçek
connectome 0.04-0.05, gerçek minimuma çok yakın; null aynı tohumlarda
0.34-1.88) — ama bu, bu gece defalarca gördüğümüz "küçük örneklemde
şanslı görünme" tuzağının bir tekrarıydı.

**Tam 8 tohum sonucu: Wilcoxon p=0.844, Cliff's δ=0.000 — TAM SIFIR,
kapı testi FAIL.** real=[2.646, 0.966, 0.382, 0.047, 0.043, 0.806,
0.796, 0.739] (medyan 0.768), null=[0.646, 1.071, 0.155, 1.879, 0.344,
0.082, 1.277, 0.307] (medyan 0.495) — kazanan/kaybeden sayısı tam
eşit, hiçbir tutarlı yön yok.

**HÜKÜM: Tek-sinek CX-modülasyonunun güçlü negatif sonucu (δ=-0.562,
n=16, EĞİTİLMİŞ okuma matrisiyle) PSO sürüsüne VE eğitimsiz okuma
matrisine geçince tamamen kayboldu.** Bu iki değişkenden hangisinin
(sürü yapısı mı, eğitim mi) belirleyici olduğu ayrıştırılmadı — ikisi
aynı anda değişti. Muhtemelen **eğitim** asıl fark yaratan değişken
(tıpkı dün geceki PSO bulgularında — eğitilmemiş "raw" nüdge hiçbir
zaman klasik PSO'dan iyi çıkmamıştı, PSO'ya CX eklemek de aynı
kalıbı tekrarlıyor gibi görünüyor). **Sonraki adım (yapılmadı):**
Aynı PSO sürüsünde okuma matrislerini EĞİTEREK tekrar denemek —
ancak o zaman "sürü mü eğitim mi" sorusu ayrıştırılabilir.

---

## 2026-09-18 — KRİTİK BUG: eğitilmiş CX-PSO sürüsü NaN patlaması, real=null bit-bit aynı çıktı

**Faz:** 2 (H2 spektrum genişletme) — kullanıcının "alternatif PSO
mimarisi" isteği üzerine hem klasik-gbest sürüsünü eğitimli hale
getirdim (`fly_cx_pso_swarm_multiseed.py`, arka plan görev `bv8habb7q`)
hem de 3 yeni sürü topolojisi ekledim (`src/flyopt/variants/swarm_topologies.py`:
`chemotaxis_step` — sürü-arkadaşını "koku" gibi algılayan, gbest'siz
merkezi olmayan çekim; `lek_pull` — en iyi top-K bireyin EN YAKININA
çekilme, "dişi çekimi/lek" davranışından esinli çoklu-çekici model;
`ring_lbest` — Kennedy&Eberhart'ın literatür-standardı halka-topolojisi
lbest PSO'su) ve bunları `fly_swarm_topologies_multiseed.py` ile aynı
eğitimli-CX-nüdge mekanizmasına bağlayıp koştum (arka plan görev
`b5y4uu5de`).

**Beklenti:** 8 tohum, real_connectome vs er_null, 4 farklı topoloji
(gbest/chemotaxis/lek/ring_lbest) için gerçek bir istatistiksel fark ya
da dürüst bir "fark yok" sonucu bekleniyordu.

**Sonuç (koşuldu, ANCAK BUG BULUNDU):** Her iki koşum da real_connectome
ve er_null için **ONDALIK 15 BASAMAĞA KADAR BİT-BİT AYNI** `gbest_fx`
değerleri üretti (Wilcoxon p=1.0000, Cliff's delta=0.000 — tam sıfır,
4 topolojinin HEPSİNDE). Bu, PROTOCOL.md'nin kendi kuralı gereği
("sonuç şüpheli derecede tutarlı çıktığında da şüphelenmek gerekir")
derhal bug olarak işaretlendi ve araştırıldı — 8 farklı tohumda rastgele
inisyalizasyon ve eğitimin TESADÜFEN aynı sonuca yakınsaması istatistiksel
olarak imkansız.

**Kök neden (doğrulandı, `_debug_explode.py` ile):** `train_cx_modulated`
(cx_modulated_search.py) hiç gradyan kırpma (grad clipping) yapmıyordu.
Sürü deneylerinde `cfg.episode_steps=N_ITERS=60`, `T_inner=4` —yani
eğitim sırasında 60×4=240 adımlık bir tekrarlayan (recurrent) açılım
backprop ediliyor. 2000 düğümlü tam ölçekte bu, epoch 0'da zaten
`grad_norm=nan` üretiyor, epoch 1'de TÜM parametreler (`gain`, `bias`,
`leak_logit`, `readout_dir`) NaN'a düşüyor. Tek istisna:
`readout_gain`/`readout_gain_bias` — bunlar HİÇBİR ZAMAN gradyan almıyor
(kayıp fonksiyonu sadece `pred_dir`'i `target_t`'ye karşı denetliyor,
`gain`'in kaybı hiç yok — bkz. aşağıdaki ikinci bug), o yüzden Adam onları
atlıyor ve rastgele başlangıç değerlerinde donuk kalıyorlar.

Sürü koşumunda `gain*pred_dir` NaN olunca `x[i] = np.clip(x[i]+v[i]+NaN,
lo,hi)` da NaN oluyor, `fx=rastrigin(NaN)=NaN` oluyor, ve
`improved = fx < pbest_fx` NaN karşılaştırmasında HER ZAMAN False
döndüğü için `pbest_x`/`pbest_fx`/`gbest_fx` **İLK ADIMDA DONUYOR** ve
bir daha hiç güncellenmiyor. `gbest_fx` aslında sadece
`min(rastrigin(x0[i]) for i in range(12))` — yani PSO hiç çalışmamış,
sadece başlangıç rastgele örneklemesinin en iyisini raporluyormuş. Bu
başlangıç `x0` de `seed` ile belirlendiği ve real/null için AYNI seed
kullanıldığı için, iki koşum bit-bit aynı çıkıyor.

**İkinci bug (ayrı, daha hafif, düzeltilmedi — sadece belgelendi):**
`cx_modulated_search.py`'nin docstring'i "CX'ten gelen kazanç ucu-ucuna
öğrenilir" diyor ama `run_episode`'daki kayıp (`F.mse_loss(pred_dir,
target_t)`) hiçbir zaman `gain`'e bağlı değil — `step = pred_dir.detach()
.numpy() * gain.item()` hesaplama grafiğinden tamamen kopuk (numpy
alanında). Yani `readout_gain`/`readout_gain_bias` HİÇ eğitilmiyor,
sürekli rastgele başlangıç değerinde kalıyor. Bu, tek-ajanlı
`fly_cx_modulated_multiseed.py` sonucunu (δ=-0.562, 16/16 tohum, en
temiz bulgu) GEÇERSİZ KILMIYOR çünkü o pipeline'da da aynı mekanizma
var ve sonuç sonlu/gerçek sayılarla üretildi (NaN'a hiç düşmedi —
episode_steps=20 kullanıyor, 60 değil) — ama "CX gerçekten
öğreniyor" iddiası o deney için de teknik olarak yanlış: asıl ölçülen
şey, substrate-bağımlı `cx_h` durumunun SABİT rastgele bir doğrusal
izdüşümünün, öğrenilmiş `readout_dir` ile birlikte ne kadar
substrate-ayırt edici olduğu.

**Düzeltme:** `train_cx_modulated`'e `grad_clip=1.0` parametresi ve
`torch.nn.utils.clip_grad_norm_` eklendi (cartpole_control.py'deki aynı
düzeltme deseni). `episode_steps=20` konfigürasyonda bile ham gradyan
normu absürt derecede büyüktü (~1e13-1e19) — Adam'ın parametre-başına
normalizasyonu şimdiye kadar bunu NaN'a düşürmeden tolere etmiş
("şanslıyız" durumu), kırpma bunu artık şansa bırakmıyor.

**Etkilenen/geçersiz sonuçlar:** `results/fly_cx_pso_swarm_trained_*`
ve `results/fly_swarm_topologies_*` (ilk koşumlar) — GEÇERSİZ, bug
düzeltildikten sonra tekrar koşulacak. `results/fly_cx_pso_swarm_summary.json`
(ilk, EĞİTİMSİZ swarm, δ=0.000) — bu ayrı bir olgu, o koşumda hiç eğitim
yok, NaN sözkonusu değil, o sonuç geçerli kalıyor (ama "eğitimsiz nüdge
sinyali kaybediyor" yorumunun doğruluğu hâlâ açık, çünkü şimdi
görüyoruz ki eğitimli versiyon da bambaşka bir nedenle bozuktu).

**Sonraki adım:** Düzeltilmiş `train_cx_modulated` ile hem klasik-gbest
eğitimli sürüyü hem de 4-topoloji karşılaştırmasını yeniden koştur.

**GÜNCELLEME (2026-09-19 gece) — malecns degree-preserving null testi
BAŞLATILDI:** Kullanıcı onayıyla `C:/projeler/fly_demos/malecns`'e
`src/build_degree_preserving_null_graph.py` (flyopt'un kendi
`degree_preserving_rewire` algoritmasının edge-list formatına
uyarlanmış hali — Maslov-Sneppen çift-kenar takası, in/out-derece TAM
korunuyor, doğrulandı: `in-degree exactly preserved: True`,
`out-degree exactly preserved: True`, 8 sweep sonrası kenarların
%100'ü değişmiş) ve `multiseed_compare_degreenull.py` (orijinal
`multiseed_compare.py`'nin ayrı bir kopyası, orijinali bozmadan) eklendi.
GPU/CUDA native Windows'ta çalıştı (WSL2'ye gerek kalmadı — calibrate.py
sadece numpy/torch kullanıyor, GeNN bağımlılığı yok). Tek-tohum smoke
test başarılı (~570s/koşum, beklenen aralıkta). Tam 8-tohum koşum
arka planda başlatıldı (`multiseed_compare_degreenull.jsonl`),
tahmini süre ~2.5 saat (gerçekte ~4.7 saat sürdü, ~540-690s/koşum).

**SONUÇ (tamamlandı, ÇOK ÖNEMLİ — oturumun en ciddi bulgusu):**

```
flywire:              [0.142, 0.128, 0.124, 0.147, 0.137, 0.142, 0.156, 0.147]
degree_preserving_null:[0.140, 0.136, 0.146, 0.137, 0.135, 0.159, 0.156, 0.150]
flywire medyan=0.142   degree_null medyan=0.143
Wilcoxon p=0.375       Cliff's delta=-0.156
gate: FAIL
```

**Orijinal er_null sonucu (δ=1.000, MÜKEMMEL ayrım, 8/8 tohumda flywire
ezici üstündü) degree-preserving null'a karşı TAMAMEN KAYBOLDU —
hatta yön BİLE HAFİFÇE TERS DÖNDÜ** (δ=-0.156: null artık flywire'dan
biraz daha iyi, istatistiksel olarak anlamsız da olsa). flywire ile
degree-preserving-null'un ham sayıları neredeyse iç içe geçmiş durumda
(medyanlar 0.142 vs 0.143 — pratikte AYNI).

**Bu, CX-modulated bulgusuyla (δ=-0.562 er_null → δ=+0.125
degree-preserving, tamamen kayboldu) AYNI KALIBIN İKİNCİ VE ÇOK DAHA
ÇARPICI ÖRNEĞİ.** Malecns δ=1.000 bu oturumun (ve muhtemelen tüm
FlyOpt projesinin) en güçlü, en çok güvenilen H1-lehine bulgusuydu.
Bu sonuç, o bulgunun da büyük olasılıkla **FlyWire'ın gerçek biyolojik
ağların tipik özelliği olan heterojen/çok-kuyruklu derece dağılımından
kaynaklandığını, "hangi nöronun kime bağlı olduğu" bilgisinden
(gerçek bağlantı özgüllüğünden) DEĞİL** olduğunu güçlü biçimde
düşündürüyor. ER null derece dağılımını hiç korumadığı için bu farkı
yakalıyordu; derece-koruyan null aynı derece dağılımını taşıdığı için
fark ortadan kalkıyor.

**Şu ana kadarki genel tablo (2 test/2 kayboldu, 1 test/1 hayatta
kaldı):**
- CX-modulated (Rastrigin, tek-ajanlı): er_null δ=-0.562 → degree_preserving δ=+0.125 — **KAYBOLDU**
- malecns (F1 sürüş, gömülü kontrol): er_null δ=1.000 → degree_preserving δ=-0.156 — **KAYBOLDU**
- slope-stability (tek-atış regresyon): er_null δ=0.906 → degree_preserving δ=1.000 — **HAYATTA KALDI, GÜÇLENDİ**

**Bu, FlyOpt'un ana tezini (connectome'un ölçülebilir avantajı ÇOK
NADİR ve KIRILGAN) önemli ölçüde GÜÇLENDİRİYOR** — er_null'a karşı
görülen "pozitif" sonuçların çoğu muhtemelen derece-dağılımı
artefaktıydı, gerçek bağlantı-özgüllüğü DEĞİL. Şu ana kadar SADECE
şev-stabilitesi tek-atış görevi iki farklı null modele karşı da
sağlam kalabildi — bu, "hangi görev tipi gerçekten bağlantı-özgül
yapıdan faydalanıyor" sorusuna dair EN GÜVENİLİR ipucu.

**Sonraki adım (yüksek öncelik, sabah için):** Bu üç sonucu
PROTOCOL_V2_TASK_SPECTRUM.md'ye işle, ve düşün: belki de FlyOpt'un asıl
katkısı "connectome bazen yardımcı olur" değil, **"er_null tek başına
yeterli bir null model değil, her iddia mutlaka degree_preserving_rewire'a
karşı da doğrulanmalı"** şeklinde metodolojik bir ders haline geliyor.

---

## 2026-09-19 — malecns'in ÜÇÜNCÜ null modeli (weight_shuffle) de tamamlandı: hikaye netleşti

**Faz:** 2 (H2'' — sabaha kadar devam, tam paritenin tamamlanması).
Şev stabilitesi için 3 null modelin üçü de koşulmuştu; aynı üçlü test
malecns'e de uygulandı — `src/build_weight_shuffle_null_graph.py`
(flyopt'un `weight_shuffle`'ının edge-list uyarlaması: topoloji BİREBİR
korunuyor, ağırlık değerleri mevcut kenarlar arasında karıştırılıyor)
ve `multiseed_compare_weightshuffle.py` (8 tohum, ~2.5 saat GPU).

**Sonuç:**
```
flywire:             [0.119, 0.110, 0.110, 0.133, 0.124, 0.146, 0.147, 0.147]
weight_shuffle_null: [0.137, 0.128, 0.141, 0.142, 0.157, 0.142, 0.156, 0.118]
flywire medyan=0.1285   null medyan=0.1415
Wilcoxon p=0.148        Cliff's delta=-0.312
gate: FAIL
```

**malecns'in NİHAİ üç-null tablosu:**
| null modeli | ne bozuyor | δ | gate |
|---|---|---|---|
| er_null | her şeyi (derece + kimlik + ağırlık-topoloji ilişkisi) | **1.000** | **PASS (mükemmel)** |
| degree_preserving_rewire | sadece kimlik (kim-kime), dereceyi korur | -0.156 | FAIL (yön hafifçe ters) |
| weight_shuffle | sadece ağırlık-kenar eşleşmesi, topolojiyi (kim-kime) birebir korur | -0.312 | FAIL (yön hafifçe ters, degree_preserving ile TUTARLI) |

**Nihai yorum (projenin en önemli tek sonucu):** malecns'in δ=1.000
"mükemmel" bulgusu SADECE er_null'a karşı ayakta duruyor. İKİ farklı,
BİRBİRİNDEN BAĞIMSIZ ve farklı şeyleri bozan (biri kimlik, biri
ağırlık-kenar eşleşmesi) daha gerçekçi null modeline karşı da
KAYBOLUYOR, hatta ikisinde de yön hafifçe TERSİNE dönüyor (null
flywire'dan biraz daha iyi görünüyor, istatistiksel olarak anlamsız
da olsa). İki farklı null'un birbiriyle TUTARLI (ikisi de zayıf-negatif)
olması, bunun tesadüf değil, gerçek bir olgu olduğunu gösteriyor: **ER
null'un kendisi anormal derecede "kolay yenilebilir" bir taban çizgisi
— tamamen rastgele, düzensiz bağlantı hiçbir gerçekçi ağ istatistiğini
(ne derece dağılımı ne ağırlık-topoloji ilişkisi) taşımadığı için
işlevsiz bir dinamiğe yol açıyor, ve BUNUNLA karşılaştırıldığında HER
graf (gerçek ya da herhangi bir gerçekçi-null) iyi görünüyor.**

**SONUÇ: Bu oturumun ve muhtemelen tüm FlyOpt projesinin en güçlü
H1-lehine bulgusu (malecns, δ=1.000) artık güvenilir kabul
edilemez.** Projenin TEK sağlam kalan pozitif bulgusu, şev-stabilitesi
tek-atış toplu-regresyon görevi (2/3 null'a karşı güçlü PASS, 3.'de de
aynı yönde zayıf eğilim). **Bu, FlyOpt'un ana negatif tezini
(connectome'un ölçülebilir avantajı istisnai ve kırılgan) bu oturumun
başındaki tahminden çok daha güçlü biçimde destekliyor.**

---

## 2026-09-19 — MEKANİZMA BULUNDU: er_null'un neden bu kadar "kolay yenildiği" (derece dağılımı, resiprositi değil)

**Faz:** 2 (H2'' — kök neden analizi). Yukarıdaki bulguyu açıklamak
için `src/analyze_null_graphs.py` ile malecns'in 4 grafının (flywire,
er_null, degree_preserving, weight_shuffle) temel yapısal istatistikleri
karşılaştırıldı (GPU/eğitim gerektirmiyor, saniyeler sürdü).

**Bulgular:**
```
                  derece (std/mean)   izole-nöron oranı   resiprosite
flywire           1.67 / 1.92         %2.71 / %2.99       %15.65
er_null           0.16 / 0.16         %0.00 / %0.00        %0.02
degree_preserving 1.67 / 1.92         %2.71 / %2.99        %0.22
weight_shuffle    1.67 / 1.92         %2.71 / %2.99        %15.65
```

**Yorum (mekanizma artık net):**
- **Derece dağılımı**: FlyWire ÇOK heterojen (bazı nöronlar binlerce
  bağlantılı "hub", bazıları neredeyse izole — %2.7-3.0'ı sıfır
  dereceli). er_null bunu TAMAMEN yok ediyor (homojen/Poisson-benzeri
  dağılım, max derece 66-69 vs gerçek 6660-7570). degree_preserving VE
  weight_shuffle ikisi de bunu birebir koruyor.
- **Resiprosite (karşılıklı bağlantı)**: FlyWire'da kenarların
  **%15.65'i karşılıklı** (A→B varsa B→A da var) — gerçek biyolojik
  geri-besleme devrelerinin izi. er_null bunu neredeyse tamamen yok
  ediyor (%0.02). **ÖNEMLİ: degree_preserving_rewire da resiprositeyi
  neredeyse tamamen yok ediyor (%0.22)** — derece dağılımını koruma
  garantisi resiprositeyi KORUMUYOR (çift-kenar-takası algoritması
  karşılıklı çiftleri de karıştırıyor). **weight_shuffle ise
  resiprositeyi TAM koruyor (%15.65, flywire ile birebir aynı)** çünkü
  topolojiye hiç dokunmuyor.

**Bu, malecns bulgusunun kaynağını ayırt etmemizi sağlıyor:** Eğer
etkinin kaynağı RESİPROSİTE olsaydı, weight_shuffle (resiprositeyi
koruyan) flywire'a yakın/üstün performans göstermeliydi — ama
GÖSTERMEDİ (δ=-0.312, flywire'dan biraz KÖTÜ). Eğer etkinin kaynağı
DERECE DAĞILIMI olsaydı, hem degree_preserving hem weight_shuffle
(ikisi de heterojen dereceyi koruyor) flywire'a yakın performans
göstermeliydi — VE İKİSİ DE GÖSTERDİ (ikisi de "null flywire'dan
biraz iyi/eşit" bandında, δ=-0.156 ve -0.312). **Bu, er_null'un
malecns'te bu kadar "kolay yenilir" olmasının ASIL nedeninin heterojen
derece dağılımının (hub nöronların ve neredeyse-izole nöronların
varlığının) EKSİKLİĞİ olduğunu, resiprositenin (karşılıklı bağlantı)
İSE İKİNCİL/ilgisiz bir faktör olduğunu güçlü biçimde gösteriyor.**
Yani: er_null'un "kolay yenilirliği" muhtemelen basitçe şudur —
tamamen homojen/rastgele bağlı bir ağ, hub-nöronların yoğun
entegrasyon/yayılım rolünü oynayamıyor, bu da onu HER TÜR gerçekçi
(gerçek ya da null) heterojen-dereceli ağa karşı yapısal olarak
dezavantajlı kılıyor — connectome'un KENDİSİNE özgü bir avantaj değil.

**Doğrulama — AYNI mekanizma CX-modulated'in kendi alt-grafında da
var:** `scripts/analyze_null_model_structure.py` (kalıcı bir tanı
aracı olarak eklendi) ile fly_cx_modulated_multiseed.py'nin AYNI
2000-düğümlü alt-grafı (encode-seed=16000) test edildi:

```
                  derece (out/in std/mean)   resiprosite
real              0.825 / 0.881              %44.48
er_null           0.129 / 0.126               %2.95
degree_preserving 0.825 / 0.881               %7.14
weight_shuffle    0.825 / 0.881              %44.48
```

**Birebir aynı desen** (er_null hem dereceyi hem resiprositeyi yok
ediyor; degree_preserving dereceyi koruyup resiprositeyi yok ediyor;
weight_shuffle ikisini de koruyor) — malecns'teki mekanizma bulgusunu
tamamen doğruluyor ve genelliyor. Artık bu, tek bir görevin tuhaflığı
değil, **er_null'un FlyOpt'ta sistematik olarak kullanılan TÜM
alt-graf ölçeklerinde aynı şekilde "gerçekçi olmayan kolay" bir taban
çizgisi olduğu** anlamına geliyor.

**Kalıcı araç eklendi:** `scripts/analyze_null_model_structure.py` —
bundan sonra herhangi bir er_null karşılaştırması yapmadan önce, gerçek
alt-grafın derece heterojenliğinin er_null'dan ne kadar farklı olduğunu
hızlıca (saniyeler içinde, eğitim gerektirmeden) kontrol etmek için
kullanılabilir; büyük fark varsa sonuç mutlaka degree_preserving_rewire'a
karşı da doğrulanmalı.

---

## 2026-09-20 — RESMİ SONUÇ: şev stabilitesi PROTOCOL.md'nin n=30 kapısını GEÇTİ

**Faz:** 1 (resmi falsifikasyon kriteri, PROTOCOL.md §2, satır 103-107).
Kullanıcı "test et" dedi — tek ayakta kalan bulguyu (şev stabilitesi
tek-atış regresyonu) PROTOCOL.md'nin TAM ÖN-KAYITLI kriterine göre test
ettim: **30 tohum, degree_preserving_rewire (birincil null), Mann-Whitney
U testi**, p<0.05 VE |Cliff's delta|>0.33. Önceki koşumlar sadece 8
tohum ve Wilcoxon işaretli-sıra (eşleşmiş) kullanmıştı — bu, protokolün
gerçek harfine uygun İLK tam test.

`fly_slope_stability_degreenull_official30.py` ile 22 yeni tohum
(8-29) koşuldu, mevcut 0-7 ile birleştirildi (~4 saat, aynı alt-graf
encode-seed=9000, aynı 300-epoch eğitim).

**SONUÇ:**
```
real medyan = 0.1173   null (degree_preserving) medyan = 0.1731
Mann-Whitney U (ÖN-KAYITLI RESMİ TEST): p = 0.000000
Wilcoxon işaretli-sıra (oturum konvansiyonu): p = 0.000001
Cliff's delta = 0.884
RESMİ KAPI: PASS
```

**Bu, projenin PROTOCOL.md'nin TAM harfine uygun (30 tohum, doğru
istatistiksel test, doğru null model) GEÇEN İLK VE TEK bulgusu.**
Hem Mann-Whitney (eşleşmemiş, resmi) hem Wilcoxon (eşleşmiş, oturum
konvansiyonu) neredeyse aynı sonucu veriyor (ikisi de p<0.00001) —
istatistiksel test seçiminin sonucu değiştirmediği, bulgunun sağlam
olduğu anlamına geliyor. Cliff's delta=0.884, "büyük etki" eşiğinin
(0.474) bile üzerinde.

**Önemli sınırlama (dürüstlük notu):** Bu SADECE degree_preserving_rewire
null'una karşı n=30'a çıkarıldı. weight_shuffle karşılaştırması hâlâ
n=16'da (δ=0.305, FAIL ama aynı yönde) — o da isterse n=30'a
çıkarılabilir, ama PROTOCOL.md'nin resmi kriteri zaten SADECE
degree_preserving_rewire'ı şart koşuyor, weight_shuffle ek bir
sağlamlık kontrolü. **Bu sonuç, şev-stabilitesi tek-atış toplu-regresyon
görev ailesi için H1'in resmi olarak DOĞRULANDIĞI anlamına geliyor** —
FlyOpt'un kurduğu günden beri aradığı, protokolün kendi kurallarına göre
geçerli ilk pozitif sonuç.

**GÜNCELLEME — weight_shuffle de n=30'a çıkarıldı (tam parite için,
resmi kriterin bir parçası değil ama tamlık için):**
`fly_slope_stability_weightshuffle_official30.py` ile 14 yeni tohum
(16-29) koşuldu, mevcut n=16 ile birleştirildi.

```
Mann-Whitney U: p=0.0877   Wilcoxon: p=0.0667   Cliff's delta=0.258
gate: FAIL
```

**Trend net biçimde ZAYIFLIYOR tohum sayısı arttıkça:** n=8'de
δ=0.375 → n=16'da δ=0.305 → n=30'da δ=0.258. Bu, oturumda defalarca
görülen "küçük-n'de umut verici, büyük-n'de sönen" kalıbın BİR ÖRNEĞİ
DAHA — weight_shuffle'daki sinyalin muhtemelen gerçek bir etki değil,
gürültü olduğunu gösteriyor (degree_preserving'in aksine, ki o n=8'den
n=30'a çıkarken GÜÇLENDİ: 1.000'den 0.884'e sadece çok hafif düştü,
hâlâ ezici biçimde anlamlı).

**Nihai, tam-parite üç-null tablosu (şev stabilitesi, tek-atış, HEPSİ n=30 veya n=16+):**
| null modeli | n | δ | p (Mann-Whitney) | gate |
|---|---|---|---|---|
| er_null | 8 | 0.906 | (Wilcoxon kullanıldı, MW koşulmadı) | PASS (informal) |
| **degree_preserving_rewire** | **30** | **0.884** | **0.000000** | **RESMİ PASS** |
| weight_shuffle | 30 | 0.258 | 0.0877 | FAIL (zayıflayan trend) |

**Nihai yorum:** Etki, PROTOCOL.md'nin talep ettiği asıl testte
(degree_preserving_rewire, n=30) ezici biçimde doğrulandı. weight_shuffle'daki
zayıf/sönen trend, bunun ağırlık değerlerinden değil TOPOLOJİDEN
kaynaklandığı yorumunu daha da güçlendiriyor — ama bu artık sadece bir
yorum notu, resmi sonucu değiştirmiyor. **FlyOpt'un resmi, protokole
uygun ilk pozitif bulgusu kesinleşti: şev stabilitesi tek-atış toplu
regresyonu.**

---

## 2026-09-20 — Şaşırtıcı düzeltme: sentetik "hub-baskın" graf da yeniliyor — etki sadece "heterojen derece" değil

**Faz:** 2 (H2'' mekanizma derinleştirme). Kullanıcı "sentetik hub-baskın
grafla da test et" dedi — önceki yorumumu (malecns/CX-modulated
analizinden: "etki muhtemelen sadece heterojen/çok-kuyruklu derece
dağılımı olmasından geliyor, fly'a özgü değil") doğrudan sınamak için.

`src/flyopt/substrates/graph_builders.py`'ye `scale_free_null` eklendi:
gerçek fly'ın derece dizisini KULLANMIYOR (degree_preserving_rewire'ın
aksine) — sıfırdan, bağımsız bir Pareto (güç-yasası) dağılımından YENİ
bir derece dizisi örnekliyor, sadece düğüm sayısı ve toplam kenar
sayısı gerçek alt-grafla eşleşiyor. Ağırlıklar gerçek dağılımdan
örnekleniyor (weight_shuffle'ın konvansiyonuyla tutarlı). Encode/decode
nöronları da bu YENİ grafın kendi bağlantı yapısında ayrıca doğrulandı
(gerçek graftan miras alınmadı — adil olması için).

**Beklenti:** Eğer etki "sadece heterojen derece dağılımı" olsaydı,
gerçek connectome ile bu sentetik hub-baskın graf arasında BÜYÜK fark
OLMAMASINI bekliyordum (degree_preserving_rewire'ın gerçek dereceyle
kazandığı gibi, herhangi bir heterojen graf da kazanmalıydı).

**Sonuç (ÇARPICI, beklentimin TERSİ):**
```
real medyan = 0.1155   sentetik-hub-baskın medyan = 0.1488
Wilcoxon p = 0.00781    Mann-Whitney p = 0.00295
Cliff's delta = 0.844
gate: PASS (n=8)
```

**Gerçek connectome, kendi türünden ama fly'a hiç özgü olmayan bir
hub-baskın graf karşısında da AÇIK ARAYLA kazanıyor.** Bu, önceki
yorumumu DÜZELTİYOR: etki "herhangi bir heterojen-dereceli ağ işe
yarar" değilmiş — **fly'ın KENDİ ÖZGÜL derece dizisi** (ya da derece
dizisinin ötesinde bir şey) önemliymiş.

**Üç null testinin BİRLİKTE anlattığı hikaye artık çok daha net:**
| null modeli | neyi koruyor | neyi bozuyor | sonuç |
|---|---|---|---|
| degree_preserving_rewire | fly'ın TAM derece dizisi | kim-kime-bağlı | gerçek KAZANIYOR (δ=0.884, n=30) |
| weight_shuffle | kim-kime-bağlı (birebir) | ağırlık değerleri | gerçek ZAR ZOR kazanıyor (δ=0.258, n=30, FAIL) |
| scale_free_null (YENİ) | sadece düğüm/kenar SAYISI | fly'ın derece dizisinin KENDİSİ | gerçek AÇIKÇA kazanıyor (δ=0.844, n=8) |

**Sonuç: yük taşıyan özellik "kim kime bağlı" değil (degree_preserving
bunu bozup kazanıyor), "ağırlık değerleri" de değil (weight_shuffle
bunu koruyup zar zor kazanıyor) — YÜK TAŞIYAN ÖZELLİK, FLY'IN KENDİ
SPESİFİK DERECE DİZİSİ (hangi nöronun TAM OLARAK kaç bağlantısı
olduğu, hangi biyolojik-evrimsel süreçle şekillendiği).** Bu, "herhangi
bir hub-baskın ağ işe yarar" (daha zayıf, ağ-teorisi-genel bir iddia)
tezinden çok daha güçlü ve daha ilginç bir sonuç: **fly'ın connectome'unun
DERECE DİZİSİ, jenerik bir güç-yasası örneklemesinden bile ayırt
edilebilir şekilde daha "iyi" bir yapı taşıyor** — muhtemelen biyolojik
evrimin şekillendirdiği, rastgele bir matematiksel modelin
yakalayamadığı bir düzenlilik (ör. giriş/çıkış derecesi arasında
korelasyon, belirli bir kuyruk şekli, ya da modüler/küme yapısı).

**Not (dürüstlük, güç sınırlaması):** Bu sadece n=8, resmi 30-tohum
kriterine henüz çıkarılmadı. Ayrıca TEK bir Pareto parametresi (shape=2.3)
denendi — farklı güç-yasası üstelleri veya farklı sentetik model
aileleri (ör. Barabási-Albert büyüme modeli, ya da giriş/çıkış derecesi
korele edilmiş bir sentetik model) denenirse sonuç değişebilir. Yine de
yön net: "herhangi bir heterojen graf yeter" hipotezi bu haliyle
YANLIŞLANDI, "fly'ın kendi derece dizisi özel" hipotezi güçlendi.

**GÜNCELLEME — n=30'a çıkarıldı, sonuç DOĞRULANDI:**
`fly_slope_stability_scalefree_official30.py` ile 22 yeni tohum (8-29)
koşuldu, mevcut n=8 ile birleştirildi.

```
real medyan = 0.1173   sentetik-hub-baskın medyan = 0.1531
Mann-Whitney U: p = 0.000001
Wilcoxon işaretli-sıra: p = 0.000003
Cliff's delta = 0.738
gate: PASS
```

n=8'deki (δ=0.844) sonuçtan biraz düştü ama HÂLÂ çok güçlü ve son
derece anlamlı (p<0.00001) — bu, oturumda tekrar tekrar görülen
"küçük-n'de umut → büyük-n'de sönme" kalıbının TERSİ: sinyal n=30'da
GÜÇLÜ KALDI, sadece biraz küçüldü (beklenen, normal bir regresyon-to-mean
etkisi, kaybolma değil).

**FlyOpt'un artık İKİ tane, birbirinden bağımsız, n=30'da resmi olarak
doğrulanmış pozitif bulgusu var** (aynı görev ailesinde, iki farklı
null'a karşı):
1. degree_preserving_rewire'a karşı: δ=0.884, p≈0.000000
2. scale_free_null'a karşı (fly'a özgü olmayan sentetik hub-baskın graf): δ=0.738, p=0.000001

**Bu, "fly'ın kendi spesifik derece dizisi özel" hipotezini artık sadece
n=8 değil, tam protokol gücünde (n=30) destekliyor.** Şev stabilitesi
tek-atış toplu-regresyon görev ailesi için H1, hem "her şeyi bozan"
hem "sadece fly'ın derece-dizisi kimliğini bozan" iki farklı sıkı teste
karşı sağlam duruyor.

---

## 2026-09-20 — Mekanizma netleşti: giriş/çıkış derecesi KORELASYONU eksik özellikmiş

**Faz:** 2 (H2'' mekanizma derinleştirme, devam). Sentetik scale_free_null'un
neden kaybettiğini anlamak için ek bir yapısal analiz yapıldı — sentetik
modelde çıkış-derecesi ve giriş-derecesi BİRBİRİNDEN BAĞIMSIZ
örneklenmişti (iki ayrı Pareto çekilimi). Gerçek grafta bu böyle mi diye
bakıldı (Spearman korelasyonu, aynı 3000-düğümlü alt-graf).

**Sonuç (net ve çarpıcı):**
```
                  giriş/çıkış-derecesi Spearman r    p
real              0.7035                              ~0
degree_preserving 0.7035 (BİREBİR AYNI)                ~0
scale_free_null   -0.0333 (pratikte SIFIR)             0.068 (anlamsız)

kenar-düzeyinde asortatiflik (kaynak-çıkış-derecesi vs hedef-giriş-derecesi):
real:             r=0.1116 (küçük ama çok anlamlı)
scale_free_null:  r=-0.0112 (pratikte sıfır)
```

**Yorum:** Gerçek sinek beyninde bir nöronun giriş derecesi ile çıkış
derecesi GÜÇLÜ biçimde korelasyonlu (r=0.70) — yani çok girdi alan
nöronlar (entegrasyon merkezleri) genelde çok da çıktı veriyor (yayılım
merkezleri de aynı zamanda) — biyolojik olarak mantıklı bir "entegrasyon-yayılım
hub'ı" özelliği. `degree_preserving_rewire` bu korelasyonu OTOMATİK
KORUYOR çünkü her düğümün kendi (giriş, çıkış) derece ÇİFTİNİ aynen
taşıyor, sadece kimin-kime-bağlı-olduğunu karıştırıyor — bu YÜZDEN
kazanıyor. `scale_free_null` ise giriş ve çıkış derecelerini BAĞIMSIZ
örneklediği için bu korelasyonu TAMAMEN yok ediyor (r≈0) — bu YÜZDEN
kaybediyor.

**Bu, projenin en net, en somut mekanistik bulgusu:** Fly connectome'unun
"özel" olan yönü ne kimin-kime-bağlı-olduğu (degree_preserving bunu
bozup kazanıyor) ne ağırlık değerleri (weight_shuffle bunu koruyup zar
zor kazanıyor) — **her nöronun giriş ve çıkış derecesinin BİRLİKTE
yüksek ya da BİRLİKTE düşük olması** (entegrasyon-yayılım kuplajı).
Sentetik bir güç-yasası modeli bu korelasyonu doğal olarak üretmiyor
(bağımsız örnekleme varsayımı altında), gerçek biyolojik ağ ise bunu
güçlü biçimde taşıyor. **Sonraki adım (yapılmadı):** giriş/çıkış
derecesi KORELE EDİLEREK üretilen bir sentetik null (ör. copula ile
r=0.70 hedefleyen bir örnekleme) tasarlanıp test edilirse, bu hipotez
daha da kesinleştirilebilir — eğer O sentetik model de kazanırsa,
"özel olan şey SADECE bu korelasyon" kanıtlanmış olur.

**GÜNCELLEME — hipotez test edildi ve YANLIŞLANDI (dürüstçe kaydediliyor):**
`scale_free_correlated_null` eklendi (`graph_builders.py`) — paylaşılan
bir "hub yoğunluğu" değerinden hem giriş hem çıkış derecesi türetiliyor,
gerçek grafın giriş/çıkış korelasyonuna YAKIN bir değer hedefleniyor.
Doğrulandı: elde edilen korelasyon r=0.6659 (gerçeğin r=0.7035'ine çok
yakın). `fly_slope_stability_scalefreecorr_multiseed.py` ile n=8 test
edildi.

**Sonuç (beklentinin aksine):**
```
real medyan = 0.1155   korele-sentetik medyan = 0.1499
Wilcoxon p = 0.0391     Mann-Whitney p = 0.0148
Cliff's delta = 0.719
gate: PASS (gerçek hâlâ kazanıyor)
```

**Korelasyonu düzeltmek boşluğu KAPATMADI** — δ=0.719, bağımsız
modelin (δ=0.844 n=8 / 0.738 n=30) sonucuyla hemen hemen AYNI
büyüklükte, degree_preserving_rewire'ın (δ=0.884-1.000) çok gerisinde.
**Yani "giriş/çıkış derecesi korelasyonu" hipotezi YANLIŞLANDI (ya da
en azından tek başına yeterli değil).** Fly'ın gerçek derece dizisinde,
iki parametreyle (heterojenlik + korelasyon) yakalanamayan BAŞKA bir
düzenlilik olmalı — muhtemelen derece dağılımının tam şekli (kuyruk
eğrisi, hub'ların birbirine göre göreli büyüklükleri, ya da modüler/kümesel
yapı gibi daha yüksek mertebeden bir özellik). Şu ana kadar sadece
`degree_preserving_rewire` (fly'ın TAM gerçek derece dizisini birebir
kopyalayan) gerçek performansı yakalayabiliyor — hiçbir BASİT jenerik
üretim süreci (bağımsız ya da korele Pareto) bunu henüz taklit edemedi.

**Dürüstlük notu:** Bu, projenin "sürpriz derecede iyi/kötü sonuçlara
şüpheyle yaklaş" kuralının olumlu bir uygulaması — cazip bir mekanistik
hikaye (giriş/çıkış korelasyonu) kurup dogrudan test ettik, tutmadı, ve
bunu gizlemek yerine kaydediyoruz. "Fly'ın derece dizisi özel" sonucu
hâlâ sağlam duruyor (iki n=30 testinde de kanıtlandı) — ama NEDEN özel
olduğu sorusu hâlâ açık.

---

## 2026-09-20/21 — Üçüncü mekanizma denemesi: MODÜLERLİK/KÜMESEL YAPI — bu kez umut verici

**Faz:** 2 (H2'' mekanizma derinleştirme). Kullanıcı "kümesel/modüler
yapıyı da dene" dedi. Önce ucuz bir yapısal ölçüm yapıldı (eğitim
gerektirmeden, `networkx.algorithms.community.louvain_communities` ile):

```
                       modülerlik (Q)   ort. kümeleme katsayısı
real                   0.3565           0.2759
degree_preserving      0.0687           0.1027
scale_free_null        0.0787           0.0622
scale_free_correlated  0.0760           0.0940
```

**Çarpıcı gözlem: ÜÇ null modelin ÜÇÜ DE gerçek grafın modülerliğini
neredeyse tamamen yok ediyor** (hepsi Q≈0.07-0.08, gerçeğin 0.36'sının
çok altında) — `degree_preserving_rewire` bile (ki bu en güçlü null,
δ=0.884) modülerliği tamamen yok ediyor, çünkü çift-kenar-takası
algoritması sadece dereceyi koruyor, hangi kümenin hangi kümeye
bağlandığını hiç umursamıyor.

**Yeni null: `community_preserving_rewire`** (`graph_builders.py`) —
hem TAM derece dizisini hem de gerçek Louvain-kümelerinin
kümeler-arası kenar sayısı matrisini (yani modülerliği) koruyor. Sadece
AYNI (kaynak-kümesi, hedef-kümesi) çiftine sahip kenarlar arasında
takas yapılıyor. Doğrulandı: elde edilen modülerlik=0.3648 (gerçeğin
0.3565'ine neredeyse birebir), derece dizisi ~aynı (3 kenar kaybı,
ihmal edilebilir).

**Sonuç (n=8, ÇOK UMUT VERİCİ):**
```
real medyan = 0.1155   community-preserving medyan = 0.1249
Wilcoxon p = 0.148      Mann-Whitney p = 0.235
Cliff's delta = 0.375
gate: FAIL (ama n=8'de eşik üstü, önceki 3 null'dan ÇOK daha zayıf bir fark)
```

**Karşılaştırma — dört null'un özet tablosu (hepsi n=8'de, δ büyüklüğüne göre sıralı):**
| null | ne koruyor | δ (n=8) |
|---|---|---|
| degree_preserving | sadece TAM derece dizisi | 0.844 (n=30'da 0.884) |
| scale_free (bağımsız) | sadece heterojenlik (fly'a özgü olmayan) | 0.844 (n=30'da 0.738) |
| scale_free_correlated | heterojenlik + giriş/çıkış korelasyonu | 0.719 |
| **community_preserving (YENİ)** | **TAM derece dizisi + TAM modülerlik** | **0.375 (ÇOK DAHA ZAYIF)** |

**Modülerliği eklemek, boşluğu diğer tüm denemelerden çok daha fazla
küçülttü** — δ 0.72-0.88 aralığından 0.375'e düştü, weight_shuffle'ın
"neredeyse fark yok" bölgesine (δ=0.258-0.305) yaklaşıyor. Bu, modülerliğin
gerçekten yük taşıyan özelliklerden biri (belki en büyüğü) olduğuna
dair GÜÇLÜ bir ilk kanıt. n=8'de umut verici ama istatistiksel güç
yetersiz — session kuralı gereği n=30'a çıkarılıyor.

**Ek, zayıf sinyal (motif analizi):** Kullanıcının Gemini ile
yaptığı beyin fırtınasından bir öneri (`literature/gemini_arastirma_tavsiyeleri.md`
§2.3) üzerine triad census (yönlü graf motifleri, Milo ve ark. 2002
standardı) da hızlıca denendi — 500 düğümlük tek bir alt-örneklemde,
tek çekilim. Feedforward-loop (030T) oranı community_preserving≈real
(0.00024/0.00023) > degree_preserving (0.00017) > scale_free (0.00013)
— yön ilginç (community_preserving en yakın) ama fark çok küçük ve
TEK örneklemden, güvenilir bir sonuç değil. Modülerlik bulgusunun
aksine bu, resmi olarak takip edilmeyecek kadar zayıf — sadece not
düşülüyor.

---

## 2026-09-19 — Üçüncü sağlamlık kontrolü: şev stabilitesi weight_shuffle null'una karşı

**Faz:** 2 (H2'' — slope-stability artık projenin TEK ayakta kalan
pozitif bulgusu, en yüksek önceliğe layık). `weight_shuffle` (aynı
TOPOLOJI/kenar kümesi birebir korunuyor, sadece ağırlık DEĞERLERİ
mevcut kenarlar arasında karıştırılıyor — degree_preserving_rewire'ın
tam tersi yönde bir kontrol: o "kim kime bağlı"yı bozup dereceyi
koruyordu, bu "ne kadar güçlü bağlı"yı bozup gerçek kimin-kime-bağlı
olduğunu koruyor) ile test edildi (`fly_slope_stability_weightshuffle_multiseed.py`,
aynı alt-graf/encode-seed=9000, aynı eğitim).

**Sonuç (n=8):** real medyan=0.1155, null medyan=0.1279, Wilcoxon
p=0.250, **Cliff's delta=0.375** — eşiği (|δ|>0.33) GEÇİYOR ama p
yetersiz — "n=8'de umut verici ama güçsüz" kalıbı, session kuralı
gereği n=16'ya çıkarıldı (`fly_slope_stability_weightshuffle_extra.py`,
arka planda, tahmini ~1-1.5 saat).

**Şu ana kadarki 3 null modelin özeti (şev stabilitesi, tek-atış):**
- er_null: δ=0.906, PASS
- degree_preserving_rewire: δ=1.000, PASS (MÜKEMMEL)
- weight_shuffle: δ=0.375 (n=8, eşik üstü ama p yetersiz) — n=16 bekleniyor

Üç null modelin ÜÇÜ DE aynı yönde (real her zaman null'dan iyi) —
bu, malecns/CX-modulated'in aksine, bu bulgunun GERÇEK bağlantı
özgüllüğünden kaynaklandığına dair en güçlü kanıt. n=16 sonucu
bekleniyor.

**GÜNCELLEME — n=16 sonucu (nihai):** real medyan=0.1203, null
medyan=0.1291, Wilcoxon p=0.211, **Cliff's delta=0.305** (n=8'deki
0.375'ten biraz düştü, eşiğin (0.33) az altına indi) — gate FAIL, ama
**YÖN AYNI KALDI** (real her zaman biraz daha iyi) — malecns/CX-modulated'deki
gibi TAM bir çöküş YOK, sadece anlamlılığa ulaşamayan zayıf-ama-tutarlı
bir eğilim.

**Nihai üç-null tablosu (şev stabilitesi, tek-atış):**
| null modeli | ne bozuyor | δ | n | gate |
|---|---|---|---|---|
| er_null | her şeyi (derece + kimlik) | 0.906 | 8 | PASS |
| degree_preserving_rewire | sadece kimlik (kim-kime), dereceyi korur | 1.000 | 8 | PASS (mükemmel) |
| weight_shuffle | sadece ağırlık DEĞERİ, topolojiyi (kim-kime) birebir korur | 0.305 | 16 | FAIL (ama aynı yönde) |

**Yorum (temiz, tutarlı bir bilimsel resim):** Topolojiyi bozan İKİ
null modelde de (er_null, degree_preserving) GÜÇLÜ ve tutarlı bir etki
var. Topolojiyi AYNEN koruyup SADECE ağırlık değerlerini karıştıran
weight_shuffle'da etki zayıflıyor (ama tersine dönmüyor). **Bu, etkinin
kaynağının spesifik sinaptik ağırlık büyüklükleri değil, gerçek
BAĞLANTI TOPOLOJİSİNİN KENDİSİ (kim kime bağlı) olduğunu güçlü biçimde
düşündürüyor** — FlyWire'ın gerçek "kablo şeması"nın şev-stabilitesi
gibi uzamsal, tek-atış toplu-regresyon görevlerinde ölçülebilir bir
katkısı olduğuna dair, bu oturumun ürettiği EN SAĞLAM, EN İYİ
SAVUNULABİLİR bulgu.

**GÜNCELLEME — asıl düzeltme ve gerçek sonuçlar:** `grad_clip=1.0`
eklemek TEK BAŞINA yetmedi — `_verify_fix.py` ile doğrulandı: epoch 0'da
zaten `grad_norm=nan` çıkıyor, yani NaN backward-pass'in İÇİNDE
(240 adımlık BPTT zincirinin gradyan hesaplamasında) oluşuyor, kırpma
sadece SONLU bir gradyan tensörünü ölçekler, zaten NaN olanı düzeltemez.
Asıl düzeltme: eğitim ve sürü-koşumu için AYRI `episode_steps` kullanmak.
Eğitim artık `episode_steps=20` (tek-ajanlı `fly_cx_modulated_multiseed.py`
ile aynı, kanıtlanmış stabil ufuk) kullanıyor; sürünün kendi 60 adımlık
koşumu zaten `torch.no_grad()` altında çalıştığı için ufuk uzunluğundan
hiç etkilenmiyor. 40 epoch'ta full-scale (2000 düğüm) doğrulama: NaN yok.
Her iki script'e de bu ayrım uygulandı (`train_cfg` ayrı, `cfg` ayrı).

**Düzeltilmiş sonuçlar (8 tohum, artık gerçek/farklı sayılar,
bit-bit-aynı bug'ı yok):**

- `fly_cx_pso_swarm_multiseed.py` (klasik gbest PSO + eğitimli CX-nüdge,
  encode/CX seçim seed=17000): real medyan=0.599, null medyan=0.230,
  Wilcoxon p=0.3125, **Cliff's delta=-0.406** — gate FAIL (p yetersiz,
  n=8'de güçsüz) ama yön ve büyüklük tek-ajanlı bulgu (δ=-0.562) ile
  AYNI YÖNDE.
- `fly_swarm_topologies_multiseed.py` (4 topoloji, aynı eğitimli
  CX-nüdge, encode/CX seçim seed=18000 — farklı bir alt-graf, bağımsız
  çekilim):
  - gbest: p=0.3828, delta=-0.375 (FAIL, ama yine |delta|>0.33 ve yön
    null-lehine — yukarıdaki bağımsız gbest koşumuyla TUTARLI)
  - chemotaxis (gbest'siz, merkezi olmayan "koku" çekimi): p=0.9453,
    delta=+0.250 (FAIL, zayıf, ters yönde)
  - lek (top-K en yakın çekici): p=0.6406, delta=-0.156 (FAIL, zayıf)
  - ring_lbest (literatür-standart halka topolojisi): p=0.9453,
    delta=-0.062 (FAIL, neredeyse sıfır)

**Yorum:** Klasik gbest-PSO + eğitimli CX-nüdge konfigürasyonu İKİ
BAĞIMSIZ koşumda da (farklı alt-graf seçim tohumu ile) |delta|>0.33 ve
aynı yönde (null lehine) çıktı — bu, "n=8'de eşik-geçen ama düşük-güç"
durumu, PROTOCOL.md kuralı gereği 16 tohuma çıkarılmayı hak ediyor.
Alternatif topolojiler (chemotaxis/lek/ring_lbest) şimdilik net bir
sinyal göstermiyor — merkezi-olmayan/çoklu-çekici mekanizmalar bu
görevde gbest'in (zayıf da olsa) tutarlılığını yakalayamadı.

**Sonraki adım:** gbest-PSO+eğitimli-CX-nüdge sonucunu 16 tohuma çıkar.

**GÜNCELLEME — 16 tohum sonucu:** `fly_cx_pso_swarm_extra.py` ile
seed 8-15 koşuldu ve orijinal 8 tohumla birleştirildi. Sonuç:
real medyan=0.263, null medyan=0.286, Wilcoxon p=0.940,
**Cliff's delta=-0.008** (n=8'deki -0.406'dan neredeyse SIFIRA
düştü) — gate FAIL, net biçimde. Bu, bu oturumda tekrar tekrar
görülen "n=8'de umut verici → n=16'da sönüyor" kalıbının BİR
ÖRNEĞİ DAHA (PSO scene_anneal, cartpole, visual_hierarchy'de de
aynısı olmuştu). Tek-ajanlı CX-modulated bulgusu (δ=-0.562,
16/16 tohum, GÜÇLENEREK) hâlâ ayakta ve hâlâ oturumun en temiz
bulgusu — ama bu bulgu bir PSO sürüsüne gömülünce KAYBOLUYOR, iki
kez ayrı ayrı doğrulandı (n=8'de bağımsız alt-graf tutarlılığı bir
tesadüf değildi ama n=16'da gerçek sinyal olmadığı da netleşti;
muhtemelen sürünün PSO-kaynaklı gürültüsü CX'in ince ayarını
tamamen boğuyor). Alternatif topolojiler (chemotaxis/lek/ring_lbest)
zaten n=8'de bile sinyal göstermemişti, n=16'ya çıkarmaya gerek yok.

**Bu deney ailesinin nihai sonucu:** CX-modulated fly-nüdge mekanizması
SADECE tek-ajanlı, sürüsüz kullanımda ölçülebilir bir connectome
avantajı gösteriyor. Herhangi bir PSO-sürü çerçevesine (klasik gbest
veya alternatif topolojiler) gömüldüğünde bu avantaj kayboluyor.

---

## 2026-09-18 — İkinci bug'ın düzeltmesi: `readout_gain` artık GERÇEKTEN eğitiliyor (uçtan uca)

**Faz:** 2 (H2'' — belgelenen kısıtı gider). Yukarıda belgelenen ikinci
bug (`readout_gain`/`readout_gain_bias` hiç gradyan almıyordu, kayıp
sadece `pred_dir`'i denetliyordu) düzeltildi: `cx_modulated_search.py`'ye
`run_episode_e2e`/`train_cx_modulated_e2e`/`evaluate_cx_modulated_e2e`
eklendi — `x` artık tüm bölüm (episode) boyunca torch tensörü olarak
kalıyor, kayıp gerçek görev hedefi (bölüm boyunca ortalama Rastrigin
değeri), böylece `gain`'in adım büyüklüğü üzerindeki etkisi nihayet
`readout_gain`'e geri gradyan taşıyor. Duyusal ışın-atma (ray-casting)
hâlâ ayrık numpy'da (meşru, geriye-yayılmayan bir "gözlem" adımı).

**Beklenti:** Uçtan-uca eğitilmiş gain, mimarinin "CX gerçekten
öğreniyor" iddiasını doğru kılıyor — soru, bunun real-vs-null tablosunu
DEĞİŞTİRİP değiştirmeyeceği (n=8, aynı encode-seed=16000, aynı alt-graf
büyüklüğü, `fly_cx_modulated_e2e_multiseed.py`).

**Sonuç (koşuldu, tutarlı):** real medyan=25.310, null medyan=22.169,
Wilcoxon p=0.250, **Cliff's delta=-0.250** — gate FAIL (eşik altı) ama
YÖN YİNE AYNI (null lehine). Orijinal (taklit-kayıplı, eğitilmemiş
gain) sonuçtan (δ=-0.562) DAHA ZAYIF, ama ÇELİŞMİYOR — aynı yönde,
daha düşük büyüklükte. Ayrıca eğitim kaybı (final_train_loss ~30-50)
orijinal taklit-tabanlı eğitimden çok daha yüksek kaldı — 20 adımlık
bir bölümde doğrudan görev-kaybından öğrenmek (seyrek, gecikmiş kredi
atama), yoğun taklit süpervizyonundan çok daha zor bir optimizasyon
problemi; ağ muhtemelen tam yakınsamadı.

**Yorum:** İki farklı eğitim rejimi (taklit-süpervizyonlu vs uçtan-uca
görev-kayıplı), iki farklı büyüklükte ama AYNI YÖNDE bir null-lehine
sinyal veriyor — bu, bulgunun eğitim detayına aşırı duyarlı bir artefakt
olmadığını, gerçek bir substrate-bağımlı etki olduğunu destekliyor.
n=8'de zaten eşik altı (0.250<0.33) olduğu için 16 tohuma çıkarmaya
gerek görülmedi (session kuralı sadece eşiği GEÇEN sonuçlar için
geçerli).

---

## 2026-09-18 — ÖNEMLİ SAĞLAMLIK BULGUSU: "en temiz bulgu" derece-koruyan null'a karşı TAMAMEN KAYBOLUYOR

**Faz:** 2 (H2'' — sağlamlık kontrolü). Bu oturum boyunca H2 spektrum
deneylerinin neredeyse tamamı `er_null` (Erdős–Rényi) kullandı —
PROTOCOL.md'nin asıl birincil null'u ise `degree_preserving_rewire`
(Maslov-Sneppen çift-kenar takası: her nöronun giriş/çıkış derecesini
TAM olarak koruyor, sadece "kim kime bağlı" bilgisini yok ediyor). ER
null derece dağılımını HİÇ korumuz — yani ER'ye karşı çıkan bir fark,
"gerçek bağlantı ÖZGÜLLÜĞÜ" yerine sadece "FlyWire'ın derece dağılımı
ER'den farklı" olgusunu yakalıyor olabilir. Oturumun en temiz bulgusunu
(`fly_cx_modulated_multiseed.py`, δ=-0.562, 16/16 tohum, er_null'a
karşı) bu daha katı null'a karşı test ettim (`fly_cx_modulated_degreenull_multiseed.py`,
aynı alt-graf/encode-seed=16000, aynı eğitim, SADECE null model
değişti).

**Beklenti:** Eğer bulgu gerçek bağlantı-özgüllüğünden kaynaklanıyorsa,
derece-koruyan null'a karşı da (daha zayıf da olsa) aynı yönde bir
sinyal görmeyi bekliyordum.

**Sonuç (koşuldu, ÇARPICI):** real medyan=24.091, null medyan=24.357,
Wilcoxon p=0.461, **Cliff's delta=+0.125** — sadece eşik altı değil,
**YÖN DE DEĞİŞTİ** (ER'ye karşı -0.562/negatif iken, derece-koruyan
null'a karşı +0.125/pratikte sıfır). Etki TAMAMEN KAYBOLDU.

**Yorum (kritik, dürüst):** Bu, oturumun "en temiz bulgusu"nun aslında
**gerçek bağlantı özgüllüğünden değil, FlyWire'ın derece dağılımının
ER-rastgele graflardan farklı olmasından** kaynaklanmış olabileceğini
güçlü biçimde düşündürüyor. FlyWire ağları gerçek biyolojik ağlar gibi
çok-kuyruklu (heavy-tailed)/heterojen derece dağılımına sahiptir (az
sayıda çok-bağlantılı "hub" nöron), ER ise homojen/Poisson dağılımlı
derece üretir — bu FARK TEK BAŞINA, hiçbir "hangi nöron kime bağlı"
bilgisi olmadan, ağın dinamik davranışını (ve dolayısıyla eğitilebilirlik/
performansını) değiştirebilir. Derece-koruyan null bunu kontrol eder;
ER etmez.

**Bu, EN AZ, şu ana kadar bu oturumda er_null ile elde edilmiş TÜM H2
spektrum sonuçlarının (CX-modulated, PSO-sürü, alternatif topolojiler,
hatta belki mushroom body/antennal lobe/visual hierarchy devreleri)
YENİDEN gözden geçirilmesi gerektiği anlamına geliyor** — hangilerinin
gerçek bağlantı-özgüllüğü mü yoksa sadece derece-dağılımı farkı mı
yakaladığı belirsiz. **malecns (δ=1.000) ve slope-stability tek-atış
(δ=0.906) da er_null ile ölçülmüştü** — bunlar oturumun EN GÜÇLÜ
pozitif bulguları, derece-koruyan null'a karşı da test edilmeleri
BÜYÜK ÖNCELİK.

**Sonraki adım (yüksek öncelik):** malecns ve slope-stability tek-atış
sonuçlarını degree_preserving_rewire null'una karşı da test et — eğer
onlar da kaybolursa, oturumun "H1 lehine" görünen TÜM bulguları
derece-dağılımı artefaktı olabilir, ki bu FlyOpt'un ana negatif
tezini (connectome'un ölçülebilir avantajı çok nadir/kırılgan) daha da
güçlendirir.

**GÜNCELLEME — slope-stability tek-atış sonucu HAYATTA KALDI:**
`fly_slope_stability_degreenull_multiseed.py` ile aynı alt-graf
(encode-seed=9000), aynı eğitim (300 epoch), SADECE null değişti
(er_null yerine degree_preserving_rewire). Sonuç: real medyan=0.1155,
null medyan=0.1760, Wilcoxon p=0.00781, **Cliff's delta=1.000
(MÜKEMMEL ayrım — 8 real değerin HEPSİ 8 null değerin HEPSİNDEN
düşük)**, gate **PASS**.

**Bu, CX-modulated bulgusunun aksine, slope-stability tek-atış
sonucunun DERECE-DAĞILIMI ARTEFAKTI OLMADIĞINI, gerçek bağlantı
özgüllüğünden (kim-kime-bağlı bilgisinden) kaynaklandığını güçlü
biçimde destekliyor** — çünkü degree_preserving_rewire tam olarak
derece dağılımını koruyup SADECE spesifik bağlantı kimliğini yok
ediyor, ve etki YİNE DE (hatta ER'den daha güçlü/temiz biçimde) ortaya
çıkıyor. Bu, oturumun EN GÜVENİLİR pozitif bulgusu haline geldi —
hem er_null'a (δ=0.906) hem degree_preserving_rewire'a (δ=1.000) karşı
iki bağımsız, sağlam PASS.

**Genel yorum:** Bu iki sağlamlık kontrolü birlikte önemli bir ayrım
ortaya koyuyor: **er_null'a karşı görülen sinyal HER ZAMAN gerçek
bağlantı-özgüllüğü anlamına gelmiyor** (CX-modulated'de kayboldu —
muhtemelen sadece FlyWire'ın heterojen derece dağılımını
yakalıyordu), **ama bazı görevlerde (uzamsal, tek-atış toplu-regresyon
— şev stabilitesi) etki gerçekten kim-kime-bağlı olduğuna kadar iniyor**.
Bu, H2'' hipotezine ek bir boyut daha ekliyor: "hangi null modele karşı
test edildiği" de, görevin kendisi kadar önemli bir değişken. **malecns
(δ=1.000, er_null) da bu ek testi hak ediyor ama GPU/gerçek-lidar
gerektirdiği için bu oturumda hemen koşulmadı — sonraki bir oturumun
öncelikli işi olmalı.**

---

## 2026-09-21 — Modülerlik mekanizması n=30'da doğrulandı: boşluğun YARISINI açıklıyor

**Not:** Bu girişten önceki "MEKANİZMA BULUNDU" / "Üçüncü mekanizma
denemesi: MODÜLERLİK" bölümleri dosyanın ortasında (bir önceki
düzenleme bir erken bağlantı noktasına eklenmiş) — kronolojik olarak
BURADAN ÖNCE okunmalı, tarih başlıklarına göre takip edin.

`community_preserving_rewire` (TAM derece dizisi + gerçek Louvain
modülerliği, Q=0.365) n=8'den n=30'a çıkarıldı (22 yeni tohum).

**SONUÇ (n=30, ANLAMLI — n=8'deki gibi sönmedi, GÜÇLENDİ):**
```
real medyan = 0.1173   community-preserving medyan = 0.1373
Mann-Whitney U: p = 0.00443     Wilcoxon: p = 0.00434
Cliff's delta = 0.429
gate: PASS
```

n=8'de δ=0.375 (p=0.148, FAIL) idi — n=30'da δ=0.429'a YÜKSELDİ ve
anlamlılık kazandı. Bu, oturumda sık görülen "n=8 umut→n=30 sönme"
kalıbının TERSİ: sinyal gerçekmiş.

**Nihai mekanizma tablosu (şev stabilitesi, δ büyüklüğüne göre sıralı,
hepsi n=30 veya n=30'a yakın güçte):**
| null | ne koruyor | δ | anlamlı mı |
|---|---|---|---|
| degree_preserving | sadece TAM derece dizisi | 0.884 (n=30) | EVET (p≈4×10⁻⁹) |
| scale_free | jenerik heterojenlik (fly'a özgü değil) | 0.738 (n=30) | EVET (p≈10⁻⁶) |
| scale_free_correlated | + giriş/çıkış korelasyonu (fly'a özgü değil) | 0.719 (n=8) | EVET (p=0.015) |
| **community_preserving** | **TAM derece dizisi + TAM modülerlik** | **0.429 (n=30)** | **EVET (p=0.004)** |
| weight_shuffle | TAM topoloji (kim-kime-bağlı), ağırlık hariç | 0.258 (n=30) | HAYIR (p=0.088) |

**Nihai, dengeli yorum:** Modülerliği derece dizisine eklemek etkiyi
YAKLAŞIK YARIYA indiriyor (0.884→0.429) AMA TAMAMEN KAPATMIYOR — geriye
hâlâ istatistiksel olarak anlamlı, orta büyüklükte bir boşluk kalıyor.
**Modülerlik/kümesel yapı, "fly'ın derece dizisi neden özel" sorusunun
büyük bir kısmını (kabaca yarısını) açıklıyor, ama tek başına yeterli
değil.** Geriye kalan boşluk muhtemelen daha ince bir yapısal özellikte
saklı — motiflerin tam dağılımı (kaba bir triad-census denemesi zayıf
ama ilginç bir sinyal verdi, güvenilir değil), kümeler-arası bağlantı
deseninin ince yapısı, ya da fly'ın TAM kendine özgü wiring'inin
(degree_preserving'in bile yakalayamadığı bir şey) kalıntısı.

**Görselleştirme:** Tüm bu null-model karşılaştırmaları interaktif bir
özet sayfasında toplandı: https://claude.ai/artifact/2pE791tBWxsfVPLRL8CpSb
(delta çubuk grafiği, ham dağılım noktaları, "ne korunur/ne bozulur"
tablosu — community_preserving artık n=30 ile güncellenecek).

**Durum:** "NEDEN fly'ın derece dizisi özel" sorusu artık kısmen
cevaplı (modülerlik ~%50), tam cevaplı değil. Bu, ileri sürülebilecek
gerçekçi bir bilimsel iddia: "modülerlik, gözlemlenen avantajın önemli
bir bileşenidir, tek açıklaması değildir."

---

## 2026-09-21 — Tür-ötesi genelleme testi: C. elegans'ta bulgu TEKRARLANMADI (dürüst negatif sonuç)

**Faz:** 2 (H2'' — genellenebilirlik). Kullanıcının Gemini beyin
fırtınasından bir öneri (§3.2, "türler arası kıyaslama" — kullanıcı
uyarısıyla "fare beyni" yerine gerçekçi olan **C. elegans**'a
daraltıldı, çünkü fare beyninin tam sinaptik-çözünürlükte bir
connectome'u henüz yok). Kullanıcı onayıyla Cook ve ark. 2019 (Nature)
C. elegans hermafrodit kimyasal connectome'u indirildi
(networks.skewed.de/net/celegans_2019, açık akademik arşiv).

**Veri:** Somatik sinir sistemi (duyusal + ara-nöron + motor nöron,
kas/farinks/cinsiyete-özgü hücreler hariç) — 272 nöron, 3390 kenar,
ortalama derece 12.5. **Yapısal olarak FlyWire'a çarpıcı derecede
benzer:** modülerlik Q=0.381 (FlyWire: 0.357), kümeleme katsayısı=0.329
(FlyWire: 0.276) — bağımsız evrimleşmiş bambaşka bir tür, ama benzer
"hub-baskın + modüler" mimari.

**Yöntem:** AYNI şev-stabilitesi tek-atış regresyon görevi, AYNI
RateBrain mimarisi/eğitim tarifi — SADECE substrate C. elegans'ın
tam ağıyla (alt-graf örneklemesi gerekmedi, ağ zaten küçük) değiştirildi.
encode=duyusal nöron havuzu, decode=motor nöron havuzu (FlyWire'daki
afferent→efferent ile birebir aynı mantık). Üç null'a karşı test
edildi: er_null, degree_preserving_rewire, community_preserving_rewire
(C. elegans'ın kendi Louvain kümeleri kullanılarak).

**SONUÇ (n=8, HER ÜÇÜNDE DE FAIL — bulgu tekrarlanmadı):**
```
                       δ        p (Mann-Whitney)   gate
er_null                0.156    0.645               FAIL
degree_preserving      0.094    0.798               FAIL
community_preserving  -0.250    0.442               FAIL (yön bile ters)
```

**Yorum (dürüst, temkinli):** FlyWire'da bulunan modülerlik-ilişkili
avantaj, AYNI mimari ve görevle C. elegans'a taşınmadı. Bu birkaç
şekilde yorumlanabilir:
1. Bulgu gerçekten FlyWire'a (ya da Drosophila'ya) özgü olabilir —
   genel bir "modüler biyolojik ağ" özelliği değil.
2. C. elegans ağı çok daha küçük (272 vs 3000 düğüm) — aynı mimari
   parametreleri (T=8 iç adım, decode_scale=0.5 vb.) bu ölçekte
   yeniden ayarlanmamış, bu da haksız bir dezavantaj yaratmış olabilir.
3. n=8 ile güç sınırlı — daha büyük örneklemde farklı çıkabilir
   (ama yönler tutarsız/karışık, net bir "umut verici ama güçsüz"
   deseni bile göstermiyor, bu ölçekte gerçek bir etki olmadığına
   işaret ediyor).

**Bu sonuç ne anlama GELMİYOR:** "FlyWire bulgusu yanlış" anlamına
gelmiyor — o bulgu kendi içinde (n=30, iki bağımsız null'a karşı)
sağlam kalmaya devam ediyor. Bu sadece **genellenebilirliğin ispatlanmadığı**
anlamına geliyor — modülerlik hipotezinin "her modüler biyolojik ağ
işe yarar" gibi genişletilmiş bir versiyonu bu ilk denemede
DOĞRULANMADI. Kapsamı daraltıyoruz: bulgu şimdilik FlyWire'a özgü
kabul edilmeli, tür-genel bir iddia yapılamaz.

**Sonraki adım (yapılmadı, düşünülmeli):** Mimari parametrelerini
C. elegans'ın ölçeğine göre yeniden ayarlayıp (daha küçük T, farklı
decode_scale) tekrar denemek — mevcut haliyle bu karşılaştırma
"aynı hiperparametrelerle" adil ama "her substrate için ayrı ayarlanmış"
değil, bu yüzden negatif sonuç kesin değil, sadece ilk işaret.

---

## 2026-09-21 — scale_free_correlated de n=30'a çıkarıldı: mekanizma tablosu artık tam tutarlı

`fly_slope_stability_scalefreecorr_official30.py` ile 22 yeni tohum
(8-29) koşuldu (bir kez oturum kesintisiyle yarıda kalıp sıfırdan
yeniden başlatıldı — sonuç etkilenmedi, dosya baştan yazıldı).

**SONUÇ (n=30):**
```
real medyan = 0.1173   scale_free_correlated medyan = 0.1569
Mann-Whitney U: p = 0.0000600     Wilcoxon: p = 0.000283
Cliff's delta = 0.604
gate: PASS (n=8'deki 0.719'dan düştü ama hâlâ ezici anlamlı)
```

**Nihai, tam n=30-tutarlı mekanizma tablosu (şev stabilitesi):**
| null | ne koruyor | δ (n=30) | p (Mann-Whitney) | anlamlı mı |
|---|---|---|---|---|
| degree_preserving | TAM derece dizisi | 0.884 | ≈0.000000 | EVET |
| scale_free | jenerik heterojenlik | 0.738 | ≈0.000001 | EVET |
| scale_free_correlated | + giriş/çıkış korelasyonu | 0.604 | 0.00006 | EVET |
| community_preserving | TAM derece + TAM modülerlik | 0.429 | 0.00443 | EVET |
| weight_shuffle | TAM topoloji, ağırlık hariç | 0.258 | 0.0877 | HAYIR |

Bu tablo artık her satırda n=30, tutarlı istatistik. Genel sıralama
(degree_preserving en güçlü, weight_shuffle tek FAIL) değişmedi —
sadece scale_free_correlated'in kesinliği n=8'in "iyi" görünümünden
n=30'un daha gerçekçi (ama hâlâ güçlü) haline geldi. Mekanizma
araştırmasının bu oturumdaki hali burada tamamlanıyor: **modülerlik
büyük bir parça, tam açıklama değil; sentetik modeller (bağımsız ya
da korele) fly'ın kendi gerçek derece dizisinin gerisinde kalıyor.**

---

## 2026-09-21 — Modülerlik "doz-yanıt" testi: BELİRSİZ sonuç

**Faz:** 2 (H2'' mekanizma, ek doğrulama). resolution=1.0 (Q=0.357,
gerçeğe en yakın) için δ=0.429 (n=30) bulunmuştu. Hipotez: korunan
modülerlik miktarı arttıkça δ de artmalı ("doz-yanıt" ilişkisi). Bunu
test etmek için `community_preserving_rewire` iki farklı Louvain
çözünürlüğüyle (0.5→Q=0.088, düşük; 2.0→Q=0.315, orta-yüksek) n=8
pilot olarak koşuldu.

**Sonuç: her ikisi de AYNI büyüklükte etki verdi** (δ=0.469, p=0.130,
n=8, ikisi de FAIL) — düşük-Q (0.088) ile orta-yüksek-Q (0.315) arasında
BEKLENEN fark gözlenmedi (hatta ham sayı olarak resolution=1.0'ın
n=30 sonucundan -0.429- bile hafifçe YÜKSEK çıktı, ama n=8 vs n=30
karşılaştırması adil değil). Not: iki koşumun real_connectome dizileri
(beklenildiği gibi, resolution'dan bağımsız) birebir aynı, null
dizileri farklı ama Cliff's delta n=8'de kaba ayrıklaştırma nedeniyle
tesadüfen aynı değere denk geldi — bu bir bug değil, gerçek bir
istatistiksel tesadüf (doğrulandı).

**Yorum:** Doz-yanıt hipotezi bu pilotla NE doğrulandı NE çürütüldü —
n=8 çok düşük güçte, ve iki nokta arasında (0.088 ile 0.315) net bir
eğilim görülmüyor. Modülerliğin etkisi sürekli/doğrusal bir doz-yanıt
yerine "eşik" tarzı davranabilir (belirli bir Q'nun üzerinde herhangi
bir miktar yeterli olabilir) — ama bunu kanıtlamak için n=30'a
çıkarılması gerekir, şimdilik yapılmadı (düşük öncelik, ana bulgu
zaten n=30'da sağlam).

---

## 2026-09-21 — 3D nöron aktivasyon görselleştirmesi

**Faz:** Görselleştirme (kullanıcı isteği: "sinek beyni optimizasyon
yaparken beyninin nereleri ateşleniyor... 3d bir şekilde görmek
istiyorum"). `Supplemental_file1_neuron_annotations.tsv`'nin
`pos_x/pos_y/pos_z` sütunlarında gerçek FlyWire 3D koordinatları
olduğu bulundu — daha önce hiç kullanılmamıştı.

`scripts/record_activation_3d.py`: şev-stabilitesi alt-grafında (3000
düğüm, encode-seed=9000) gerçek connectome ile bir RateBrain eğitildi
(300 epoch, mse 0.52→0.14), sonra tek bir gerçek arama episode'u
(12 adım × 8 iç-adım = 96 kare) `torch.no_grad()` altında koşulup HER
substep'teki gizli durum `h` (3000 nöron × 96 kare) kaydedildi.
Episode gerçek ve anlamlı: fx (şev güvenlik faktörü arama hedefi)
2.391'den 2.025'e monoton düşüyor — arama gerçekten çalışıyor.

3D görselleştirme (Three.js, WebGL point cloud, gerçek FlyWire
koordinatlarında): https://claude.ai/artifact/2GXprm8qXPtPqC21CgYALm
— duyusal (encode) nöronlar yeşil, motor (decode) nöronlar turuncu,
diğerleri aktivasyon büyüklüğüne göre ısı renginde; oynat/durdur,
kare-kare gezinme, serbest 3D döndürme/yakınlaştırma.

---

## 2026-09-21 — İkinci fiziksel görev sınıfı: konsol kiriş tasarımı — umut verici, n=30'a çıkarılıyor

**Faz:** 2 (H2'' — görev-genellenebilirlik). Kullanıcı isteğiyle
(Gemini'nin FEA/yapısal önerisi) `benchmarks_structural.py` ile şev
stabilitesinden bağımsız, yapısal mühendislik alanında (konsol kiriş
kesit tasarımı: b,h → minimum gerilme+malzeme maliyeti) aynı tek-atış
regresyon çerçevesi test edildi. degree_preserving_rewire'a karşı,
n=8:

```
real medyan = 0.115   degree_preserving medyan = 0.178
Wilcoxon p = 0.039     Mann-Whitney p = 0.083
Cliff's delta = 0.531
gate: FAIL (eşik üstü, anlamlılık yetersiz — tanıdık "n=8 umut verici" deseni)
```

**Önemli:** Yön ve büyüklük şev-stabilitesinin kendi n=8 sonucuyla
(δ=0.906) TUTARLI — bu, bulgunun tek bir görevin tuhaflığı olmadığına,
gerçekten farklı bir fiziksel/mühendislik alanına da taşınabileceğine
dair ilk işaret. Session kuralı gereği n=30'a çıkarılıyor
(`beam_design_degreenull_official30.py`, arka planda, ~4 saat).

---

## 2026-09-21 — Erkek sinek CNS connectome'unda genellenebilirlik testi: NEGATİF (n=8, gate FAIL)

**Faz:** 2 (H2' — birey/cinsiyet-genellenebilirlik, C. elegans'ın tür-
genellenebilirlik testine paralel). Kullanıcı isteğiyle ("aynı çalışmayı
bir de erkek sinek beyni üzerinden ilerletelim") resmi bulgumuzun
(dişi FlyWire FAFB, şev stabilitesi, degree_preserving_rewire, δ=0.884
n=30) AYNI mimari/görev/null ile **erkek Drosophila CNS connectome**'unda
tekrarlanıp tekrarlanmadığı test edildi.

Veri: `fly_demos/malecns` projesinde (ayrı, üçüncü-parti bir proje —
kendi DAgger/F1 sürüş pipeline'ı var, o kullanılmadı) zaten indirilmiş
ham veriden (`data/graph_w5/edges.npz`, `neurons.parquet`, 166.700 nöron,
6.24M kenar) `build_malecns_adjacency.py` ile bizim formatımıza
çevrildi — yeni indirme gerekmedi. Afferent/efferent havuzları
`neurons.parquet`'in `superclass` sütunundan (birincil duyusal:
ol/cb/vnc_sensory; birincil motor: cb/vnc_motor + efferent) FlyWire'ın
`flow` alanına benzer mantıkla tanımlandı (17.937 afferent / 925
efferent).

**Sonuç (n=8 tohum, `malecns_slope_stability_degreenull_multiseed.py`):**

```
real medyan = 0.1485   degree_preserving medyan = 0.1494
Wilcoxon p = 0.742      Mann-Whitney p = 0.442
Cliff's delta = 0.250
gate: FAIL (eşik altı — n=30'a çıkarma kuralını TETİKLEMİYOR)
```

**Değerlendirme:** C. elegans testiyle aynı sonuç deseni — dişi FlyWire
FAFB'de bulunan avantaj, erkek CNS connectome'unda net şekilde
tekrarlanmıyor (δ=0.250, beam design'ın 0.531'inin bile altında,
istatistiksel olarak sıfırdan ayırt edilemez). PROTOCOL.md'nin
"hiperparametre kovalama" yasağı gereği n=8'de eşiği geçmeyen bu sonuç
n=30'a çıkarılmıyor, olduğu gibi negatif kaydediliyor.

**Önemli metodolojik not:** Bu, "connectome'lar hiçbir zaman avantaj
sağlamıyor" anlamına gelmiyor — tam tersine dişi FlyWire FAFB'de δ=0.884
(n=30, çok güçlü) ve konsol kiriş görevinde de δ=0.531 (n=8, aynı dişi
FlyWire alt-grafı) var. Şu ana kadarki desen: avantaj **belirli bir
connectome'a (dişi FlyWire FAFB'nin bu 3000-düğümlük alt-grafına) özgü
olabilir**, "sinek/nematod connectome'u genel olarak her zaman avantaj
sağlar" şeklinde bir evrensel iddia değil. Hem tür (C. elegans) hem
birey/cinsiyet (erkek CNS) eksenlerinde genellenebilirlik şu ana kadar
NEGATİF çıktı — bu, resmi bulgunun kapsamını daraltan, dürüstçe
kaydedilmesi gereken bir sınırlama.

---

## 2026-09-21 — Konsol kiriş tasarımı n=30'a çıkarıldı: İKİNCİ resmi PASS (δ=0.613)

**Faz:** 2 (H2'' — görev-genellenebilirlik). `beam_design_degreenull_official30.py`
(seeds 8-29, önceki n=8 ile birleştirilmiş) tamamlandı.

```
=== KONSOL KIRIŞ TASARIMI, DEGREE_PRESERVING NULL, n=30 ===
real medyan = 0.1183   degree_preserving medyan = 0.1824
Mann-Whitney U p = 0.000046
Wilcoxon işaretli-sıra p = 0.000003
Cliff's delta = 0.613
gate (Mann-Whitney p<0.05 AND |delta|>0.33): PASS
```

**Önemli:** Bu, dişi FlyWire FAFB alt-grafının **ikinci bağımsız
görevde** (şev stabilitesi δ=0.884'ün ardından, tamamen farklı bir
fiziksel/mühendislik alanında — yapısal kesit tasarımı) degree_preserving_rewire'a
karşı resmi olarak PASS aldığı ilk kayıt. n=8'deki (δ=0.531, gate FAIL
ama eşik-üstü) "umut verici ama yetersiz güç" deseni n=30'da net bir
PASS'e dönüştü — session'ın tekrarlanan "n=8 eşik-geçen → n=30 gerçek
PASS" deseniyle tutarlı.

Aynı gün içindeki erkek-CNS ve C. elegans negatif sonuçlarıyla birlikte
okununca resim netleşiyor: avantaj **dişi FlyWire FAFB'nin bu spesifik
3000-düğümlük alt-grafına** bağlı görünüyor, ama o alt-graf üzerinde
**birden fazla bağımsız göreve** (şev stabilitesi + kiriş tasarımı)
genelliyor — yani "tek bir görevin tuhaflığı" açıklaması artık zayıf,
"bu connectome'un bu alt-yapısının, tek-atış vektör-regresyonu tipi
görevlerde genel bir avantajı var" açıklaması güçleniyor.

---

## 2026-09-21 — Erkek CNS negatif sonucunun kontrolü: bug yok, ama yapısal olarak DAHA modüler olmasına rağmen bulgu yok

**Faz:** 2 (H2''' — mekanizma/tanı). Kullanıcı sordu: "neden dişi
sineğin yaptığını erkek sinek yapamadı, araştırdın mı, yanlış bir
işlem mi yaptık doğruladın mı?" Bu soruyu doğrudan test ettim: iki
alt-grafı (dişi FlyWire seed=9000, erkek CNS seed=9000) aynı
`select_connected_encode_decode`/`build_subgraph_bfs` prosedürüyle
yeniden kurup yapısal özelliklerini doğrudan karşılaştırdım.

**Bug kontrolü — TEMİZ:** İki alt-graf da sağlıklı — 6/6 encode, 30/30
decode nöronu tam doğrulanmış (gerçek çok-hop bağlantı ile), izole
düğüm yok, her ikisi de gerçek kenarlarla dolu. Erkek CNS'in negatif
sonucu bir pipeline hatasından KAYNAKLANMIYOR.

**Ama beklenmedik bir yapısal fark var:**

| Özellik | Dişi FlyWire (bulgu VAR) | Erkek CNS (bulgu YOK) |
|---|---|---|
| Alt-graf yoğunluğu | 0.01927 | 0.00853 (yarısından az) |
| Ortalama derece | 102.50 | 46.18 (yarısından az) |
| Modülerlik (Louvain Q) | 0.3565 | **0.6056 (çok daha yüksek)** |
| Ortalama kümeleme | 0.2759 | 0.3055 (biraz daha yüksek) |
| Topluluk sayısı | 9 | 6 |

**Bu, 2026-09-20/21'deki "modülerlik boşluğun yarısını açıklıyor"
mekanizma bulgusuyla GERİLİM içinde:** erkek CNS alt-grafı dişi
FlyWire'dan belirgin şekilde DAHA modüler (0.606 vs 0.357) ve biraz
daha kümeli, ama gerçek-vs-null avantajı GÖSTERMİYOR. Eğer modülerlik
tek başına yeterli bir mekanizma olsaydı, erkek CNS'te EN AZ eşit güçte
bir etki beklenirdi. Bu, modülerliğin (tek başına) yeterli bir açıklama
olmadığını, muhtemelen sadece "gerekli ama yetersiz" bir bileşen
olduğunu gösteriyor.

**En çarpıcı fark yoğunluk/derece:** erkek CNS alt-grafında düğüm başı
ortalama bağlantı sayısı dişininkinin yarısından az. Hipotez (TEST
EDİLMEDİ, spekülatif): RateBrain'in T=8 alt-adımlık sinyal yayılımı,
daha seyrek bir ağda hem gerçek hem null substrate'i benzer şekilde
"bilgi-aç" bırakıyor olabilir — yapının avantajını ortaya çıkaracak
kadar zengin bir bağlantı yoğunluğu bulunmuyor olabilir. Bu, ileride
doğrudan test edilebilir (ör. dişi FlyWire alt-grafını yapay olarak
erkek CNS yoğunluğuna seyreltip aynı deneyi tekrarlamak).

---

## 2026-09-21 — KRİTİK: alt-graf-tohum sağlamlık testi — seed=9000 istisna olabilir

**Faz:** 2 (H2'''' — resmi bulgunun kapsamı/genelliği). Bu oturumdaki
HER deney (şev stabilitesi δ=0.884 n=30, kiriş tasarımı δ=0.613 n=30,
3D görselleştirme) aynı encode/decode seçim tohumunu (seed=9000)
kullandı — yani hep AYNI, tek, rastgele seçilmiş 3000-nöronluk
alt-grafı test etti. Bu hiç değiştirilmemiş bir gizli değişkendi.
`fly_slope_stability_subgraphseed_robustness.py`, 2 YENİ encode/decode
seçim tohumuyla (9001, 9002 — aynı prosedür, farklı başlangıç) resmi
görevi degree_preserving_rewire'a karşı n=8 tekrar test etti.

**Sonuç:**

```
seed=9000 (resmi, n=30):  delta=0.884   Mann-Whitney p≈0.000000
seed=9001 (n=8):          delta=-0.219  Mann-Whitney p=0.51   FAIL
seed=9002 (n=8):          delta=0.250   Mann-Whitney p=0.44   FAIL
```

**Bu, projenin şu ana kadarki en önemli sınırlama bulgusu:** İki
bağımsız alternatif alt-graf, resmi bulgunun büyüklüğüne YAKLAŞAMADI
bile — biri ters yönde. Bu, δ=0.884'ün "dişi FlyWire FAFB'nin genel
bir özelliği" olmaktan çok, **seed=9000'in seçtiği spesifik alt-grafa
özgü bir istisna** olabileceğini gösteriyor. Kullanıcı onayıyla,
8 YENİ alt-graf tohumuyla (9003-9010, n=4 her biri, genişlik-odaklı
tarama) `fly_slope_stability_subgraphseed_sweep.py` başlatıldı —
amaç: δ'nın çoklu bağımsız alt-graf çekiminde DAĞILIMINI ölçüp
seed=9000'in tipik mi istisna mı olduğunu netleştirmek. Sonuç bekleniyor.

**Önemli:** Bu bulgu henüz resmi sonucu ÇÜRÜTMÜYOR (n=2 alternatif de
kendi başına küçük bir örneklem — tıpkı erken pozitif sonuçlara
güvenmediğimiz gibi, erken negatif/karışık sonuçlara da hemen
güvenmiyoruz), ama resmi bulgunun "dişi FlyWire'ın genel özelliği"
şeklindeki geniş yorumunu ciddi şekilde sorguluyor ve ek doğrulama
gerektiriyor.

---

## 2026-09-21 — Alt-graf-tohum taraması tamamlandı: endişe büyük ölçüde HAFİFLEDİ, "bölgesel uzmanlaşma" tablosu doğrulandı

**Faz:** 2 (H2'''' devamı). GPU'ya geçiş sonrası (RTX 4060, 13.8x
hızlanma — bkz. aşağıdaki GPU girişi) 8 YENİ alt-graf tohumuyla
(9003-9010, n=4 her biri) `fly_slope_stability_subgraphseed_sweep.py`
tamamlandı. Toplam 11 bağımsız alt-graf çekimi (9000-9010):

```
seed=9000: delta=0.884 (n=30, resmi)
seed=9001: delta=-0.219 (n=8)
seed=9002: delta=0.250 (n=8)
seed=9003: delta=0.250 (n=4)
seed=9004: delta=-0.250 (n=4)
seed=9005: delta=1.000 (n=4)
seed=9006: delta=0.375 (n=4)
seed=9007: delta=0.750 (n=4)
seed=9008: delta=0.625 (n=4)
seed=9009: delta=0.375 (n=4)
seed=9010: delta=1.000 (n=4)
```

**Sonuç: 11 alt-graftan 9'u (%82) POZİTİF yönde, medyan delta=0.500.**
Sadece 2 tohum (9001, 9004) negatif çıktı. Bu, önceki girişteki
"seed=9000 tek bir şanslı istisna olabilir" endişesini büyük ölçüde
hafifletiyor — çoğu bağımsız alt-graf gerçekten avantajlı yönde,
sadece büyüklük değişken (0.25'ten 1.00'a). Kullanıcının önerdiği
**"beyin bölgesel olarak uzmanlaşmıştır, hepsi eşit güçlü olmak
zorunda değil"** hipotezi bu veriyle uyumlu: etki genel ama
derece-derece değişken, "tek bir şanslı seçim" açıklamasından çok
"heterojen ama çoğunlukla pozitif bir dağılım" tablosu doğru.

**Uyarı/sınırlama:** n=4 küçük bir örneklem (Cliff's delta 1/16
adımlarla kesikli, ±1.0 uçlarına kolayca sıçrayabilir) — tek tek
tohum değerlerine aşırı güvenmemeli, asıl bilgi AGREGE örüntüde
(9/11 pozitif). İleride daha yüksek n ile (GPU sayesinde artık ucuz)
bu tohumların bir kısmı n=30'a çıkarılıp daha kesin bir dağılım
elde edilebilir.

---

## 2026-09-21 — GPU'ya geçiş: 13.8x hızlanma, saatler süren işler dakikalara indi

**Faz:** altyapı. Kullanıcı isteği ("GPU kullanın, tüm kaynakları
kullanın") üzerine makinede boşta duran bir RTX 4060 (8GB) bulundu.
`RateBrain.forward()`/`train()` ve `SlopeContinuousBrain`'de cihaz-
farkındalığı eksikliği (h/inject/pre tensörleri hep CPU'da
oluşturuluyordu) bulunup düzeltildi. Ölçüm: CPU 300 epoch=298s,
CUDA 300 epoch=21.6s — **13.8x hızlanma**. Kuyruktaki tüm scriptler
(alt-graf taraması, T=16 simetrik test, durum-sıfırlama ablasyonu)
GPU'ya geçirildi.

Ayrıca kullanıcının isteğiyle, Colab'da GPU üzerinde çalıştırılabilir
kapsamlı bir doğrulama notebook'u yazıldı
(`notebooks/flyopt_colab_verification_suite.ipynb`) — alt-graf
genelliği, T-duyarlılığı ve (aşağıda ayrıntılı açıklanan) Pareto
korelasyon iddiasının null-kontrollü testini içeriyor.

---

## 2026-09-21 — Dış "Pareto/Foraging biyolojik bilgelik" iddiası: NULL-KONTROLLÜ testte ÇÜRÜTÜLDÜ

**Faz:** metodoloji/doğrulama. Kullanıcının paralel olarak danıştığı
Gemini, `RateBrain`'i 2 boyutlu ("Enerji"/"Ödül") çıktı ile 10.000
rastgele senaryoda çalıştırıp iki çıktı boyutu arasında -0.657
korelasyon bulmuş ve bunu "sineğin evrimsel Pareto-dengesi içgüdüsü"
olarak yorumlamıştı (`biyolojik_yanit_sonuclari_10000.csv`). **Hiçbir
null-model karşılaştırması yapılmamıştı.**

Aynı ölçüm (eğitilmemiş/rastgele başlangıç ağırlıklı RateBrain, dim=2,
T=20) hem gerçek connectome'da hem degree_preserving_rewire null'unda
tekrarlandı:

```
gercek connectome (egitilmemis): corr=-0.444
degree_preserving null (egitilmemis): corr=-0.245
```

**Sonuç: null model de güçlü negatif korelasyon gösteriyor.** Bu,
korelasyonun connectome'a özgü "biyolojik bilgelik" olmadığını,
2 boyutlu bir doğrusal okuma katmanının (readout matrix), paylaşılan
yüksek-boyutlu bir gizli durumdan iki çıktı türetirken genel olarak
gösterdiği bir matematik özelliği olduğunu kanıtlıyor. Bu proje için
tekrarlayan bir ders bir kez daha doğrulandı: **null-model
karşılaştırması olmadan hiçbir korelasyon/bulgu "biyolojik" olarak
yorumlanamaz** (bkz. malecns/CX-modulated'in er_null'a karşı çöküşü,
2026-09-18/19).

---

## 2026-09-21 — T=16 SİMETRİK test tamamlandı: "erkek CNS'e zaman tanı" hipotezi DESTEKLENMEDİ

**Faz:** 2 (H2''''' — erkek CNS negatif sonucunun T-parametresi
açıklaması). `T_symmetric_robustness.py` (GPU, ~10 dk) hem dişi
FlyWire hem erkek CNS'i AYNI koşumda T=8 yerine T=16 ile,
degree_preserving_rewire'a karşı, n=8 tekrar test etti (p-hacking
riskinden kaçınmak için tek taraflı değil simetrik).

**Sonuç:**

```
                  T=8 (referans)      T=16 (bu test)
disi FlyWire:     delta=0.884 (n=30)  delta=0.844 (n=8)   -- degismedi, guclu
erkek CNS:        delta=0.250 (n=8)   delta=0.031 (n=8)   -- ZAYIFLADI
```

Dişi taraf T artışından etkilenmedi (beklendiği gibi — zaten yeterli
yoğunluğa sahip). Erkek CNS tarafı ise T=16'da GÜÇLENMEK yerine
neredeyse sıfıra indi (δ=0.031, Mann-Whitney p=0.96). **Bu, "erkek
CNS'in seyrek ağı sadece daha fazla düşünme süresine ihtiyaç
duyuyor" hipotezini DESTEKLEMİYOR** — kontrollü/simetrik test bunu
göstermedi.

**Önemli bağlam:** Kullanıcının paralel olarak danıştığı Gemini,
kontrolsüz bir T=20 testinde (sadece erkek CNS, dişi kontrolü yok)
δ=0.469 bulmuştu (n=8, gate FAIL). Bizim kontrollü T=16 sonucumuz
(δ=0.031) bununla çelişiyor. En olası açıklama: Gemini'nin n=8
sonucu küçük-örneklem gürültüsüydü (bu oturumda defalarca görülen
"n=8 umut verici → daha büyük/kontrollü testte sönme" deseninin bir
örneği daha), gerçek bir "sinyale zaman tanıma" etkisi değil. T=16 ile
T=20 arasındaki fark tam eşleşmeyi engelliyor ama yön (T artışının
erkek CNS'i güçlendirmesi bekleniyordu, tersi oldu) net bir uyarı
işareti.

**Sonuç olarak erkek CNS negatif bulgusu (δ=0.250, T=8) hâlâ ayakta
duruyor** — ne yapısal (density/degree) ne T-parametresi açıklaması
şu ana kadar bunu tersine çevirebildi.

---

## 2026-09-21 — Durum-sıfırlama ablasyonu: kalıcı hafıza kısmen zararlı, ama tek başına yeterli açıklama değil

**Faz:** 2 (H2 — çok-adımlı/kapalı-döngü görevin negatif sonucunun
mekanizması). Kullanıcının fikri ("her adıma sanki ilk kez geliyormuş
gibi davransın") test edildi: `SlopeContinuousConfig.reset_state_each_step=True`
ile beynin durumu (`h`) episode boyunca değil her adımın başında
sıfırlanıyor (x yine adım adım ilerliyor, best_fs yine episode
boyunca izleniyor).

**Sonuç:**

```
Kalici durum (v4, er_null'a karsi, n=8):              delta=-0.156  FAIL
Her adimda sifirlama (bu test, degree_preserving'e karsi, n=8): delta=+0.125  FAIL
```

Yön negatiften zayıf pozitife döndü (üstelik bu sefer daha sıkı bir
null'a karşı test edildi) — **kalıcı hafızanın gerçekten kısmen
zararlı olduğuna dair zayıf bir işaret**. Ama δ=0.125 hâlâ eşiğin
(0.33) çok altında — yani "hafızayı kaldırınca güçlü bulgu ortaya
çıktı" değil, "biraz iyileşti ama çok-adımlı arama süreci hâlâ
tek-atış versiyonundaki (δ=0.884) gibi güçlü bir avantaj
göstermiyor." Sonuç: kapalı-döngü/çok-adımlı görev sınıfının kendi
başına (hafızadan bağımsız olarak) tek-atış görevlerden daha az
avantajlı olduğu görüşü destekleniyor.

---

## 2026-09-21 — Alt-graf yapısal özellikleri δ'yı açıklamıyor (n=11, anlamlı korelasyon yok)

**Faz:** 2 (H2'''''' — bölgesel uzmanlaşma hipotezinin mekanizması).
Kullanıcının "beyin bölgesel uzmanlaşmıştır" hipotezini doğrudan test
etmek için 11 alt-graf tohumunun (9000-9010) yapısal özellikleri
(yoğunluk, modülerlik Q, kümeleme katsayısı, ortalama derece, topluluk
sayısı) ile daha önce ölçülen δ değerleri arasında Spearman
korelasyonu hesaplandı (`correlate_subgraph_structure_with_delta.py`,
eğitim gerektirmiyor, sadece yapısal hesap).

**Not (yan bulgu):** Bu analiz sırasında `select_connected_encode_decode`'ın
çok-hop BFS döngüsünün 3-4. hop'ta frontier'ın grafın neredeyse
tamamına (135-136k/139k düğüm) yayılması nedeniyle hop başına
30-90 saniyeye kadar yavaşladığı doğrulandı — bu YAVAŞ ama gerçek bir
hesaplama, çökme/kilitlenme değil (PowerShell'in bu ortamdaki CPU
ölçümü güvenilmez çıktı, yanlışlıkla "takılı kaldı" sanılıp bir kere
gereksiz yere durdurulup yeniden başlatıldı).

**Sonuç: hiçbir yapısal özellik anlamlı korelasyon göstermedi**
(hepsi p>0.1):

```
density        : rho=-0.078  p=0.820
modularity_Q    : rho=+0.229  p=0.499
avg_clustering  : rho=-0.421  p=0.197
mean_degree     : rho=-0.124  p=0.717
n_communities   : rho=-0.499  p=0.118
```

En güçlü (ama yine de anlamsız) eğilimler kümeleme katsayısı ve
topluluk sayısı ile negatif yönde. **Bölgesel uzmanlaşma hipotezi
hâlâ makul** (alt-graf taramasının %82 pozitif oranı duruyor), ama
bu kaba graf-teorik ölçütlerle açıklanamıyor — gerçek sürücü muhtemelen
daha spesifik bir şey (hangi hücre tiplerinin/nöropillerin dahil
olduğu) ya da n=11'in küçüklüğünden kaynaklanan gürültü. Mekanizma
sorusu açık kalıyor.

---

## 2026-09-22 — Hücre tipi bileşimi ipucu buldu: `ascending` nöron oranı δ ile korele (keşifsel, temkinli)

**Faz:** 2 (H2'''''' devamı, otonom takip — kaba yapısal metrikler
başarısız olunca daha ince taneli bir hipotez test edildi).
`correlate_celltype_composition_with_delta.py`, 11 alt-grafın FlyWire
`super_class` bileşimini (optic/central/sensory/ascending/descending/
motor/vb. oranları) δ ile Spearman korelasyonuna soktu.

**Sonuç (n=11, 9 kategori test edildi — çoklu karşılaştırma
düzeltmesi YAPILMADI, aşağıda belirtiliyor):**

```
frac_ascending         : rho=+0.618  p=0.043  (en guclu)
frac_sensory_ascending : rho=+0.600  p=0.051  (sinirda)
frac_sensory           : rho=-0.544  p=0.084  (egilim, negatif)
frac_visual_projection : rho=+0.476  p=0.139
frac_central            : rho=+0.467  p=0.148
```

**Yorum (keşifsel, doğrulanmamış):** Alt-grafta `ascending` (VNC'den
beyne aktarım yapan, kısmen önceden-işlenmiş bilgi taşıyan ara-nöron)
oranı yüksekse δ daha güçlü; saf `sensory` (birincil duyusal, ham
girdi) oranı yüksekse δ daha zayıf eğiliminde. Biyolojik olarak
makul bir hipotez (ascending nöronlar zaten bir ölçüde entegrasyon
yapıyor, saf duyusaldan daha "yapılandırılmış" olabilir) ama:

- **n=11 küçük, 9 kategori test edildi, düzeltme yapılmadı** —
  Bonferroni düzeltmesiyle (α=0.05/9≈0.0056) hiçbir sonuç anlamlı
  kalmaz. Bu bulgu KEŞİFSEL bir ipucu, kanıtlanmış bir mekanizma
  değil.
- İleride doğrulama: bilerek yüksek/düşük `ascending` oranlı
  alt-graflar inşa edip bağımsız bir n ile önceden-tescillenmiş
  (pre-registered) bir testle doğrulanmalı.

---

## 2026-09-22 — Gemini'nin "sinek vs PSO" notebook'ları incelendi ve DOĞRU kıyaslamayla yeniden kurgulandı: hiçbir avantaj yok

**Faz:** metodoloji/doğrulama. Kullanıcı Gemini'nin yazıp Colab'da
çalıştırdığı 4+ notebook'u (`Benchmark_Duyu_Motoru.ipynb`,
`Benchmark_Karsilastirma_Colab.ipynb`, `Benchmark_Evrimlesmis_Sinek.ipynb`,
`Mega_Biyolojik_Turnuva.ipynb`, `FlyOpt_Colab_Missions.ipynb`) ve
çıktılarını (CSV/PNG) inceledi.

**Kritik metodolojik hata bulundu:** Bu notebook'lar, sinek beynine
hedef fonksiyonun (Sphere/Rastrigin) GERÇEK matematiksel gradyanını
(`obj.sum().backward()`, `x.grad`) doğrudan sensory girdi olarak
besleyip, gradyansız klasik PSO'ya karşı yarıştırmıştı. Bu, birinci-
derece (gradyanlı) bir yöntemi sıfırıncı-derece (gradyansız) bir
yönteme karşı test etmek demek — **connectome gerçek, null, hatta
rastgele olsa da hemen hemen aynı avantajı gösterirdi**, çünkü
avantaj yapıdan değil gradyan-enjeksiyonundan geliyor. Hiçbir
null-connectome karşılaştırması yoktu (18 "Mega Turnuva" varyantının
hepsi AYNI gerçek connectome'u kullanıyordu). `FlyOpt_Colab_Missions.ipynb`'nin
"Foraging/Pareto" hücresi de daha önce çürütülen aynı eğitilmemiş/
rastgele-girdi tasarımını tekrarlıyordu (`biyolojik_yanit_sonuclari_10000.csv`'nin
kaynağı muhtemelen bu).

**Düzeltilmiş versiyon:** Kullanıcı onayıyla ("aynı mantığı null
connectome baseline ile yeniden kurgula") `sensory_fly_nullcontrol_benchmark.py`
(+ Colab versiyonu `notebooks/sensory_fly_nullcontrol_benchmark.ipynb`)
yazıldı — AYNI gradyan-enjeksiyon tasarımı, T=20, erkek CNS, ama PSO
yerine **aynı alt-grafın degree_preserving_rewire null'una karşı**,
n=8 bağımsız tohumla (bu projenin standart istatistiğiyle).

**Sonuç: HİÇBİR AVANTAJ YOK.**

```
Sphere:     real_medyan=0.00040  null_medyan=0.00023  delta=0.094  p=0.798  FAIL
Rastrigin:  real_medyan=0.23956  null_medyan=0.40683  delta=0.000  p=1.000  FAIL (tam berabere)
```

Bu, hem şüpheyi doğruluyor (PSO'ya karşı "zafer" connectome'a özgü
değilmiş) hem de projenin daha önceki bulgusuyla (soyut/uzamsal-
olmayan görevlerde — Rastrigin, Fly-Surrogate — connectome hiç avantaj
göstermemişti, PROTOCOL_V2_TASK_SPECTRUM.md §5 satır 1-2) tam tutarlı.
Gemini'nin notebook'larının mühendisliği (Colab/Drive/GPU entegrasyonu)
sorunsuzdu, ama kıyaslama tasarımı projenin bilimsel sorusuna kanıt
sağlamıyordu — bu artık düzeltilmiş haliyle kayıt altında.

---

## 2026-09-22 — "Evrensel Turnuva" (5 arazi): kendi ölçütüyle bile PSO kazandı

**Faz:** metodoloji/doğrulama, kapanış notu. `Benchmark_Evrensel_Turnuva.ipynb`
(Gemini, Colab, kullanıcı tarafından koşuldu) 18 Fly varyantını 5 klasik
optimizasyon arazisinde (Sphere, Rastrigin, Ackley, Rosenbrock, Beale)
PSO'ya karşı test etti — yukarıdaki gradyan-enjeksiyonu tasarımının
aynısı, düzeltilmemiş (null-connectome kontrolü yok).

**Sonuç: PSO, 5 arazinin 5'inde de, denenen 18 varyantın EN İYİSİNİ bile
geride bıraktı** (`gemini_parallel_findings/outputs/Evrensel_Turnuva_Sonuclari.csv`):

```
             PSO           en iyi Fly (18 varyanttan)
Sphere:      4.91e-29      6.77e-08
Rastrigin:   0.0 (tam)     3.48e-02
Ackley:      9.54e-07      1.18e-02
Rosenbrock:  0.0 (tam)     9.30e-03
Beale:       0.0 (tam)     3.81e-06
```

**Önemli:** Bu, kendi (kusurlu) kıyaslama eksenlerinde bile "Sinek PSO'yu
yeniyor" anlatısının tutmadığını gösteriyor — üstelik gradyan doğrudan
verildiği halde. `sensory_fly_nullcontrol_benchmark.py`'nin (real vs
null connectome, δ≈0, tam berabere) bulgusunu tamamlayan bağımsız bir
ikinci kanıt: bu tip soyut/düşük-boyutlu matematik fonksiyonlarında
(Sphere/Rastrigin/Ackley/Rosenbrock/Beale) FlyOpt mimarisinin **hiçbir
versiyonu** (gerçek, null, ya da PSO'ya karşı) avantaj göstermiyor —
projenin en başından beri bilinen "soyut/uzamsal-olmayan görevlerde
connectome hiç avantaj göstermiyor" bulgusunu üçüncü, bağımsız bir
koşumla doğruluyor. Bu hat kapatıldı sayılabilir.

---

## 2026-09-22 — Sakkadik navigasyon hipotezi test edildi: DESTEKLENMEDİ

**Faz:** 2 (H2 — çok-adımlı görevin negatif sonucunun mekanizması,
üçüncü deneme). Kullanıcının gözlemi ("sinekler serkeş serkeş
dolanıyor, düz gitmek yerine hep bir manevra halindeler") ve gerçek
biyolojik bulguyla (Drosophila uçuşu düz-bacak + ani balistik
sakkad'lardan oluşur, sürekli yörünge entegrasyonu değil) motive
edilen yeni bir görev tasarlandı: `fly_saccadic_navigation_multiseed.py`
— 20 küçük adım yerine 6 SEYREK, BÜYÜK (step_scale=3.0, duman testiyle
ayarlandı) "sakkad" kararı, her kararda beyin durumu sıfırlanıyor,
kararlar arasında hiç düzeltme yok.

**Sonuç: δ=0.062, p=0.878, FAIL — durum-sıfırlama ablasyonundan
(δ=0.125) bile zayıf, istatistiksel olarak sıfırdan ayırt edilemiyor.**

```
Tek-atislik (resmi):                    delta=0.884  (n=30)
Surekli, kalici hafiza (v4):            delta=-0.156 (n=8)
Surekli, sifirlanan hafiza (ablasyon):  delta=+0.125 (n=8)
Sakkadik, seyrek/buyuk, sifirlanan:     delta=+0.062 (n=8)
```

**Yorum:** Hipotez desteklenmedi. Sorun "adımların küçük mü büyük mü,
sık mı seyrek mi olduğu" değilmiş — herhangi bir çok-adımlı/çok-bacaklı
yörünge (ister sürekli küçük düzeltme ister seyrek büyük sakkad), tek-
atışlık kararın gücüne yaklaşamıyor. Asıl ayrım "tek karar" ile "birden
fazla kararın zincirlenmesi" arasında görünüyor, sakkadik olup
olmamasında değil. Sinek uçuşunun gerçekten sakkadik olması (biyolojik
olarak doğru bir gözlem) ile burada ölçtüğümüz hesaplama avantajının
mekanizması birbirinden bağımsız çıktı — biri diğerini açıklamıyor.
Çok-adımlı görev sınıfının negatif sonucu (§ yukarıdaki girişler)
üçüncü bir bağımsız denemeyle de doğrulanmış oldu.

---

## 2026-09-22 — Tek-atışlık SINIFLANDIRMA görevi: FAIL (öğrenilebilir ama fark zayıf)

**Faz:** 2 (H2 — tek-atışlık avantajın hesaplama TİPİNE mi yoksa görev
FORMUNA mı bağlı olduğu). Sakkadik navigasyonun desteklenmemesi
sonrası, kaynaklar kanıtlanmış güçlü yöne (tek-atışlık) ama YENİ bir
hesaplama tipine (sürekli regresyon değil, ayrık sınıflandırma)
yönlendirildi. `fly_octant_classification_multiseed.py`: aynı
doğrulanmış makine (rays/teacher_delta, şev stabilitesi arazisi,
RateBrain, aynı alt-graf seed=9000), ama çıktı doğru kaçış yönünün
işaret deseni (2³=8 "oktan") sınıflandırması — çapraz-entropi kaybı,
şans seviyesi %12.5.

**Sonuç:**

```
sans seviyesi: 12.5%
real medyan: 55.5%   null medyan: 50.0%
Mann-Whitney p=0.674   Cliff's delta=0.141   FAIL (esigin altinda, n=30'a cikarilmiyor)
```

**Değerlendirme:** Görev tasarımı sağlıklı — hem gerçek hem null ağ
şans seviyesinin çok üzerinde öğreniyor (%50-55), yani anlamlı/
öğrenilebilir bir görev. Ama gerçek-null farkı zayıf ve anlamsız,
n=8'de eşiği (0.33) geçmediği için session kuralına göre n=30'a
çıkarılmıyor, net FAIL olarak kaydediliyor. **Bu, "tek-atışlık her
zaman kazanır" genellemesini sorguluyor** — bulgu belki sadece SÜREKLİ
REGRESYON (yön tahmini) tipi görevlere özgü olabilir, ayrık
sınıflandırmaya genellenmiyor olabilir. Tek-atışlık x regresyon/
sınıflandırma ayrımı artık görev spektrumuna eklenen yeni bir eksen.

---

## 2026-09-22 — Lob rekabeti: görsel-kökenli alt-graf TERS yönde, koku/dokunma-kökenli eşiği geçti

**Faz:** 2 (H2'''''' devamı — hücre-tipi bileşimi ipucunun takibi).
Kullanıcının önerisiyle, alt-graf inşasında afferent havuzu rastgele
karışık kullanmak yerine bilerek ANATOMİK OLARAK KISITLANDI:
`fly_lobe_competition_multiseed.py`, `cell_class` sütununu (zaten
indirilmiş `Supplemental_file1_neuron_annotations.tsv`, yeni veri
gerekmedi) kullanarak iki alt-graf kurdu — **görsel** (fotoreseptörler,
havuz n=11390) ve **koku/dokunma/tat** (olfactory+mechanosensory+
gustatory+..., havuz n=5593). Aynı efferent (motor) havuzu, aynı şev
stabilitesi görevi, aynı degree_preserving_rewire null, n=8.

**Sonuç:**

```
gorsel (optik) kokenli:      delta=-0.219  (TERS yonde)
koku/dokunma kokenli:        delta=+0.375  (esigi gecti, p=0.235, gate FAIL -- n=30'a cikariliyor)
referans (karisik havuz, resmi seed=9000, n=30): delta=0.884
```

**İlk yorum (n=30 sonucu bekleniyor):** Görsel/optik kökenli alt-graf
hiç avantaj göstermiyor, hatta hafifçe ters yönde — bu, ascending-nöron
ipucuyla (§ önceki giriş) ve "her bölge eşit değil" hipoteziyle uyumlu.
Koku/dokunma kökenli alt-graf n=8'de eşiği geçti (session kuralı
gereği n=30'a çıkarılıyor) — karışık havuzun (δ=0.884) gücüne
yaklaşıp yaklaşmadığı henüz belirsiz.

**n=30 sonucu:** `lobe_chemo_mechano_official30.py` tamamlandı —
n=8'deki umut verici sinyal (δ=0.375) n=30'da küçüldü:

```
real medyan=0.1667  null medyan=0.1670
Mann-Whitney p=0.154   Wilcoxon p=0.021 (karisik sinyal)   Cliff's delta=0.216
gate: FAIL
```

**Sonuç (iki lob testi birlikte):** Ne görsel-kökenli (δ=-0.219) ne
koku/dokunma-kökenli (δ=0.216, n=30) alt-graf, karışık havuzun gücüne
(δ=0.884) yaklaşabiliyor. Bu, **tek bir duyu modalitesine kısıtlamanın
avantajı zayıflattığını** gösteriyor — bulgu muhtemelen birden fazla
duyu tipinin (görsel+koku+dokunma) BİRLİKTE bulunmasından geliyor, tek
bir modaliteden değil. Bu, doğal olarak bir sonraki hipotezi (çoklu-
duyu birleşimi/entegrasyonunun kendisinin avantajın bir parçası olması)
motive ediyor — henüz test edilmedi.

---

## 2026-09-22/23 — Çoklu-duyu birleşimi hipotezi DÜZGÜN KONTROLLE test edildi: DESTEKLENMEDİ

**Faz:** 2 (H2'''''''' — lob rekabeti bulgusunun bir "fusion" etkisi mi
yoksa alt-graf-tohum şansı mı olduğunun ayrımı). Yukarıdaki tek-örnekli
lob rekabeti sonucunun (mixed=0.884 vs visual=-0.219/chemo=0.216, n=30)
gerçek bir "duyu birleşimi" mekanizmasından mı, yoksa daha önce
belgelenen (§ alt-graf-tohum genelliği, δ aralığı -0.25 ile 1.00) sıradan
alt-graf-tohum varyansından mı geldiğini ayırt etmek için,
`sensory_fusion_seedsweep.py` her üç koşulu (mixed/visual/chemo) 2
BAĞIMSIZ alt-graf tohumuyla (9000, 9011), n=4 her biri, test etti
(ilk versiyon sistem düşük-bellek uyarısıyla durduruldu, hafifletilip
her koşul ayrı process olarak yeniden koşuldu).

**Sonuç:**

```
              seed=9000   seed=9011    medyan
mixed:            1.000      0.125     0.562
visual:            0.375      0.250     0.312
chemo:             0.625      0.625     0.625
```

**"Çoklu-duyu birleşimi" hipotezi DESTEKLENMEDİ.** Karışık havuzun
kendi içinde bile δ aşırı değişken (0.125-1.000) — seed=9011'de
neredeyse sıfır avantaj. Koku/dokunma tek başına bu küçük taramada en
yüksek VE en tutarlı medyanı verdi (0.625, iki tohumda da). Görsel tek
başına da pozitif çıktı (önceki tek-örnekli testin -0.219'unun aksine).
**Sonuç: önceki lob rekabeti testindeki çarpıcı fark, büyük ihtimalle
gerçek bir duyu-birleşimi mekanizmasından değil, sıradan alt-graf-tohum
şansından kaynaklanıyordu** — n=1 örneklemin (her koşul için tek
alt-graf) yeterli olmadığının bir kez daha teyidi. n=4 küçük bir
örneklem, kesin değil, ama amaç zaten kesinlik değil "tek-örnek farkı
gerçek mi şans mı" sorusuna kaba bir cevaptı — cevap "muhtemelen şans"
yönünde çıktı. Bu hat, ek doğrulama olmadan kapatılıyor.

---

## 2026-09-23 — Sinek+PSO warm-start hibriti: iki denemede de avantaj bulunamadı

**Faz:** 2 (H2 — kullanıcının fikri, "PSO'ya sinek ile başlangıç noktası
verilse ne olur?"). `fly_pso_warmstart_hybrid.py`: eğitilmiş tek-atışlık
RateBrain'in (hem gerçek hem degree_preserving null connectome) tahmin
ettiği yönler, PSO'nun başlangıç popülasyonunu "ısıtmak" için kullanıldı
— aynı şev stabilitesi arazisi, aynı alt-graf (seed=9000).

**İlk deneme (PARÇACIK=300, İTERASYON=80): test tasarımı kusurluydu.**
Duman testi PSO'nun bu bütçeyle rastgele başlangıçtan bile aynı optimuma
yakınsadığını gösterdi (std=0.0001, 10 rastgele koşum) — yani hiçbir
başlangıç noktası (gerçek, sahte, rastgele) bir fark yaratamazdı. Sonuç
δ=0.000 (tam berabere) — hipotez ÇÜRÜTÜLMEDİ, sadece ÖLÇÜLEMEDİ.

**Düzeltilmiş deneme (PARÇACIK=15, İTERASYON=10):** Duman testi gerçek
varyans olduğunu doğruladı (std=0.023, 10 rastgele koşum) — bu bütçede
başlangıç noktası gerçekten önemli olabilir. n=8 tam koşum:

```
baseline (rastgele) medyan: 1.6125
gercek-isitma medyan:       1.6796
sahte-isitma medyan:        1.6796
Wilcoxon p=0.031  Mann-Whitney p=0.713  Cliff's delta (gercek vs sahte)=0.125
gate: FAIL (esigin altinda, n=30'a cikarilmiyor)
```

**Ham veri karışık:** seed=2'de gerçek-ısıtma çarpıcı şekilde kazandı
(0.988 vs baseline 1.478 vs sahte 1.475), ama seed=5'te tam tersi oldu
(gerçek-ısıtma 2.179, baseline'dan [1.716] bile kötü). Diğer seed'lerde
neredeyse fark yok. **Sonuç: düzgün test edilmiş warm-start hibriti,
güvenilir bir avantaj göstermedi** — bazı tekil seed'lerde büyük kazanç
görülse de ortalamada anlamsız/tutarsız. Bu fikir, mevcut haliyle
desteklenmiyor; büyük tekil kazanımların (seed=2) gerçek mi gürültü mü
olduğu n=8 ile ayırt edilemiyor.

---

## 2026-09-23/24 — Gece otonom bataryası: 9 yeni görev kategorisi (sinirlarin disinda "hunerler")

**Bağlam:** Kullanıcı, sınır-haritalama fazının optimizasyon-eksenli
görevlerin ötesine geçmesini istedi: "bence artık sineğin sınırlarını
keşfedelim sadece optimizasyon problemleri değil farklı problemler
tanımlayalım... sabaha kadar çalış". Tam otonom, soru sormadan çalışma
izni verildi. Aşağıdaki 9 görev, resmi n=8 pilot + eşik geçilirse n=30
protokolüyle, degree_preserving_rewire null'una karşı test edildi.

### 1. Yapısal hasar direnci (lezyon testi) — `fly_lesion_robustness_multiseed.py`

Eğitilmiş şev-stabilitesi ağının (gerçek vs null) rastgele %10/20/30
tekrarlayan-kenarı sıfırlanınca (yalnızca çıkarım anında, yeniden eğitim
yok) MSE'nin bozulma oranı (ablasyonlu/temel MSE, düşük=daha sağlam)
ölçüldü.

**n=8 pilot sonucu (X=20% resmi kapı):**
```
real medyan=2.612   null medyan=1.588
Mann-Whitney p=0.0019   Cliff's delta=-0.875   gate: PASS (esik gecildi)
```
Üç seviyenin de (10/20/30%) kapıyı geçmesi (delta -0.875 ile -0.938
arası) n=30'a genişletmeyi tetikledi (bkz. bir sonraki günlük girdisi).
**Dikkat: işaret ters yönde** — gerçek connectome, aynı derece dizisine
sahip null'dan DAHA FAZLA bozuluyor, "biyolojik ağlar rastgele hasara
karşı daha dayanıklıdır" sezgisinin aksi yönünde bir bulgu adayı.

### 2. Duyusal gürültü direnci — `fly_noise_robustness_multiseed.py`

Aynı eğitilmiş ağlara, girdi ray'lerine orantılı Gauss gürültüsü (std =
girdinin kendi std'sinin %15'i) eklenince MSE bozulma oranı ölçüldü, n=8.

### 3. Çapraz-görev sıfır-atış transferi — `fly_crosstask_zeroshot_transfer_multiseed.py`

Yalnızca şev-stabilitesinde eğitilen ağ (yeniden eğitim yok, aynı
encode/decode nöronları), kiriş-tasarımı görevine (3B'ye "dummy" 3.
boyutla gömülü, beam_cost 3. boyutu görmezden geliyor) doğrudan
uygulandı. n=8.

### 4. Az-örnekli veri verimliliği — `fly_fewshot_efficiency_multiseed.py`

Standart 200 örnek yerine yalnızca 6 örnekle (2 problem x 3 başlangıç)
300 epoch eğitim, büyük bir tutulan test setinde değerlendirme. n=8.

### 5. Yol entegrasyonu (dead-reckoning) — `fly_path_integration_multiseed.py`

YENİ bir zamansal görev (santral-kompleks/kafa-yönü sistemine biyolojik
benzetme): T=8 adımlık rastgele 2B adım dizisi, her adımda FARKLI bir
vektör enjekte edilerek (RateBrain.forward()'ın sabit enjeksiyonundan
farklı, özel bir `forward_sequence` ile), son adımda kümülatif toplamın
negatifini ("eve dönüş vektörü") tahmin etme. n=8.

### 6. Kaotik zaman serisi tahmini (Mackey-Glass) — `fly_mackey_glass_prediction_multiseed.py`

Projenin kendi referans listesindeki (Costi ve ark. 2025) reservoir-computing
literatürüyle doğrudan bağlantılı standart bir kıyas: tau=17 Mackey-Glass
serisi, kayan 8-değerlik pencere ile bir sonraki değeri tahmin. Her
tohum kendi bağımsız üretilmiş serisini kullanıyor (%70/%30 eğitim/test
bölünmesi); pencereler arası durum sıfırlanıyor (tam zaman-zinciri BPTT
değil, izlenebilirlik için bilinçli bir basitleştirme — belgelendi). n=8.

### 7. Gecikmeli eşleştirme (çalışma belleği) — `fly_delayed_match_sample_multiseed.py`

t=0'da rastgele 2B örnek vektör enjekte edilir, ardından 6 adım boş
girdi (saf dikkat dağıtıcı gecikme), son adımda orijinal örneğin geri
çağrılması istenir. n=8.

### 8. Anomali/sapma tespiti — `fly_anomaly_detection_multiseed.py`

T=8 vektörlük dizide 7'si normal (küçük Gauss kümesi), 1'i (rastgele,
önceden bilinmeyen konumda) belirgin şekilde farklı; ağ son adımda
sapan vektörün gerçek değerini regresyonla tahmin ediyor (ayrık konum
sınıflandırması DEĞİL — sınıflandırma çerçevelemesinin daha önce
başarısız olduğu bulgusuyla tutarlı, EXPERIMENTS.md 2026-09-22/23).

### 9. Looming / çarpışma-zamanı tahmini — `fly_looming_ttc_multiseed.py`

Sineğin en iyi belgelenmiş refleks devresi (LGMD-tipi loom algılama),
sürekli regresyon olarak çerçevelendi: standart 1/mesafe hiperbolik
genişleme formülüyle T=8 adımlık görünen-boyut dizisi, son adımda kalan
çarpışma-zamanının tahmini.

**Not:** Bu girdi, gece bataryasının başlangıç kaydıdır — her testin tam
sonucu tamamlandıkça ayrı günlük girdileriyle eklenecek; bu girdi
yalnızca tasarımları ve #1'in ilk (n=8) sonucunu sabitliyor.

---

## 2026-09-23/24 — RESMİ SONUÇ: Lezyon direnci testi (n=30) — gerçek connectome DAHA KIRILGAN

**Faz:** Gece bataryası #1, `fly_lesion_robustness_official30.py`. n=8
pilot üç ablasyon seviyesinin (10/20/30%) hepsinde kapıyı geçti; n=30'a
genişletildi (22 yeni tohum, 8-29, orijinal 0-7 ile birleştirildi).

```
X=10%: real medyan=2.498  null medyan=1.450  Mann-Whitney p<0.00001  delta=-0.913  PASS
X=20%: real medyan=2.468  null medyan=1.588  Mann-Whitney p<0.00001  delta=-0.844  PASS
X=30%: real medyan=2.892  null medyan=1.639  Mann-Whitney p<0.00001  delta=-0.876  PASS
```

(Metrik: ablasyonlu MSE / temel MSE, düşük = daha sağlam. Kapı X=20%'de
resmi.)

**Bu, projenin resmi bulgularından biri olacak kadar sağlam, ama daha
önceki hiçbir sonucun aksi yönünde bir bulgu.** Şev-stabilitesinde
(δ=0.884, aynı alt-graf, aynı eğitim) üstün olan gerçek connectome, aynı
eğitilmiş ağın rastgele kenar-silme altındaki davranışına bakıldığında,
degree_preserving_rewire null'undan DAHA HIZLI bozuluyor (bozulma oranı
~1.6-1.9x daha yüksek). Yorum: gerçek connectome'un modüler/yapısal
avantajı, görev-performansını artırırken aynı zamanda onu belirli
kenarlara daha bağımlı (daha az dağıtık/yedekli) kılıyor olabilir — yani
"daha iyi çalışan ama daha kırılgan" bir yapı. Bu, "biyolojik ağlar
hasara karşı doğal olarak daha dayanıklıdır" sezgisiyle çelişiyor (en
azından rastgele/hedefsiz hasar ve bu spesifik tek-atışlık görev için);
hedefli/kademeli hasar veya farklı bir görev bu dengeyi değiştirebilir,
test edilmedi.

---

## 2026-09-23/24 — RESMİ SONUÇ: Gürültü direnci testi (n=30) — gerçek connectome DAHA KIRILGAN (tekrar)

**Faz:** Gece bataryası #2, `fly_noise_robustness_official30.py`. n=8
pilotta mükemmele yakın ayrışma (delta=-1.000) görülünce n=30'a
genişletildi (22 yeni tohum + orijinal 8, aynı ağlar/eğitim).

```
n=30: real medyan=2.286  null medyan=1.072  Mann-Whitney p<0.000001  Cliff's delta=-0.978  PASS
```

Girdiye orantılı Gauss gürültüsü (rel. std %15) eklenince gerçek
connectome'un hata oranı ortalama ~2.1x artıyor, null'unki ise neredeyse
değişmiyor (~1.07x). **Lezyon-direnci sonucuyla (aynı gece, δ=-0.84 ila
-0.91) aynı yönde ve daha da güçlü bir tekrar** — iki bağımsız pertürbasyon
türü (yapısal hasar ve girdi gürültüsü) aynı sonucu veriyor: gerçek
connectome'un görev-performansı avantajı, sağlamlık pahasına geliyor
gibi görünüyor. Bu artık tek bir testin garipliği değil, tutarlı bir
örüntü — sınır haritasına eklenen ikinci, bağımsız doğrulanmış "ters
yönlü" resmi bulgu.

## 2026-09-23/24 — Yol entegrasyonu (dead-reckoning) n=8: FAIL

`fly_path_integration_multiseed.py`. T=8 adımlık rastgele 2B adım dizisi
entegre edilip "eve dönüş vektörü" tahmin edildi (santral-kompleks/kafa-
yönü sistemine biyolojik benzetme, ama tamamen yeni bir görev tasarımı,
literatürden alınmadı).

```
real medyan=1.3958  null medyan=1.3749  Mann-Whitney p=0.645  Cliff's delta=-0.156  FAIL
```

Neredeyse tam berabere — gerçek connectome bu temel zamansal-entegrasyon
görevinde nulldan hiçbir şekilde ayrışmıyor. Sınıflandırma ve sakkadik-
manevra FAIL'leriyle birlikte, avantajın zamansal entegrasyon görevlerine
de genellenmediğini gösteren üçüncü bağımsız negatif.

---

## 2026-09-23/24 — RESMİ SONUÇ: Mackey-Glass kaotik zaman serisi tahmini (n=30) — gerçek connectome DAHA KÖTÜ

**Faz:** Gece bataryası #6, `fly_mackey_glass_prediction_official30.py`.
n=8 pilot kapıyı geçti (delta=-0.719, p=0.0148); 22 yeni bağımsız üretilmiş
Mackey-Glass serisiyle (8-29) n=30'a genişletildi.

```
n=30: real medyan=0.8715  null medyan=0.8423  Mann-Whitney p=0.00423  Cliff's delta=-0.431  PASS
```

Standart, literatürle doğrudan bağlantılı (Costi ve ark. 2025) bir
reservoir-computing benchmarkında, gerçek connectome degree_preserving
null'undan anlamlı şekilde DAHA KÖTÜ tahmin performansı gösteriyor.
Etki büyüklüğü lezyon/gürültü testlerinden daha küçük (-0.43 vs -0.84/-0.98)
ama yön aynı ve istatistiksel olarak sağlam. **Üçüncü bağımsız "ters
yön" bulgusu** — gerçek connectome'un fiziksel/uzamsal tek-atışlık
regresyonundaki üstünlüğü, zamansal/kaotik tahmin görevine hiç
taşınmıyor, aksine kötüleşiyor.

## 2026-09-23/24 — Gecikmeli eşleştirme n=8 (FAIL) ve anomali tespiti n=8 (FAIL)

İki ek zamansal görev, hiçbiri kapıyı geçmedi:
- Gecikmeli eşleştirme (çalışma belleği): delta=-0.391, p=0.207, FAIL.
  Gerçek ağın hatası tüm tohumlarda ~0.50-0.51 gibi çok dar bir aralıkta
  sabit kalıyor — muhtemelen dejenere/ortalama-tahmine yakınsıyor.
- Anomali/sapma tespiti: delta=-0.156, p=0.645, FAIL — neredeyse tam
  berabere.

Gece bataryasının zamansal-görev ekseni özeti (yol entegrasyonu, Mackey-
Glass, gecikmeli eşleştirme, anomali tespiti): 4 testten yalnızca 1'i
(Mackey-Glass) kapıyı geçti, o da TERS yönde. **Avantajın zamansal/
sıralı görevlere genellenmediğine dair tutarlı kanıt.**

---

## 2026-09-23/24 — GECE BATARYASI: 9 görev, konsolide sonuç ve sentez

**Tam sonuç tablosu:**

| # | Görev | n | δ | p (MW) | Kapı |
|---|-------|---|---|--------|------|
| 1 | Lezyon direnci (X=20%) | 30 | -0.844 | <0.00001 | **PASS (ters yön)** |
| 2 | Gürültü direnci | 30 | -0.978 | <0.000001 | **PASS (ters yön)** |
| 3 | Çapraz-görev sıfır-atış transfer | 8 | -0.406 | 0.195 | FAIL |
| 4 | Az-örnekli veri verimliliği | 8 | 0.469 | 0.130 | FAIL |
| 5 | Yol entegrasyonu | 8 | -0.156 | 0.645 | FAIL |
| 6 | Mackey-Glass zaman serisi | 30 | -0.431 | 0.0042 | **PASS (ters yön)** |
| 7 | Gecikmeli eşleştirme | 8 | -0.391 | 0.207 | FAIL |
| 8 | Anomali/sapma tespiti | 8 | -0.156 | 0.645 | FAIL |
| 9 | Looming/çarpışma-zamanı | 8 | 0.031 | 0.959 | FAIL |

**Sentez.** 9 testten 3'ü istatistiksel kapıyı geçti (n=30'da doğrulandı)
— ama HİÇBİRİ gerçek connectome'u null'a göre ÜSTÜN göstermedi. Üçü de
(lezyon direnci, gürültü direnci, Mackey-Glass) gerçek connectome'un
null'dan anlamlı şekilde DAHA KÖTÜ/DAHA KIRILGAN olduğunu gösterdi.
Kalan 6 test (çapraz-görev transfer, az-örnekli verimlilik, yol
entegrasyonu, gecikmeli eşleştirme, anomali tespiti, looming) hiçbir
yönde anlamlı fark bulamadı.

Bu gecenin en önemli katkısı, sınır haritasına yeni bir eksen eklemesi:
**"görev performansı" ile "sağlamlık/dayanıklılık" birbirinden bağımsız,
hatta muhtemelen ters ilişkili eksenler.** Gerçek connectome'un
şev-stabilitesi/kiriş-tasarımındaki güçlü avantajı (δ=0.884/0.613),
onu rastgele yapısal hasara ve girdi gürültüsüne karşı daha DİRENÇSİZ
yapıyor gibi görünüyor — modülerlik/özelleşme muhtemelen performans için
gerekli olan şeyin ta kendisi, ama bu özelleşme yedeklilik/dağıtıklık
pahasına geliyor. Ayrıca, hiçbir yeni görev kategorisinde (zamansal
entegrasyon, çalışma belleği, anomali tespiti, az-örnekli öğrenme,
çapraz-görev transfer, looming algılama) gerçek connectome lehine bir
avantaj bulunamadı — bu, önceki bulgunun ("avantaj yalnızca tek-atışlık,
ileri-beslemeli, sürekli uzamsal yön regresyonunda") ne kadar dar
olduğunu bir kez daha, bağımsız yollarla teyit ediyor.

**Genel tablo artık şöyle özetlenebilir:** gerçek FlyWire alt-grafı,
evrimleştiği (varsayılan) hesaplama sınıfında (fiziksel/uzamsal tek-atış
yön bulma) sağlam bir avantaja sahip, ama bu avantaj (a) dar bir görev
sınıfına özgü, (b) başka türlere/bireylere genellenmiyor, VE artık (c)
sağlamlık/dayanıklılık pahasına geliyor gibi görünüyor. Bu üçüncü nokta
kullanıcının orijinal sorusuna ("sineğin başka hüneri var mı?") dolaylı
bir yanıt: hayır, test edilen hiçbir yeni "hüner" doğrulanmadı, ama
mevcut hünerin bir maliyeti olduğu keşfedildi.

**Batarya tamamlandı.** 9/9 görev test edildi (n=8 pilot, 3'ü n=30'a
genişletildi). Yeni bir görev fikri kendiliğinden üretilmeyecek —
sonuçlar kullanıcının sabah incelemesini bekliyor. Makale artifact'ı
(flyopt_makale_2026-09.html) bilinçli olarak GÜNCELLENMEDİ; bu, sonraki
oturumun kullanıcıyla birlikte gözden geçirdikten sonra yapacağı bir iş.

---

## 2026-09-24 — DÜZELTME: Gece bataryasının 3 "ters yön PASS" bulgusunun 2'si sorgulanıyor, 5 zamansal görev GEÇERSİZ

Kullanıcı uyurken, yukarıdaki 9 sonucu rapora işlemeden önce bağımsız
olarak doğruladım -- şaşırtıcı/hikayeye çok iyi oturan sonuçlar (özellikle
yönü ters çeviren "PASS" bulguları) tam olarak bu projenin geçmişte en çok
hataya düştüğü yer (yanlış null, ölçüm artefaktı, aşırı güçlü bütçe vb.),
o yüzden ham veriye indim.

### 1) Lezyon direnci (X=20%, resmi kapı) -- metrik artefaktıydı, düzeltilince FAIL

Orijinal metrik "ablasyonlu_MSE / kendi_taban_MSE"siydi. Sorun: gerçek
connectome'un tabanı zaten çok daha düşük (örn. seed=0: gerçek=0.110,
null=0.217 -- bu, bilinen resmi delta=0.884 avantajının ta kendisi). Aynı
mutlak hata artışını daha küçük bir paydaya bölmek, gerçek ağı otomatik
olarak "daha kırılgan" gösterir -- bu bir yapı özelliği değil, aritmetik
bir yapaylık.

Kanıt (n=8 pilot, X=20%): mutlak ablasyonlu MSE'de gerçek ağ sadece 3/8
tohumda null'dan iyi, oran metriğinde ise 0/8. Aynı ham veriyi MUTLAK MSE
üzerinden (oran değil) yeniden hesapladım (n=30, orijinal 0-7 + uzatma
8-29 birleştirilmiş):

```
X=10%: real medyan=0.2964  null medyan=0.2506  MW p=0.0035  delta=-0.44  PASS (zayif)
X=20%: real medyan=0.2935  null medyan=0.2809  MW p=0.246   delta=-0.176 FAIL  <== RESMI KAPI
X=30%: real medyan=0.3361  null medyan=0.2965  MW p=0.047   delta=-0.30  FAIL
referans (ablasyonsuz taban): delta=0.944, PASS -- bilinen avantaji dogru sekilde yeniden uretiyor
```

**Duzeltilmis sonuc: resmi kapida (X=20%) FAIL.** Dogru yorum: hasar
altinda gercek connectome'un avantaji buharlasip null'un seviyesine
YAKINSIYOR (istatistiksel olarak ayirt edilemez hale geliyor), ama
null'dan guvenilir sekilde DAHA KOTU oldugu iddia edilemez. "Gercek
connectome hasara karsi daha kirilgan" basligi geri cekilmistir; dogrusu
"avantaj hasara karsi dayanikli degil" (farkli, daha zayif bir iddia).

### 2) Gurultu direnci -- n=8 duzeyinde mutlak metrikle teyit edildi, n=30 icin mutlak-deger dogrulamasi eksik

n=8 pilotun ham verisinde (`real_noisy`/`null_noisy` sutunlari, oran
degil) ayni gate hesaplamasini mutlak MSE uzerinden tekrarladim:

```
n=8, mutlak gurultulu MSE: real medyan=0.2566  null medyan=0.1807
MW p=0.0147  delta=-0.719  PASS (ters yon)
```

Bu, oran-tabanli n=8 sonucuyla (delta=-1.000) ayni yonde ve gate'i geciyor
-- lezyon testinin aksine, burada mutlak terimde de 7/8 tohumda gercek
agin gurultulu hatasi null'dan yuksek, yani baseline-farki artefaktina o
kadar bagli degil. **Bu bulguyu gecerli/tentatif olarak koruyorum.** Ama
n=30 genisletme scripti (`fly_noise_robustness_official30.py`) sadece
orani kaydetti, mutlak `noisy`/`baseline` degerlerini diske yazmadi --
yani resmi n=30 rakami (delta=-0.978) hala oran-metrigine dayaniyor ve tam
bagimsiz mutlak-deger teyidi bekliyor (seed 8-29'un mutlak-degerlerle
yeniden kosulmasi gerekiyor, bu gece yapilmadi, sure kisiti).

### 3) BES zamansal/sirali gorev (yol entegrasyonu, gecikmeli eslestirme, anomali tespiti, looming, Mackey-Glass) -- GECERSIZ, mimari kusur nedeniyle

Bu besi de fork'un o gece icat ettigi YENI bir "sirali enjeksiyon"
mekanizmasini paylasiyor (`forward_sequence`, her script'te ayri ayri
kopyalanmis). Ham verideki uyari isareti: bircok tohumda gercek ve null
aglarin test MSE'leri neredeyse BIREBIR ayni (or. anomali seed=0: 1.26447
vs 1.26447; looming seed=0: 0.75746 vs 0.75730; gecikmeli-eslestirme
seed=0: 0.50295 vs 0.50295 -- tam esit). Bu, iki farkli agin rastgele
degil, IKISININ DE ayni dejenere/sabit cozume cokmus olmasinin imzasidir.

Dogrulama: her gorevin hedef dagilimi icin "sifir tahmin et" veya
"ortalamayi tahmin et" gibi TRIVIYAL bir tahminin MSE'sini hesapladim:
- Mackey-Glass: triviyal "son degeri tekrarla" (persistence) taban
  MSE≈0.018 (seri son derece puruzsuz/otokorelasyonlu, tau=17 gecikmeli
  ODE). Ama HEM gercek HEM null ag ≈0.83-0.90 MSE veriyor -- persistence
  taban cizgisinden ~45 KAT daha kotu, "sifir/ortalama tahmin et"
  seviyesinde (0.85). Hicbiri gercekte serinin yapisini ogrenmemis.
- Yol entegrasyonu: hedef = -8 rastgele 2B adimin toplami, teorik "sifir
  tahmin et" varyansi ≈1.33 -- gozlenen MSE'ler (1.32-1.43) tam bu
  seviyede.
- Anomali tespiti: hedef aykiri-deger vektoru, teorik "sifir tahmin et"
  MSE'si ≈1.19 -- gozlenen (1.19-1.27) tam bu seviyede.
- Looming: hedef=(T_total-7)/10 ~ U(0.3,3.3), "ortalamayi tahmin et"
  MSE'si =0.75 -- gozlenen (0.735-0.757) tam bu seviyede.

Kok-neden teshisi (Mackey-Glass uzerinde dogrudan test edildi): 1500
epoch'a cikarmak VE decode_scale'i 0.5→2.0→5.0 araliginda degistirmek
HICBIR SEYI degistirmedi -- `pred_std=0.0000` her kosulda sabit kaldi
(ag, girdiden tamamen bagimsiz SABIT bir sayi cikariyor). Bu bir
hiperparametre sorunu degil, mimari bir kusur: bu 5 gorevin hepsinde
okuma (decode), SON enjeksiyondan HEMEN SONRA, ARADA HIC YAYILMA ADIMI
BIRAKMADAN yapiliyor. Halbuki resmi/dogrulanmis gorevlerde (sev-
stabilitesi, kiris) AYNI ray vektoru T=8 adimin HEPSINDE tekrar tekrar
enjekte ediliyor -- sinyal surekli tazeleniyor. Bu yeni 5 gorevde ise her
adimda FARKLI bir deger SADECE BIR KEZ enjekte ediliyor; son adimin
enjeksiyonu decode noronuna (genelde birkac hop uzakta) ulasmak icin hic
zaman bulamiyor, onceki adimlarin bilgisi de kisa T=8 ve zayif/egitilmemis
kenar agirliklariyla yolda sonumleniyor.

**Sonuc: bu 5 test, connectome hakkinda HICBIR SEY soylemiyor -- ne PASS
ne FAIL, sadece "gecersiz/tasarim hatali" olarak isaretlenmeli.**

**DUZELTME (ayni gece, ikinci gecis): yukaridaki "yayilma icin adim
kalmiyor" hipotezi test edildi ve YANLIS cikti.** Mackey-Glass icin
`forward_sequence`'a enjeksiyon dizisinden SONRA 0/2/4/8 ekstra "durulma"
(sifir-girdi) adimi eklendi (800 epoch, decode_scale=2.0) -- HICBIRI
`pred_std=0.0000` cokusunu duzeltmedi. Daha derin bir tani yapildi:
egitilmemis bir ag uzerinde, encode noruna T=8 adim boyunca DEV bir
sinyal (deger=100) tekrar tekrar enjekte edilip decode norolarindaki
etkisine bakildi -- **fark tam olarak 0.0**, sadece enjeksiyon
norununun kendisinde deger degisiyor (0.996, tanh doygunlugu). Yani bu
gorevde secilen TEK encode norunu (index 418), decode norolarinin
HICBIRINE, hicbir yol uzerinden ulasmiyor -- 8 adim, hatta 8 ek durulma
adimi bile fark etmez, cunku baglanti YOK (agirlik kucuk degil, YOL YOK
ya da agirlik tam sifir).

**Gercek kok neden:** `select_connected_encode_decode` fonksiyonu
baglantiyi muhtemelen encode ADAYLARININ TOPLULUGU icin dogruluyor
(n_encode_candidates=200 icinden secilen grubun EN AZ BIRININ decode
kumesine ulastigini kontrol ediyor olabilir), tek tek HER encode noronu
icin degil. Resmi/dogrulanmis gorevlerde N_ENCODE=6 (sev-stabilitesi
rays) oldugu icin 6 adaydan en az biri calissa yeterli -- kusur
gizleniyor. Ama bu 5 yeni gorevin coğunda N_ENCODE=1 veya 2 (tek/iki
skaler enjeksiyon noktasi) kullanildi; sansa bagli olarak o TEK nokta
decode kumesine gercekten baglanmayan bir aday olabiliyor, ve oyle oldu.

Somut, doğru duzeltme onerisi (ileride bir oturumda, `rate_brain.py`
icinde): `select_connected_encode_decode`, HER encode noronu icin ayri
ayri (decode kumesine en az bir yonlu yol var mi) dogrulamali; N_ENCODE
kucuk oldugunda uygun aday bulunana kadar yeniden secim yapmali. Ayrica
her yeni gorev tasarimi icin, egitim BASLAMADAN once, "buyuk-genlik-
enjeksiyon -> decode noronlarinda gozle-gorulur fark var mi" seklinde
ucuz bir on-kontrol (bu teshiste kullanilan yontemin ta kendisi)
standart pratik haline getirilmeli.

**Ikinci, daha kesin kok-neden bulundu (kod okuyarak, `rate_brain.py`
satir 68-116 ve 172-225):** yukaridaki "encode-adaylari topluluk halinde
dogrulaniyor, tek tek degil" varsayimim da tam dogru degil --
`select_connected_encode_decode` her adayi ZATEN ayri ayri, TAM GRAF
uzerinde dogruluyor (`ranked` listesi her adayin KENDI reach kumesine
gore siralaniyor, `len(r)>0` filtresi tek tek uygulaniyor). Asil kopukluk
BURADA degil, iki ayri BFS prosedurunun UYUMSUZLUGUNDA: (1)
`select_connected_encode_decode`, baglantiyi TAM grafta (139k dugum)
dogruluyor; (2) `build_subgraph_bfs` ise SONRADAN, encode+decode
dugumlerinin bilesiminden disa dogru BAGIMSIZ bir BFS ile 3000 dugumluk
bir ALT KUME cikariyor (docstring "GUARANTEES forward connectivity"
diyor ama bu SADECE encode/decode'un ait oldugu bagli bilesenin genis
tutulmasini saglar, adim-1'de tam grafta BULUNAN spesifik YOLUN
ayni sirayla/ozellikle o 3000 dugume dahil edilecegini GARANTI ETMEZ --
dugum-ID sirasina gore acgozlu genisliyor, spesifik yolu degil).
N_ENCODE=6 oldugunda (resmi gorevler) 6 bagimsiz adaydan en az birinin
YOLU sans eseri bu 3000'lik altkumeye dahil olma ihtimali yuksek;
N_ENCODE=1-2 oldugunda (bu 5 yeni gorev) bu sans dusuyor, ve bu gece
tam da bunun oldugu dogrudan gosterildi (encode dugumu 418, tam grafta
DOGRULANMIS olmasina ragmen, cikarilan 3000-dugumlu alt-grafta HICBIR
decode dugumune ulasamiyor).

**Onemli uyari:** bu, `rate_brain.py`'nin PAYLASILAN, projenin TUM resmi
bulgularinin (delta=0.884/0.613 dahil) uzerine kuruldugu cekirdek
altyapisinda bir bosluk. Bu gece bu fonksiyonlara DOKUNULMADI (bilincli
tercih) -- boyle temel bir degisiklik, mevcut resmi sonuclari (n=30
regresyon testi olarak yeniden calistirip AYNI delta=0.884/0.613
degerlerini verdigini dogrulamadan) YAPILMAMALI. Bu, gelecek bir
oturumun dikkatli, ayri bir gorevi olarak birakiliyor -- bu gece
otonom olarak boyle riskli bir degisikligi yapmamak bilinçli bir
karardi, ihmal degil.

### Protokol-uyum notu: 2 test n=30'a genisletiliyor

Fork'a verdigim direktifte yanlislikla "gate = p<0.05 VE |delta|>0.33"yi
genisletme kriteri olarak yazdim -- oysa bu projenin YERLESIK kurali
"|delta|>0.33 TEK BASINA n=8'de genisletmeyi tetikler, p-degerinden
BAGIMSIZ OLARAK". Bu benim hatamdi, fork'un degil. Iki test bu yuzden
atlanmisti:
- Capraz-gorev sifir-atis transfer (delta=-0.406, n=8'de p=0.195 ama |delta|>0.33)
- Az-ornekli veri verimliligi (delta=0.469, n=8'de p=0.130 ama |delta|>0.33)

Her ikisi de GUVENILIR (tek-atislik ray tabanli, dogrulanmis) pipeline'i
kullaniyor -- mimari kusur riski yok. `fly_crosstask_zeroshot_transfer_official30.py`
ve `fly_fewshot_efficiency_official30.py` yazildi ve n=30'a genisletme su
an arka planda calisiyor; sonuclar ayri bir gunluk girdisinde
raporlanacak.

### Guncellenmis genel tablo (bu duzeltmeden sonra)

| Test | Onceki iddia | Duzeltilmis durum |
|---|---|---|
| Lezyon direnci (X=20%) | PASS (ters yon), delta=-0.844 | **FAIL** (mutlak metrik, delta=-0.176, p=0.246) |
| Gurultu direnci | PASS (ters yon), delta=-0.978 | **Tentatif PASS korunuyor** (n=8 mutlak teyitli; n=30 mutlak teyidi eksik) |
| Mackey-Glass | PASS (ters yon), delta=-0.431 | **GECERSIZ** (her iki ag da ogrenmemis) |
| Yol entegrasyonu | FAIL | **GECERSIZ** (ayni mimari kusur) |
| Gecikmeli eslestirme | FAIL | **GECERSIZ** (ayni mimari kusur) |
| Anomali tespiti | FAIL | **GECERSIZ** (ayni mimari kusur) |
| Looming/TTC | FAIL | **GECERSIZ** (ayni mimari kusur) |
| Capraz-gorev transfer | FAIL (n=8) | n=30'a genisletiliyor (protokol geregi) |
| Az-ornekli verimlilik | FAIL (n=8) | n=30'a genisletiliyor (protokol geregi) |

**Durust ozet:** gece bataryasinin "3 bagimsiz ters-yon PASS'i, saglamlik-
performans odunlesimi kesfettik" basligi ABARTILI cikti. Elde kalan
saglam bulgu: az sayida test (yalnizca gurultu direnci, tentatif) gercek
connectome'un bir eksende dezavantajli olabilecegini one suruyor, ama bu
gece bir "odunlesim teorisi" ilan etmek icin yeterli degil. Bes yeni gorev
kategorisi teknik bir kusur yuzunden hicbir sey kanitlamadi/curutmedi.
Ders: yeni bir gorev tasarimi her ne kadar makul gorunse de, ONAYLANMADAN
(triviyal taban-cizgi karsilastirmasi + pred.std()>0 kontrolu olmadan)
hicbir "sasirtici" sonuc resmi bulgu sayilmamali -- tipki bu projenin
er_null ve PSO-butce derslerinde daha once ogrendigi gibi.

---

## 2026-09-24 — RESMI SONUC: Protokol-uyum genisletmeleri tamamlandi -- 2 YENI GECERLI bulgu (n=30)

Yukaridaki duzeltmede atlandigi tespit edilen iki test, dogru protokolle
(magnitude-tetikli, guvenilir tek-atislik ray pipeline'i kullanarak)
n=30'a genisletildi.

**Az-ornekli veri verimliligi** (`fly_fewshot_efficiency_official30.py`,
sadece 6 egitim ornegiyle egitim, 300 epoch, buyuk held-out sette test):

```
n=30: real medyan=0.1774  null medyan=0.2028  Wilcoxon p=0.0000706  MW p=0.0144  delta=0.369  PASS
```

Gercek connectome, cok az ornekle (6 taneyle) egitildiginde bile null'dan
anlamli sekilde daha iyi genelliyor -- resmi sev-stabilitesi bulgusunun
(delta=0.884, 200 ornekle) "tam veri" versiyonundan bagimsiz, GENUINE bir
veri-verimliligi/az-ornekli-ogrenme avantaji.

**Capraz-gorev sifir-atis transferi** (`fly_crosstask_zeroshot_transfer_official30.py`,
sadece sev-stabilitesinde egitilmis agin, hic yeniden egitim yapmadan
kiris-tasarimi gorevinde dogrudan degerlendirilmesi):

```
n=30: real medyan=0.7154  null medyan=0.5670  Wilcoxon p=0.000153  MW p=0.00205  delta=-0.464  PASS (ters yon)
```

Burada gercek connectome null'dan anlamli sekilde DAHA KOTU -- sev-
stabilitesinde egitilmis gercek ag, kiris-tasarimina sifir-atis
transferinde null'dan daha basarisiz. **Bu iki sonuc bir arada tutarli
bir hikaye anlatiyor:** gercek connectome'un yapisi, egitildigi GOREV
SINIFI icin cok az ornekle bile guclu bir induktif onyargi/hizli ogrenme
sagliyor (az-ornekli avantaj), ama bu ozellesme, FARKLI bir gorev
sinifina egitimsiz aktarilirken bir maliyete donusuyor (zayif transfer).
Yani "genel amacli esneklik" degil, "hedeflenen goreve keskin
uzmanlasma" -- ki bu tam olarak evrimsel/biyolojik bir sinir agindan
beklenecek bir desen.

### Nihai duzeltilmis tablo (bu gecenin TUM sonuclari, ham veri dogrulamasi sonrasi)

| Test | n | delta | p (MW) | Durum |
|---|---|---|---|---|
| Lezyon direnci (X=20%) | 30 | -0.176 | 0.246 | FAIL (duzeltildi) |
| Gurultu direnci | 8 (mutlak) / 30 (oran) | -0.719 / -0.978 | 0.015 / <0.000001 | **PASS, tentatif** (n=30 mutlak teyidi bekliyor) |
| **Az-ornekli veri verimliligi** | **30** | **0.369** | **0.0144** | **PASS (gercek daha iyi)** |
| **Capraz-gorev sifir-atis transferi** | **30** | **-0.464** | **0.00205** | **PASS (gercek daha kotu)** |
| Yol entegrasyonu | 8 | -- | -- | GECERSIZ |
| Mackey-Glass | 30 | -- | -- | GECERSIZ |
| Gecikmeli eslestirme | 8 | -- | -- | GECERSIZ |
| Anomali tespiti | 8 | -- | -- | GECERSIZ |
| Looming/TTC | 8 | -- | -- | GECERSIZ |

**Bu gecenin durust bilancosu:** 9 orijinal test + 2 protokol-duzeltme
genisletmesi = 11 test calistirildi. 5'i mimari kusur yuzunden gecersiz.
Kalan 6'dan 3'u kapiyi geciyor: az-ornekli verimlilik (gercek DAHA IYI),
capraz-gorev transfer (gercek DAHA KOTU), gurultu direnci (gercek DAHA
KOTU, tentatif). Lezyon direnci ilk basta "DAHA KOTU" gibi gorundu ama
metrik artefakti oldugu ortaya cikinca FAIL'e dondu.

**Guncellenmis sentez:** dogru sonuc "performans-saglamlik odunlesimi"
degil, daha ince bir sey -- gercek connectome'un sev-stabilitesi/kiris-
tasarimi sinifindaki avantaji (a) cok az ornekle bile hizla ortaya
cikiyor (az-ornekli PASS), (b) ama bu sinifin DISINA (farkli bir fiziksel
gorev bile olsa) sifir-atis olarak tasinmiyor, aksine null'dan kotu
tasiniyor (transfer PASS, ters yon), (c) girdi gurultusune karsi kirilgan
olabilir (tentatif), (d) yapisal hasara karsi ozel bir kirilganligi YOK
(lezyon FAIL, duzeltildi). Bu, "genel-amacli hesaplama ustunlugu" degil,
"cok dar, cok keskin, egitim-sinifina asiri-uyumlu bir uzmanlasma"
tablosunu guclendiriyor -- sinir haritasindaki en dar noktayi bir kez
daha, ama bu sefer DOGRU yontemle teyit ediyor.

---

## 2026-09-24 — RESMI SONUC: Gurultu direncinin n=30 mutlak-deger teyidi tamamlandi -- artik TENTATIF DEGIL

Son acik uc kapatildi. `fly_noise_robustness_absolute_official30.py`,
seed 8-29'u AYNI egitim/degerlendirme protokolüyle ama bu sefer mutlak
`noisy`/`baseline` MSE degerlerini de diske yazarak yeniden calistirdi
(n=8 pilotun zaten mevcut mutlak degerleriyle birlestirilerek n=30):

```
n=30, MUTLAK gurultulu MSE: real medyan=0.2669  null medyan=0.1913
Wilcoxon p=3.7e-09  Mann-Whitney p=1.4e-09  Cliff's delta=-0.911  PASS
```

Oran-tabanli resmi rakamla (delta=-0.978) neredeyse ayni buyuklukte ve
ayni yonde -- lezyon direncinin aksine, bu sonuc bir metrik artefakti
DEGIL, gercek ve saglam. **Gurultu direnci artik tam olarak dogrulanmis
resmi bir bulgu (tentatif etiketi kaldirildi):** girdiye orantili (%15)
Gauss gurultusu eklendiginde, gercek connectome'un mutlak hatasi
null'dan (degree_preserving_rewire) anlamli sekilde daha fazla artiyor.

### Gecenin TAM ve NIHAI bilancosu (tum duzeltmeler sonrasi)

| Test | n | delta | p (MW) | Durum |
|---|---|---|---|---|
| **Gurultu direnci** | **30 (mutlak)** | **-0.911** | **1.4e-09** | **PASS (gercek daha kotu) -- TAM TEYITLI** |
| **Az-ornekli veri verimliligi** | **30** | **0.369** | **0.0144** | **PASS (gercek daha iyi)** |
| **Capraz-gorev sifir-atis transferi** | **30** | **-0.464** | **0.00205** | **PASS (gercek daha kotu)** |
| Lezyon direnci (X=20%) | 30 (mutlak) | -0.176 | 0.246 | FAIL (ilk PASS iddiasi metrik artefaktiydi) |
| Yol entegrasyonu / Mackey-Glass / gecikmeli eslestirme / anomali / looming | -- | -- | -- | GECERSIZ (paylasilan altyapida BFS-uyumsuzlugu kokenli mimari kusur, dokunulmadi) |

Bu gece toplam 13 test/genisletme calistirildi (9 orijinal + 2 protokol-
duzeltme + 1 lezyon-yeniden-analiz + 1 gurultu-mutlak-teyit), ikisi
(gurultu, ilk once lezyon sanildi) derinlemesine sorgulandi, biri (5
zamansal gorev toplu olarak) mimari kok nedenine kadar izlendi ve
PAYLASILAN ALTYAPIYA DOKUNULMADAN geriye birakildi. Uc GENUINE, tam
dogrulanmis yeni bulgu kaldi (gurultu direnci, az-ornekli verimlilik,
capraz-gorev transferi) -- ucu de sinir haritasini ayni yonde
keskinlestiriyor: gercek connectome'un avantaji COK dar bir gorev
sinifina keskin bicimde uzmanlasmis, bu uzmanlasma az veriyle bile hizla
ortaya cikiyor, ama disariya (baska bir goreve VEYA girdi gurultusune)
karsi kirilgan.

**Kullanicinin sorusuna ("sinegin baska huneri var mi?") nihai, durust
yanit:** hayir, test edilen hicbir YENI beceri kategorisinde (zamansal
entegrasyon, calisma bellegi, anomali tespiti, az-ornekli ogrenme haric,
capraz-gorev transferi haric) bir avantaj dogrulanmadi -- ama mevcut tek
huner (tek-atislik uzamsal regresyon) hakkinda uc onemli yeni ayrinti
ogrenildi: cok az veriyle bile calisiyor, baska goreve tasinmiyor, ve
girdi gurultusune karsi hassas. Bu gecenin gercek katkisi yeni bir "PASS"
degil, mevcut PASS'in etrafina cizilen cok daha net bir sinir.

**Gece bataryasi + dogrulama + protokol-duzeltmesi TAMAMEN TAMAMLANDI.**
Rapor artifact'i (flyopt_makale_2026-09.html) hala bilincli olarak
GUNCELLENMEDI -- kullanicinin sabah, bu gunlukteki tum duzeltme surecini
(ilk abartili iddialar + kendi kendine yapilan dogrulama + nihai
duzeltilmis sonuclar) gorup onaylamasi bekleniyor.

---

## 2026-09-24 (sabah) — 17 commit push edildi, makale v3 yayinlandi, yeni 4-testlik tur baslatildi

Kullanici uyandi, gece bataryasinin tum duzeltme surecini onayladi ("bu
kadar sağlam bir bilimsel duruş... Baş Araştırmacı'dan beklenen en üst
düzey bilimsel olgunluk"). 17 bekleyen commit `origin/master`'a push
edildi. Rapor artifact'i (flyopt_makale_2026-09.html) v3'e guncellendi:
yeni ozet/sonuc, yeni §3.8 (gece bataryasi + duzeltme sureci tam
seffaflikla), lezyon/gurultu ayrimi netlestirildi. Gemini'nin "F1 arabasi
toprak yolda traktörden kotu" benzetmesi kullaniciya iletilirken
yumusatildi (capraz-gorev transferi AYNI problem ailesinin komsu bir
uyesiydi, tamamen alakasiz bir alan degil) ve "sıradan ağların aylar
süren eğitimini saniyede geçiyor" iddiasi makaleye ALINMADI (hic egitim-
suresi veya standart mimari karsilastirmasi yapilmadi, asilsiz olurdu).

Kullanici "Faz 3" (MuJoCo/hexapod robotik) onerisine karsi kendi
verimizle (§3.1: kapali-dongu/cok-adimli FAIL) celisen bir varsayima
dayandigi icin temkinli yaklasilmasi onerildi; kullanici bunun yerine
"daha fazla sinir kesfedelim, nerede kullanildigini gorelim" dedi. Kisa
bir web-arastirmasi (ayri bir subagent) konnektom-tabanli hesaplamanin
hala tamamen akademik oldugunu, hicbir ticari uruncin literal haritalanmis
connectome kullanmadigini dogruladi (Opteran Technologies bile boecek
devre *prensiplerinden* esinleniyor, literal connectome degil) -- bu bizim
"dar, gorev-spesifik avantaj" bulgumuzla tutarli.

**Yeni 4 test (kullanici: "hepsini yap"), kapali-dongu tuzagina
DUSMEYECEK sekilde secildi:**

1. **Ucuncu fiziksel gorev (basincli kap/silindir tasarimi):** yeni
   `benchmarks_vessel.py` -- sev-stabilitesi (limit-denge) ve kiristen
   (egilme gerilmesi) mekanik olarak farkli bir arıza modu (ince-cidarli
   silindirde cevresel/hoop gerilme). Egitim ONCESI dogrulama yapildi
   (gecersiz-nokta orani %25, gradyan dejenere degil -- gecen geceki
   dersin dogrudan uygulamasi).
   ```
   n=8: real medyan=0.1793  null medyan=0.1857  MW p=0.959  delta=-0.031  FAIL
   ```
   **Onemli sonuc: tam berabere.** MSE degerleri (0.14-0.29) trivial
   "sifir tahmin et" tabanindan (~0.5-1.0) belirgin iyi -- yani aglar
   gercekten ogreniyor, bu dejenere bir cokus DEGIL, gercek bir FAIL.
   Sonuc: "dar sinif" iddiasi sev-stabilitesi + kiristen daha da dar
   olabilir -- 2/3 fiziksel gorevde avantaj var, ucuncude yok.

2. **Girdi-olcegi (sensing radius) genellemesi:** resmi sev-stabilitesi
   agi (egitim RAY_RADIUS=1.0), hic yeniden egitilmeden farkli bir
   yaricapta sifir-atis test edildi.
   ```
   n=8, r=0.1 (10x kucuk): delta=0.500, p=0.105 -- esigi gecti, n=30'a genisletiliyor
   n=8, r=1.0 (referans):  delta=1.000, p=0.0002, PASS (beklendigi gibi)
   n=8, r=5.0 (5x buyuk):  delta=0.906, p=0.0011, PASS
   ```
   **En temiz yeni bulgu:** avantaj, egitim yaricapinin 5 KATI farkli bir
   duyu olceginde bile guclu sekilde korunuyor -- once test edilen
   diger genellenemezlik eksenlerinin (gorev, tur, birey) aksine, bu
   eksen (girdi OLCEGI) avantaji tasiyor gibi gorunuyor.

3. **Agirlik kuantizasyonu direnci** (donanim/nöromorfik dagitim
   sorusuyla ilgili): ayni resmi ag, cikarim-zamaninda N-bit sabit-nokta
   kuantize edildi. Fonksiyon egitim ONCESI dogrulandi (taban MSE dogru
   restore ediliyor, bkz. smoke-test).
   ```
   n=8, kuantizasyonsuz taban: delta=1.000, PASS (beklendigi gibi)
   n=8, 8-bit: delta=-0.281, p=0.382, FAIL
   n=8, 4-bit: delta=0.000, p=1.000, FAIL (tam berabere)
   n=8, 2-bit: delta=0.469, p=0.130 -- esigi gecti, n=30'a genisletiliyor
   ```
   **Onemli sonuc:** tam hassasiyette devasa avantaj (delta=1.0), ama
   EN HAFIF kuantizasyonda bile (8-bit) tamamen buharlasip gurultu
   seviyesine dusuyor. Lezyon-direnci deneyiyle ayni desen: avantaj tam
   float32 hassasiyetine bagimli, donanima (dusuk-bit nöromorfik cipler
   dahil) dogrudan tasinmasi beklenmemeli.

4. **Alt-graf boyutu duyarliligi:** resmi 3000-nöronluk secimin 1000,
   6000 ve 10000 boyutlarinda da tutup tutmadigi test ediliyor (calisiyor,
   sonuc bekleniyor).

Uc yeni genisletme (2-bit kuantizasyon n=30, r=0.1 n=30) ve alt-graf
boyutu taramasi (4 boyut x n=8) su an arka planda paralel calisiyor.

---

## 2026-09-24 — Altyapi notu: 3 paralel is GPU VRAM'ini tikadi, saatlerce donuk kaldi, kok neden bulunup duzeltildi

Yukaridaki 3 is paralel baslatildiktan sonra saatlerce (10:11'den 13:21'e
kadar, ~3 saat) hicbir ilerleme kaydetmedi -- dosya zaman damgalari
donuktu ama surecler "calisiyor" gorunuyordu. `nvidia-smi` ile teshis
edildi: GPU (8GB VRAM, RTX 4060) 7931/8188 MB doluydu -- 3 paralel
egitim sureci ARTI kullanicinin es-zamanli acik olan Tekla Structures
(yapisal muhendislik CAD yazilimi) uygulamasi VRAM'i tuketmisti,
muhtemelen bellek-baskisi/thrashing kaynakli bir donmaya yol acti. 3
surec `Stop-Process` ile durduruldu (dogru PID'ler `Get-CimInstance
Win32_Process` ile komut satirlariyla dogrulanarak tespit edildi,
Tekla'ya dokunulmadi), GPU bellegi 420 MB'a dustu (dogrulandi), ve 3 is
bu kez PARALEL degil SIRAYLA yeniden baslatildi -- hem tekrar
tikanmayi onlemek hem de kullanicinin es-zamanli GPU kullanimina saygili
olmak icin. Ders: bu makinede (paylasilan 8GB VRAM, kullanicinin kendi
uygulamalari da GPU kullanabiliyor) coklu-paralel GPU egitim isi
baslatmadan once mevcut VRAM'i kontrol etmek gerekiyor.

**2-bit kuantizasyon n=30 (yeniden, sirayla) SONUCLANDI:**
```
n=30: real medyan=0.2208  null medyan=0.2259  MW p=0.206  Wilcoxon p=0.135  delta=0.191  FAIL
```
n=8'deki esik-asan sinyal (delta=0.469) gurultuymus -- n=30'da temiz bir
FAIL'e donustu. Kuantizasyon hikayesi artik tam kapandi: 8-bit (FAIL),
4-bit (FAIL, tam beraberlik), 2-bit (FAIL, n=30 ile dogrulandi) -- hicbir
kuantizasyon seviyesinde avantaj yok veya geri gelmiyor.

**Girdi-olcegi (r=0.1) n=30 (GPU catismasi nedeniyle iki kez yeniden
baslatildi, sonunda tamamlandi) SONUCLANDI:**
```
n=30: real medyan=0.1440  null medyan=0.1727  MW p=5.46e-06  Wilcoxon p=1.64e-07  delta=0.684  PASS
```
n=8'deki esik-asan ama anlamsiz sinyal (delta=0.500, p=0.105) n=30'da
GERCEK ve saglam bir PASS'e donustu. Simdi girdi-olcegi genellemesi
IKI yonde de tam dogrulandi: r=5.0 (5x buyuk, n=8 PASS) VE r=0.1 (10x
kucuk, n=30 PASS) -- egitim yaricapinin (r=1.0) 50 KATLIK bir araligi
kapsayan olcek-degisimine karsi avantaj tutarli sekilde korunuyor. Bu,
gecenin/sabahin EN TEMIZ yeni bulgusu: connectome'un avantaji ne
cok-adimli kontrole, ne farkli bir goreve, ne agirlik kuantizasyonuna,
ne de yapisal hasara dayanikli -- ama GIRDI OLCEGINE (sensing radius)
saglam sekilde dayanikli. Bu, avantajin mekanizmasinin olcek-bagimsiz
bir yon/gradyan-tahmini prensibi olabilecegine, spesifik bir sayisal
araliga ezberlenmis olmadigina isaret ediyor.

Sirada: alt-graf boyutu duyarliligi taramasi (1000/3000/6000/10000
noron, n=8 her boyut) baslatiliyor.

---

## 2026-09-24 — RESMI SONUC: Alt-graf boyutu duyarliligi -- 4 boyutun HEPSINDE PASS

**Faz:** GPU catismasi nedeniyle bir kez donup yeniden baslatildi (bkz.
yukaridaki altyapi notu), sonunda tamamlandi. Ayni encode/decode tohumu
(9000), ayni egitim tarifi, SADECE `SUBGRAPH_SIZE` degisiyor.

```
n=8, boyut=1000  (resmi 3000'in 1/3'u):  real med=0.1008 null med=0.1585 MW p=0.00016 delta=1.000 PASS
n=8, boyut=3000  (resmi referans):        real med=0.1204 null med=0.1749 MW p=0.00016 delta=1.000 PASS
n=8, boyut=6000  (resmi 3000'in 2 kati):  real med=0.1254 null med=0.1802 MW p=0.00295 delta=0.844 PASS
n=8, boyut=10000 (resmi 3000'in 3+ kati): real med=0.1506 null med=0.1774 MW p=0.00466 delta=0.813 PASS
```

**Cok temiz, tek yonlu bir sonuc: DORT boyutun DA HEPSI kapiyi geciyor.**
delta, boyut buyudukce hafifce zayifliyor (1.000 -> 1.000 -> 0.844 ->
0.813) ama her boyutta hala esigin (0.33) cok uzerinde ve p<0.005. Bu,
resmi delta=0.884 bulgusunun "tam 3000 noron" secimine ozgu sansli bir
sonuc OLMADIGINI kesin olarak gosteriyor -- avantaj 10 kati bir boyut
araliginda (1000-10000 noron) saglam. Hafif zayiflama trendi ilginc bir
ek gozlem (belki daha buyuk alt-graflar daha fazla "gorevle alakasiz"
noron iceriyor, belki null modelin serbestlik derecesi buyudukce artiyor)
ama spekulatif, test edilmedi.

### Sabah/gece turunun (4 yeni sinir testi) NIHAI ozet tablosu

| Test | n | delta | p (MW) | Durum |
|---|---|---|---|---|
| **Girdi-olcegi (r=5.0, 5x buyuk)** | 8 | 0.906 | 0.0011 | **PASS** |
| **Girdi-olcegi (r=0.1, 10x kucuk)** | 30 | 0.684 | 5.46e-06 | **PASS** |
| **Alt-graf boyutu (1000/3000/6000/10000)** | 8 (her biri) | 1.00/1.00/0.84/0.81 | hepsi <0.005 | **PASS (4/4 boyut)** |
| Ucuncu fiziksel gorev (basincli kap) | 8 | -0.031 | 0.959 | FAIL (aciklanmamis) |
| Agirlik kuantizasyonu (8/4/2-bit) | 8-30 | -0.28/0.00/0.19 | hepsi >0.13 | FAIL (tum seviyelerde) |

**Sentez:** bu turun en carpici bulgusu, avantajin GIRDI OLCEGINE ve ALT-
GRAF BOYUTUNA karsi son derece SAGLAM olmasi -- bu ikisi, sinir
haritasinda simdiye kadar test edilen ve avantajin TUTARLI SEKILDE
KORUNDUGU ilk iki eksen (digerleri: gorev tipi, tur, birey, hasar,
gurultu, kuantizasyon, transfer -- hepsi FAIL veya ters yon). Yani
connectome'un avantaji "3000 norona ya da r=1.0 olcegine ezberlenmis" bir
sey degil, bu iki boyutta gercekten GENELLENEN bir yapisal ozellik. Ayni
zamanda ucuncu fiziksel gorev (basincli kap) hala aciklanamadan FAIL
kaliyor ve kuantizasyon hikayesi tam kapandi (hicbir seviyede avantaj
yok/geri gelmiyor).

Sirada: girdi-olcegi genellemesinin (r=0.1) modulerlik tarafindan
aciklanip aciklanmadigini test eden bir mekanizma-analizi
(`fly_rayradius_nullfamily_mechanism.py`, null-model ailesi taramasi)
baslatiliyor.

---

## 2026-09-24 — RESMI SONUC: Girdi-olcegi genellemesinin mekanizmasi -- AYNI modulerlik hikayesi tekrarlaniyor

**Faz:** `fly_rayradius_nullfamily_mechanism.py` (community_preserving_rewire
cagrisinda eksik `communities` argumani nedeniyle bir kez cöktu --
`fly_slope_stability_communitypreserving_official30.py`'daki
`compute_real_communities` (Louvain, seed=0) ile duzeltilip
`_part2.py`'de tamamlandi, gercek-ag MSE'leri yeniden hesaplanmadan
ilk kosumdan tasindi). Ana slope-stability mekanizma analiziyle (delta=
0.884 -> community_preserving'de 0.429) BIREBIR AYNI null-model ailesi,
ama bu sefer r=0.1 sifir-atis genelleme kosulu icin.

```
n=8, r=0.1 zero-shot, null-model ailesi:
degree_preserving:      real med=0.1457  null med=0.1816  MW p=0.00295  delta=0.844  PASS
scale_free:             real med=0.1457  null med=0.1797  MW p=0.00466  delta=0.813  PASS
scale_free_correlated:  real med=0.1457  null med=0.1718  MW p=0.04988  delta=0.594  PASS (sinirda)
community_preserving:   real med=0.1457  null med=0.1647  MW p=0.38228  delta=0.281  FAIL
weight_shuffle:         real med=0.1457  null med=0.1567  MW p=0.27863  delta=0.344  FAIL (esigi zar zor geciyor, p cok uzak, n=30 genisletmesi ONERILIR ama kritik degil)
```

**Ana bulgunun mekanizmasiyla BIREBIR AYNI orunt:** modulerligi de koruyan
null (community_preserving) devreye girince avantaj (delta=0.844) neredeyse
tamamen cokuyor (delta=0.281, p=0.38, acikca anlamsiz). Bu, girdi-olcegi
genellemesinin de AYNI yapisal kaynaktan (agin modulerlik/kumesel
organizasyonu) geldigini gosteriyor -- iki bagimsiz olarak kesfedilen
bulgu (ana delta=0.884 PASS ve girdi-olcegi genellemesi PASS) ayni
mekanizmaya bagli. Bu, sinir haritasina guclu bir tutarlilik katiyor:
modulerlik sadece performans avantajini degil, o avantajin GIRDI
OLCEGINE karsi saglamligini da tasiyor gibi gorunuyor.

### Guncellenmis mekanizma tablosu (iki bulgu yan yana)

| Null model | Ana bulgu (delta) | Girdi-olcegi r=0.1 (delta) |
|---|---|---|
| degree_preserving | 0.884 | 0.844 |
| scale_free | 0.738 | 0.813 |
| scale_free_correlated | 0.604 | 0.594 |
| community_preserving | 0.429 | 0.281 |
| weight_shuffle | 0.258 (FAIL) | 0.344 (FAIL) |

Iki sutun carpici derecede paralel -- ayni null modelde ayni yonde
zayifliyor, ayni null modelde (community_preserving) esigin altina
dusuyor. Bu paralellik, "girdi-olcegi genellemesi" ve "ana tek-atislik
avantaj"in AYNI tek bir yapisal mekanizmanin iki farkli tezahuru
oldugunu güclu sekilde destekliyor -- iki ayri/bagimsiz mekanizma degil.

**weight_shuffle n=30 (protokol geregi, esik-siniri sinyal icin) SONUCLANDI:**
```
n=30: real medyan=0.1435  null medyan=0.1544  MW p=0.0724  Wilcoxon p=0.0087  delta=0.271  FAIL
```
n=8'deki esik-asan sinyal (delta=0.344) yine gurultuymus -- n=30'da
temiz bir FAIL'e dondu (tipki 2-bit kuantizasyonda oldugu gibi). Mekanizma
tablosu artik TAM ve KAPALI:

| Null model | Ana bulgu (n=30) | Girdi-olcegi r=0.1 (n=8/30 karisik) |
|---|---|---|
| degree_preserving | 0.884 PASS | 0.844 PASS (n=8) |
| scale_free | 0.738 PASS | 0.813 PASS (n=8) |
| scale_free_correlated | 0.604 PASS | 0.594 PASS (n=8, sinirda) |
| community_preserving | 0.429 PASS | 0.281 FAIL (n=8) |
| weight_shuffle | 0.258 FAIL | 0.271 FAIL (**n=30, dogrulandi**) |

Iki sutun uctan uca tutarli -- ayni null'larda ayni yonde zayifliyor, ayni
iki null'da (community_preserving, weight_shuffle) esigin altina
dusuyor. **Girdi-olcegi genellemesi ve ana tek-atislik avantaji, ayni
yapisal mekanizmaya (agin modulerlik/kumesel organizasyonu) bagli --
bagimsiz iki mekanizma degil, tek bir mekanizmanin iki farkli
tezahuru.**

---

## 2026-09-24 — Basincli kap FAIL'inin arastirilmasi: bir hipotez kismen dogrulandi, kismen curutuldu

Kullanici: "ucuncu gorevin neden basarisiz oldugunu arastir." Adayi
netlestirmek icin once vessel gorevinin parametre olceklerini diger
gorevlerle karsilastirdim:

```
R (yaricap): araligin sadece %0.87'si kadar adim -- ray_radius=0.01 cok ince
t (kalinlik): araligin %12.66'si kadar adim -- ray_radius=0.01 cok kaba (araligin 1/8'i!)
kiris (b,h): ikisi de %3.5, simetrik
sev (3 param): %1.3-1.9 arasi, nispeten simetrik
```

t ekseni icin adim, diger tum gorevlerden 4-10 kat daha kaba -- bu,
sonlu-fark probunun t yonunde gercekten "lokal" olmayabilecegi
hipotezini dogurdu. Dogrudan test: 100 rastgele noktada, ray_radius=0.01
ile hesaplanan ogretmen yonunu, cok kucuk bir yaricapla (1e-6) hesaplanan
"gercek" yerel gradyana kosinus benzerligiyle karsilastirdim:

```
radius=0.01:  ortalama kosinus-benzerligi=0.846  std=0.507  min=-0.994 (bazen TAM TERS!)
radius=0.001: ortalama kosinus-benzerligi=0.964  std=0.252  min=-0.797
```

Bu, ogretmen sinyalinin radius=0.01'de sik sik cok bozuk (bazen ters
yonde bile) oldugunu dogruladi. Hipotez: bu kotu ogretmen sinyali HER
IKI agi da (gercek ve null) esit sekilde bozarak, gercek bir avantaj
varsa bile onu gurultunun altina gomuyor olabilir.

**Dogrudan test (n=8, `fly_vessel_design_smallradius_multiseed.py`,
RAY_RADIUS=0.001):**
```
real medyan=0.0201  null medyan=0.0202  MW p=0.798  Wilcoxon p=0.383  delta=0.094  FAIL
```

**Sonuc: hipotez KISMEN dogrulandi, KISMEN curutuldu.** Mutlak hata
degerleri cok dustu (0.14-0.29 -> 0.002-0.04 araligina) -- yani gorev
gercekten cok daha iyi ogreniliyor, ilk teshis (kaba yaricap = kotu
ogretmen sinyali) DOGRUYDU. Ama gercek-null farki HALA yok (delta=0.094,
orijinal delta=-0.031'e cok yakin) -- yani "kotu ogretmen sinyali"
GAP'in KAYBOLMA nedeni DEGIL. Duzeltilmis, cok daha temiz bir ogretmen
sinyaliyle bile connectome hicbir avantaj gostermiyor.

**Guncellenmis durum:** basincli kap FAIL'i icin en makul aday
aciklama (parametre-olcegi/ray_radius uyumsuzlugu) elendi. Gorevin
kendisi (ince-cidarli silindir, cevresel gerilme) connectome'un
avantajli oldugu diger iki gorevden (sev, kiris) gercekten farkli bir
seyler istiyor olabilir -- ama bu "seyler" ne oldugu hala bilinmiyor.
Baska adaylar (henuz test edilmedi): hedef-fonksiyonun egrilik/kosullanma
yapisi, R ve t arasindaki cok farkli hassasiyet (P*R/t teriminde t'nin
payda'da olmasi -- 1/t hassasiyeti diger gorevlerde gorulmeyen bir
nonlineerlik), ya da basitce bu 3. gorevin sansli/sanssiz bir alt-graf
etkilesimi. Spekulasyon yapmadan "acik soru" olarak birakiliyor.

---

## 2026-09-24 — Dorduncu fiziksel gorev: Euler kolon burkulmasi -- FAIL, ama muhtemel aciklamasi var

Kullanici: "bir fiziksel problem ara onu da test edelim." Istatistiksel
gucu artirmak icin (o ana kadar 2 PASS / 1 FAIL) yeni, mekanik olarak
GERCEKTEN farkli bir ariza modu secildi: **Euler elastik burkulma
kararsizligi** (eksenel basinc altinda ani stabilite kaybi -- egilme
gerilmesi/kiris, cevresel gerilme/vessel, veya limit-denge/sev'den hicbiri
degil). `benchmarks_column.py`, x=(b,h) dikdortgen kesit, kiris'in AYNI
simetrik [0.03,0.6] kutusunu kullanarak vessel'in R/t olcek-uyumsuzlugu
hatasini bilerek TEKRARLAMAMAK icin tasarlandi.

**Egitim ONCESI dogrulama (vessel dersinin dogrudan uygulamasi):**
```
gecersiz-nokta orani: %0.0
maliyet araligi: 2.3e4 - 1.0e7 (diger gorevlerle ayni mertebede)
ogretmen-sinyal kalitesi (kosinus-benzerligi, tum adaylarda): 1.0000, std=0.0000
```
Ogretmen sinyali MUKEMMEL kalitede (vessel'deki 0.85/std=0.51 sorunuyla
tam tersi) -- bu goreve guvenle gecildi.

**n=8 sonucu:**
```
real medyan=0.0653  null medyan=0.0700  MW p=0.599  Wilcoxon p=0.125  delta=0.172  FAIL (esik altinda, n=30 GEREKMEZ)
```

**Ilginc bir yan-gozlem:** 8 tohumun 7'sinde gercek ve null agin MSE'si
neredeyse BIREBIR ayni (ornek: seed=0, real=0.071810655 vs
null=0.071810752 -- 7. ondalige kadar ozdes), sadece 1 tohumda (seed=3)
belirgin fark var. Muhtemel aciklama: ogretmen sinyalinin mukemmel
kalitesi (kosinus-benzerligi=1.0) bu gorevin COK KOLAY/iyi-kosullanmis
oldugunu gosteriyor olabilir -- hemen hemen HER makul ag (yapisindan
bagimsiz) neredeyse en iyi coze yakinsiyor, yapisal bir avantaja yer
kalmiyor ("tavan etkisi"). Bu, vessel'in aciklanamayan gizeminden
FARKLI bir durum -- burada makul, test edilebilir bir aday aciklama var
(dogrulanmadi ama spekulasyon degil, dogrudan olcumden cikan bir
gozlem).

### Guncellenmis fiziksel-gorev istatistigi: 2 PASS / 2 FAIL (4 gorev)

| Gorev | Ariza modu | DIM | delta | Durum |
|---|---|---|---|---|
| Sev stabilitesi | limit-denge (kayma) | 3 | 0.884 | PASS |
| Kiris tasarimi | egilme gerilmesi | 2 | 0.613 | PASS |
| Basincli kap | cevresel (hoop) gerilme | 2 | -0.031 | FAIL (aciklanamadi) |
| Kolon burkulmasi | elastik kararsizlik | 2 | 0.172 | FAIL (muhtemelen "tavan etkisi") |

4 fiziksel gorevden 2'si PASS -- "dar sinif" sadece 2 spesifik gorevin
sansli bir rastlantisi degil ama her fiziksel muhendislik goreviyle de
otomatik calismiyor. Ornek boyutu hala kucuk (n=4 gorev), kesin bir
desen (orn. "hangi ariza modlari calisir") cikarmak icin yetersiz.

---

## 2026-09-24 — DUZELTME: Kolon burkulmasi FAIL'i gecersizdi, kalibrasyon hatasi bulundu ve duzeltildi -- ASIL SONUC PASS

Kullanici "başka bir fiziksel problem bul" dedi. Besinci gorevi
(kiris SEHIMI/servis-edilebilirlik, benchmarks_deflection.py) egitim
oncesi dogrulama sirasinda cok tanidik sayilar cikinca (kolon'un
diagnostik cikisina neredeyse BIREBIR ayni: gecersiz-oran 0.000,
maliyet araligi 5-6 basamaga kadar ozdes) supheleneip geri donup
kolon'u yeniden inceledim.

**Kok neden bulundu:** her iki gorevde de (kolon VE sehim) maliyet
terimi (cost_weight*b*h) yapisal terimi (P/Pcr veya sehim/izin-verilen)
TAMAMEN eziyordu -- oran payi tipik noktalarda **%0.00**. Sebep: Euler
burkulma yuku (Pcr) ve sehim, kesit derinligine (h) KUP ile orantili
hassasiyet gosteriyor (I~h^3), ve orijinal [0.03,0.6] kutusu bu terimi
5 mertebeye varan bir araliga yayiyordu -- sabit bir yuk/maliyet-agirligi
araligi bu genislikte asla dengelenemedi. Sonuc: ag sadece "b,h'yi
kuculte" gibi trivial, yapisal hesap gerektirmeyen bir yonu ogreniyordu
-- **kolon'un ilk raporlanan FAIL'i (delta=0.172) GECERSIZDI**, connectome
hakkinda hicbir sey soylemiyordu (5 gecersiz gorevle ayni ailede bir
hata, ama farkli kok neden: bu sefer BFS-baglanti degil, terim-olcegi
dengesizligi).

**Duzeltme:** (b,h) kutusu [0.1,0.3]'e daraltildi (h^3 araligi 8000x'ten
27x'e indi) VE yuk, sabit bir aralik yerine REFERANS kesitin kapasitesinin
bir orani olarak turetildi. Dogrulama: oran-payi medyani %0.00 -> %36
(kolon) / %28 (sehim), ogretmen-sinyal kalitesi (kosinus-benzerligi)
~0.99-1.00 korundu.

**Duzeltilmis kolon burkulmasi sonucu (n=8):**
```
real medyan=0.2643  null medyan=0.4589  MW p=0.00295  Wilcoxon p=0.00781  delta=0.844  PASS (esigi asti, n=30'a genisletiliyor)
```

**Bu, ilk raporlanan FAIL'in tam tersi -- kolon burkulmasi ASLINDA
GUCLU bir PASS.** Fiziksel gorev istatistigi yeniden degisti: **3 PASS
(sev, kiris, kolon) / 1 aciklanamamis FAIL (vessel)**, n=30 kesinlesirse.

### Ders: kendi kurdugum bir gorevin "kolay" gorunmesi bir bulgu degil, bir kirmizi bayrak olabilir

Onceki kolon FAIL'i icin verdigim "tavan etkisi" aciklamasi (gorev cok
kolay, her ag yakinsıyor) KULAGA MAKUL geliyordu ama YANLISTI -- gercek
sebep cok daha sikici (bir terim digerini trivial sekilde eziyordu).
Bu, projenin er_null/PSO-butce derslerinin bir kez daha teyidi: her
"kolay/temiz" gorunen sonuc, dogru aciklamayi bulana kadar supheyle
karsilanmali. Bu sefer fark ILGISIZ bir gorevin (sehim) diagnostik
ciktilarinin tesaduf-otesi benzerligi sayesinde yakalandi -- bagimsiz
bir gorevi ayni sekilde insa etmek, ilkindeki gizli bir hatayi ortaya
cikarmak icin beklenmedik bir capraz-kontrol oldu.

### RESMI: Kolon burkulmasi n=30 dogrulandi

```
n=30: real medyan=0.2470  null medyan=0.4666  MW p=5.00e-09  Wilcoxon p=1.86e-08  delta=0.88  PASS
```

Neredeyse ana bulguyla (delta=0.884) ayni buyuklukte -- kolon burkulmasi
artik ANA BULGUYLA ESDEGER GUCTE, tam anlamiyla resmi, dort-bagimsiz-
seviyede dogrulanmis (n=8 -> kalibrasyon hatasi bulundu -> duzeltildi ->
n=8 tekrar -> n=30) ucuncu fiziksel PASS. Fiziksel gorev istatistigi:
**3 PASS (sev delta=0.884, kiris delta=0.613, kolon delta=0.88) / 1
aciklanamamis FAIL (vessel delta=-0.031)**.

Sirada: besinci gorev (kiris sehimi/servis-edilebilirlik,
`fly_deflection_multiseed.py`, ayni kalibrasyon dersi ONCEDEN
uygulanmis haliyle) test ediliyor.

---

## 2026-09-24 — Besinci fiziksel gorev: kiris sehimi (servis-edilebilirlik) -- MUKEMMEL PASS

Kalibrasyon dersi (kutu daraltma + referans-kesit oranli yuk turetme)
ONCEDEN uygulanarak tasarlanan besinci gorev, ilk denemede tam ayrisma
verdi:

```
n=8: real medyan=0.0292  null medyan=0.2820  MW p=0.000155  Wilcoxon p=0.00781  delta=1.000  PASS (mukemmel ayrisma, n=30'a genisletiliyor)
```

Gercek agin hatasi (0.024-0.114) ile null'unki (0.22-0.49) arasinda
HIC ortusme yok. Bu, sev-stabilitesi disindaki EN GUCLU pilot sonuc.
n=30'a genisletiliyor.

### RESMI: Kiris sehimi n=30'da MUKEMMEL PASS -- projenin EN GUCLU sonucu

```
n=30: real medyan=0.0349  null medyan=0.3372  MW p=3.02e-11  Wilcoxon p=1.86e-09  delta=1.000  PASS (tam ayrisma)
```

**30 tohumun HEPSINDE gercek agin hatasi (0.016-0.114) ile null'unki
(0.19-0.49) arasinda SIFIR ortusme.** Bu, ana sev-stabilitesi bulgusunu
(delta=0.884) bile gecen, PROJENIN SIMDIYE KADARKI EN GUCLU, EN TEMIZ
sonucu.

### Fiziksel gorev istatistigi: 4 PASS / 1 aciklanamamis FAIL (5 gorev)

| Gorev | Ariza modu / kriter | delta (n=30) | Durum |
|---|---|---|---|
| Sev stabilitesi | limit-denge (kayma) | 0.884 | PASS |
| **Kiris sehimi** | **servis-edilebilirlik (sehim)** | **1.000** | **PASS (mukemmel)** |
| Kolon burkulmasi | elastik kararsizlik | 0.88 | PASS |
| Kiris tasarimi | egilme gerilmesi (mukavemet) | 0.613 | PASS |
| Basincli kap | cevresel (hoop) gerilme | -0.031 | FAIL (aciklanamadi) |

**Dogru kalibre edildiginde (terim dengesi + olcek simetrisi
korunarak), avantaj tek-atislik fiziksel/uzamsal regresyon gorevlerinin
BUYUK COGUNLUGUNDA (5 gorevden 4'unde) tutarli ve guclu sekilde ortaya
cikiyor.** Vessel artik "3'te 1 FAIL" degil, "5'te 1, giderek daha
belirgin bir istisna" gibi goruunuyor -- kendi basina aciklanmamis
kalsa da, genel oruntu ("dar sinif" = tek-atislik fiziksel yon
regresyonu) daha once dusunulenden cok daha GENIS ve SAGLAM cikti.

Bu gecenin/gunun en onemli metodolojik dersi: **kolon burkulmasinin ilk
FAIL'i (delta=0.172) tam bir kalibrasyon hatasiydi, duzeltilince
delta=0.88'e donustu.** Vessel'in FAIL'i de benzer bir kalibrasyon
sorunundan kaynaklanabilir mi sorusu artik cok daha guclu bir sekilde
acik kaliyor -- ayni "terim dengesi" kontrolu vessel_cost'a da
uygulanmali (henuz yapilmadi, bir sonraki adim olarak not edildi).

---

## 2026-09-24 — GIZEM COZULDU: Basincli kap FAIL'i de kalibrasyon hatasiymis -- duzeltilince MUKEMMEL PASS

Vessel'in terim dengesi kontrol edildi: hoop-stress terimi maliyet
terimini tipik noktalarda **%97.4** payla eziyordu (kolon'un ilk
hatasinin AYNASI -- orada maliyet, burada fizik terimi domine ediyordu).
`cost_weight` araligi [5e6,8e7] -> [5e8,4e9] olarak yeniden kalibre
edildi (hoop-stress payi medyani artik %44.4). Ayrica bu yeniden
dengeleme, R/t olcek-asimetrisini de yeniden gun yuzune cikardi --
radius=0.01'de ogretmen sinyali hala kotuydu (ortalama kosinus-
benzerligi=0.92, min=-0.92); radius=0.001'e dusurulunce mukemmele cikti
(ortalama=0.9998, min=0.98).

**Yeniden kalibre edilmis vessel testi (n=8, `fly_vessel_design_recalibrated_multiseed.py`):**
```
real medyan=0.0375  null medyan=0.2178  MW p=0.000155  Wilcoxon p=0.00781  delta=1.000  PASS (tam ayrisma, n=30'a genisletiliyor)
```

**GIZEM TAMAMEN COZULDU.** Basincli kap tasariminin ilk raporlanan
FAIL'i (delta=-0.031, "acikianamamis" olarak isaretlenmisti) TAMAMEN
bir kalibrasyon artefaktiydi -- connectome hakkinda hicbir sey
soylemiyordu. Duzeltilince kiris sehimi gibi mukemmel bir ayrisma
veriyor.

### NIHAI fiziksel gorev istatistigi: 5/5 PASS

| Gorev | Ariza modu / kriter | delta (n=8, vessel haric n=30) |
|---|---|---|
| Sev stabilitesi | limit-denge (kayma) | 0.884 (n=30) |
| **Kiris sehimi** | servis-edilebilirlik | **1.000 (n=30)** |
| **Basincli kap** | **cevresel (hoop) gerilme** | **1.000 (n=8, n=30 bekleniyor)** |
| Kolon burkulmasi | elastik kararsizlik | 0.88 (n=30) |
| Kiris tasarimi | egilme gerilmesi | 0.613 (n=30) |

**Test edilen BES fiziksel/muhendislik gorevinin BESI de, dogru kalibre
edildiginde, guclu ve tutarli bir connectome avantaji gosteriyor.**
"Dar sinif" hipotezi (tek-atislik, fiziksel/uzamsal yon regresyonu)
artik cok daha genis bir dogrulukla destekleniyor -- onceki "2/3" veya
"belirsiz" tablo, byuk olcude METODOLOJIK HATALARIN (terim dengesizligi,
olcek uyumsuzlugu) urunuymus, connectome'un gercek sinirlarinin degil.

**Bu gecenin/gunun en buyuk dersi:** iki bagimsiz "acilanamayan FAIL"
(kolon, vessel) ikisi de ayni kok-neden ailesinden (terim dengesizligi,
farkli yonlerde) cikti ve ikisi de duzeltilince en guclu PASS'lere
donustu. Bu, projenin sinir haritasinin BUYUK bolumunun ("avantaj
FAIL ediyor" olarak isaretlenen eksenlerin bir kismi) aslinda test
tasarimi hatalarindan kaynaklanmis olabilecegine dair guclu bir uyari --
ozellikle YENI, henuz tam dogrulanmamis gorevler icin. Onceden dogrulanmis
GENELLENEMEYEN eksenler (tur, birey, gorev-TIPI degisikligi, girdi
gurultusu, kuantizasyon) icin bu risk daha dusuk (cunku onlar farkli bir
GOREVI degil, AYNI dogrulanmis fiziksel gorevi farkli kosullarda test
ediyordu) -- ama ileride ekelenecek her YENI fiziksel/matematiksel gorev
icin, artik STANDART pratik: onceden terim-dengesi + olcek-kontrolu +
ogretmen-sinyal-kalitesi ucuzlu dogrulamasi ZORUNLU.

### RESMI: Basincli kap n=30'da dogrulandi -- 5/5 fiziksel gorev artik TAM RESMI PASS

```
n=30: real medyan=0.0288  null medyan=0.1580  MW p=7.12e-09  Wilcoxon p=1.86e-09  delta=0.871  PASS
```

n=8'deki mukemmel delta=1.000'den biraz dustu (bazi null tohumlari
dusuk cikti) ama hala cok guclu ve tam anlamda resmi. **NIHAI, TAM
DOGRULANMIS fiziksel gorev tablosu:**

| Gorev | Ariza modu / kriter | delta (n=30) | p (MW) |
|---|---|---|---|
| **Kiris sehimi** | servis-edilebilirlik | **1.000** | 3.02e-11 |
| Sev stabilitesi | limit-denge (kayma) | 0.884 | ≈0 |
| **Basincli kap** | cevresel (hoop) gerilme | **0.871** | 7.12e-09 |
| Kolon burkulmasi | elastik kararsizlik | 0.88 | 5.00e-09 |
| Kiris tasarimi | egilme gerilmesi | 0.613 | ≈0 |

**Test edilen BES fiziksel/muhendislik gorevinin BESI de resmi kapiyi
(n=30, p<0.05, |delta|>0.33) geciyor.** Sifir istisna. Bu, projenin en
onemli metodolojik ve bilimsel donusu: onceki "dar, kirilgan, bazen
aciklanamayan FAIL'ler iceren" tablo, buyuk olcude iki bagimsiz
kalibrasyon hatasinin (terim dengesizligi, iki zit yonde) urunuymus.
Dogru kalibre edildiginde, "tek-atislik fiziksel/uzamsal yon
regresyonu" sinifi COK DAHA GENIS ve SAGLAM bir sekilde destekleniyor.

Bu, ozetin/makalenin/README'nin ana anlatisinin guncellenmesini
gerektiriyor -- "dar sinif, 2/3 ya da 2/4 fiziksel gorevde calisiyor"
yerine artik "tek-atislik fiziksel/uzamsal regresyon sinifinin
TAMAMINDA (test edilen 5/5) tutarli ve genellikle cok guclu bir
connectome avantaji" denilebilir.

# Faz 1 Ön Tescili — Kapı Deneyi

**Tarih:** 2026-09-14. PROTOCOL.md §4.5 kuralı: bu dosya deneyler
koşulmadan önce yazıldı ve koşulduktan sonra değiştirilmeyecek. Revizyon
gerekirse `faz_1_v2.md` açılır.

## Hipotez

H1 / H0, falsifikasyon kriteri: PROTOCOL.md §2, birebir. Özetle:
degree-preserving null ile gerçek connectome arasında final best-fx
üzerinden (30 eşleştirilmiş seed) Wilcoxon signed-rank p < 0.05 **ve**
Cliff's delta |δ| > 0.33 bulunmazsa H1 bu görev sınıfı için reddedilir.

## Kapsam

- **Mimari:** yalnızca Fly-Proposer (PROTOCOL.md Faz 1).
- **Fonksiyonlar:** 10-D Sphere (debug/sağlık kontrolü, gate kararına
  girmez) ve 10-D Rastrigin (asıl gate testi). Sınırlar: Sphere
  [-5, 5]^10, Rastrigin [-5.12, 5.12]^10 (standart).
- **Substrate'ler:** `flywire`, `er_null`, `degree_preserving_null`,
  `weight_shuffle_null` (factory.build_all, null_kinds=("er",
  "degree_preserving", "weight_shuffle")). `sign_flip_null` bu fazda
  koşulmuyor (PROTOCOL.md Faz 1 metninde açıkça listelenmemiş).
- **Seed:** 30, substrate'ler arasında paylaşılan (paired/eşleştirilmiş
  tasarım, PROTOCOL.md §4.3).
- **Evaluation bütçesi:** 10.000 (kullanıcı onayı, 2026-09-14 — bkz.
  "Bütçe kararı" aşağıda).

## Encoding / Decoding — kapsam kararı ve bloklayıcı

PROTOCOL.md Faz 1, encoding ve decoding için ikişer alternatif ister:
- Encoding: (a) rastgele n duyusal nörona akım enjeksiyonu, (b) gerçek
  duyusal sınıflara (görsel/olfaktör) topografik eşleme.
- Decoding: (a) motor nöron popülasyonundan lineer readout, (b) rastgele
  n nöron alt kümesinden readout.

Her iki (a) alternatifi de nöron sınıflandırma verisini
(`Supplemental_file1_neuron_annotations.tsv`, flyconnectome/flywire_annotations
GitHub) gerektiriyor — bu dosya **2026-09-14 itibarıyla henüz indirilmedi**
(bkz. `data/raw/README.md`).

**Karar:** Ana kapı testi (gate deneyi), sınıflandırma verisi gerektirmeyen
kombinasyonla koşulacak:
- Encoding: rastgele seçilmiş n=10 nörona akım enjeksiyonu, **tüm
  popülasyondan** seçilerek (protokolün "(a) rastgele duyusal nöron"
  alternatifinin, duyusal kısıtlama olmadan gevşetilmiş hali — bu bir
  sapma, gizlenmiyor).
- Decoding: rastgele seçilmiş K=50 nöron alt kümesinden ateşleme
  hızı readout'u (protokolün decode (b) alternatifi, tam karşılık).

Bu, protokolün istediği 4 kombinasyondan (encode a/b × decode a/b) sadece
birini kapsar. **Sınıflandırma verisi geldiğinde**, en azından
encode(gerçek duyusal sınıf) × decode(motor nöron) kombinasyonu ayrı bir
ön tescille (`faz_1b.md`) koşulup, "(b) ile (a) farksız çıkması anatominin
işe yaramadığının kanıtı" analizi tamamlanacak. **Bu eksiklik falsifikasyon
kapısının ana kararını geçersiz kılmaz** — kapı, connectome topolojisinin
GENEL olarak (bu belirli encode/decode altında) null'lardan ayrılıp
ayrılmadığını test ediyor; ama pozitif bir sonucun "connectome'un anatomik
yapısı mı yoksa sadece rekürren seyrek dinamiği mi işe yarıyor" sorusuna
tam cevabı, sınıflandırma verisi gelmeden eksik kalır.

## Fly-Proposer mimarisi

- (1+1) elitist hill-climber: `x_0` rastgele başlatılır (seed'e bağlı);
  her iterasyonda `FlyProposer.propose(x, ctx)` bir aday üretir, objective
  değerlendirilir, iyileşme varsa kabul edilir. Bu döngü substrate'ten
  bağımsız — tüm substrate'ler için birebir aynı kod (PROTOCOL.md §3).
- `FlyProposer`: `x` (10 boyut) → seçilen 10 nörona akım olarak enjekte
  edilir (`step_scale` ile ölçeklenir) → substrate `T=15` adım koşulur →
  K=50 readout nöronunun ortalama ateşleme hızı → sabit rastgele
  (substrate'e özel olmayan, sadece dim/K'ya bağlı) lineer projeksiyonla
  `Δx`'e dönüştürülür (`decode_scale` ile ölçeklenir) → `x_candidate =
  x + Δx`, sınırlara clip edilir.
- Encode/decode nöron indeksleri ve okuma projeksiyon matrisi **seed'e
  bağlı ama substrate'ten bağımsız** seçilir — aynı seed için tüm
  substrate'ler aynı indeksleri/matrisi kullanır (adil karşılaştırma,
  PROTOCOL.md §4.1 ruhu: substrate'in kendisi dışında hiçbir şey
  değişmemeli).

## Adil karşılaştırma / hiperparametre bütçesi (PROTOCOL.md §4.1)

Ayarlanacak hiperparametreler: `step_scale` ∈ {0.5, 1.0, 2.0},
`decode_scale` ∈ {0.2, 0.5, 1.0} (3×3 = 9 kombinasyon). Her substrate için
**eşit** bir ön-tarama bütçesi: 9 kombinasyon × 3 seed × 300 evaluation
(Rastrigin). En iyi ortalama best-fx veren kombinasyon o substrate için
ana koşumda kullanılır. Bu tarama, her substrate için ayrı ayrı ve eşit
harcanır — gerçek connectome'a daha fazla ayar denemesi verilmez.

## İstatistik

- Ana gate testi: `flywire` vs `degree_preserving_null`, 30 eşleştirilmiş
  seed, Wilcoxon signed-rank (`experiment/stats.py::wilcoxon_paired`),
  Cliff's delta, eşik `passes_falsification_gate` (p<0.05 ve |δ|>0.33).
- İkincil (bilgilendirici, kapı kararını etkilemez): `flywire` vs
  `er_null`, `flywire` vs `weight_shuffle_null`, Holm-Bonferroni ile
  çoklu karşılaştırma düzeltmesi.
- Metrik: 10.000 evaluation sonunda ulaşılan best-fx (minimizasyon,
  düşük daha iyi).

## Bütçe kararı (2026-09-14, iki aşamalı — ikinci düzeltme geçerli)

**İlk karar:** Kullanıcıya sunulan tahminle (10k eval / T=15 adım, ~2-4
saat paralel) kullanıcı 10k eval'ı onayladı.

**Düzeltme:** Gerçek koşumda bu tahmin ciddi şekilde yanlış çıktı. Sırasıyla:
(1) `torch.set_num_threads(1)` ile worker başına tek thread kullanmak, CSR
sparse matmul'ün thread'lerle iyi ölçeklendiğini gözden kaçırdı (izole
tek-thread: ~37ms/adım vs 4-thread: ~8ms/adım) — worker başına 1 thread
kullanmak paralellikten kazanılandan fazlasını kaybettirdi. (2) 12 worker
× 2 thread konfigürasyonu, her worker'ın kendi graf kopyasını (~1-2GB)
bellekte tutması nedeniyle OOM ile öldürüldü. (3) İzole tek görev
ölçümünde 300 eval/flywire ~37.7s (tam beklenen hız) çıkarken, **bu iş
yükünün çekirdek sayısıyla değil bellek bant genişliğiyle sınırlı
olduğu** ortaya çıktı: 2 paralel görev agregat ~27.3s/görev-eşdeğeri, 6
paralel görev ise **daha kötü** (~29.7s/görev-eşdeğeri) — bu makinede
paralellik 2'nin ötesinde fayda sağlamıyor. Bu ölçümlerle tam tasarımın
(10k eval) gerçekçi maliyeti ~30+ saat çıktı, onaylanan ~2-4 saatin çok
üzerinde (PROTOCOL.md §7.3 "maliyet öngörünün 2 katını aşarsa dur ve sor"
tetiklendi).

**İkinci karar (kullanıcı onayı, 2026-09-14):** Evaluation bütçesi
küçültüldü — **main_budget=1500, sphere_debug_budget=200,
tuning_budget=100**, T=15 adım, 30 seed ve 4 substrate **değişmedi**.
Tahmini süre ~5 saat (2 worker × 4 thread, ölçülen en iyi konfigürasyon).
Bu, istatistiksel tasarımı (seed sayısı, eşleştirme, testler, eşikler)
değiştirmiyor — sadece evaluation sayısını, dolayısıyla arama
sürecinin ne kadar "derin" arayabildiğini azaltıyor. Bu bir sınırlama
olarak kabul ediliyor, gizlenmiyor: gate geçilse de geçilmese de, sonuç
raporunda "1.5k evaluation ile" diye belirtilecek.

## Durma kararı

Gate testi tamamlandığında: geçerse Faz 2.1'e (Fly-Proposer genişletme)
geçiş kullanıcıya önerilir. Geçmezse, PROTOCOL.md §5 Faz 1 metnine göre
proje durmaz ama "seyrek rekürren reservoir" çerçevesine dönüşüm
kullanıcıya önerilir — karar kullanıcıya bırakılır, otomatik verilmez.

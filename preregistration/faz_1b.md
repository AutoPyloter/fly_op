# Faz 1b Ön Tescili — Anatomik Encode/Decode ile Kapı Deneyi Tekrarı

**Tarih:** 2026-09-15. PROTOCOL.md §4.5 kuralı: bu dosya deneyler
koşulmadan önce yazıldı ve koşulduktan sonra değiştirilmeyecek.

## Bağlam

Faz 1'in ana kapı testi (`preregistration/faz_1.md`) **geçmedi**
(EXPERIMENTS.md 2026-09-14 "FAZ 1 KAPI DENEYİ SONUCU": Wilcoxon p=0.6309,
Cliff's δ=0.001, flywire vs degree_preserving_null). Ama o test, encode/
decode için protokolün 4 kombinasyonundan sadece birini kullandı: **tüm
popülasyondan rastgele** enjeksiyon + **tüm popülasyondan rastgele**
okuma — sınıflandırma verisi o an mevcut değildi.

Kullanıcı `Supplemental_file1_neuron_annotations.tsv` dosyasını indirdi
(2026-09-14, `data/raw/`). Bu dosya `flow` kolonuyla her nöronun
**afferent** (duyusal girdi, n=19.261 eşleşti/19.262) veya **efferent**
(motor çıktı, n=1.489/1.489) olduğunu veriyor — protokolün encode(a)/
decode(a) alternatiflerini ("gerçek duyusal nörona enjeksiyon",
"motor nöron popülasyonundan readout") artık gerçek veriyle test
edebiliyoruz.

## Hipotez

Aynı H1/H0 ve falsifikasyon kriteri (PROTOCOL.md §2), bu sefer
**anatomik olarak gerçek** encode/decode ile. Ek olarak, protokolün
kendi sorduğu ikincil soru: Faz 1'in (rastgele encode/decode) ile Faz
1b'nin (anatomik encode/decode) sonuçları **farklı mı**? Farklı değilse,
bu "connectome'un anatomik yapısının işe yaramadığının doğrudan kanıtı"
(PROTOCOL.md Faz 1 metni) — yani hem H1 hem "anatomi önemli" hipotezi
reddedilir.

## Tasarım

Faz 1 ile **birebir aynı**: 10-D Rastrigin, T=15, 30 paylaşılan seed,
4 substrate (flywire, er_null, degree_preserving_null, weight_shuffle_null),
main_budget=1500, sphere_debug_budget=200, tuning_budget=100,
max_workers=2, THREADS_PER_WORKER=4 (Faz 1'in maliyet dersleri
uygulanıyor — bkz. EXPERIMENTS.md 2026-09-14 "Maliyet aşımı").

**Tek fark — encode/decode havuzları:**
- Encoding: `dim=10` nöron, **afferent havuzundan** (n=19.261) rastgele
  seçilir (protokol encode(a), artık gerçek duyusal popülasyona
  kısıtlanmış).
- Decoding: `n_readout=50` nöron, **efferent havuzundan** (n=1.489)
  rastgele seçilir (protokol decode(a), gerçek motor popülasyonu).
- Havuzlar `data/processed/afferent_indices.npy` ve
  `efferent_indices.npy`'den yüklenir (adjacency sırasına göre
  indekslenmiş, `proofread_root_ids_783.npy` ile eşleştirilmiş).
- Bu iki havuz biyolojik olarak ayrık (afferent ≠ efferent), kesişim
  kontrolü kodda var (`FlyProposer.__init__` assert).

Hiperparametre tarama ızgarası, kabul kriterleri, istatistik (Wilcoxon
signed-rank, Cliff's delta, gate eşiği p<0.05 ve |δ|>0.33) Faz 1 ile
aynı.

## Ek analiz (Faz 1 vs Faz 1b karşılaştırması)

Gate sonucundan bağımsız olarak, Faz 1'in flywire-rastgele sonuçları ile
Faz 1b'nin flywire-anatomik sonuçları da karşılaştırılacak (Mann-Whitney,
farklı seed'ler arası tasarım olmadığı için — aynı seed'ler ama farklı
encode/decode kullanıldığından paired olabilir, Wilcoxon tercih edilir).
Bu, "encode/decode tasarımı sonucu değiştiriyor mu" sorusuna cevap verir.

## Durma kararı

Faz 1'in durma kapısı mantığı (PROTOCOL.md Faz 1) burada da geçerli.
Ek olarak: Faz 1b de geçmezse, H1'in bu görev sınıfı için reddi daha
sağlam temellere oturur (iki farklı encode/decode tasarımıyla da null
sonuç) — pivot önerisi daha güçlü gerekçelendirilir. Faz 1b geçerse ama
Faz 1 geçmediyse, bu "anatomik yapı önemli, rastgele encode/decode
gürültüye boğuyor" bulgusu olur — kendi başına ilginç bir sonuç,
kullanıcıya ayrıca bildirilir.

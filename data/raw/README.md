# FlyWire v783 ham verisi

**Karar (2026-09-14):** Codex'in canlı indirme portalı (`codex.flywire.ai`)
sürekli güncellenen bir veritabanından besleniyor — protokolün istediği sabit
v783 snapshot'ı değil. Tekrarlanabilirlik için kullanıcıyla birlikte
**Zenodo'daki dondurulmuş v783 arşivi** kullanılmasına karar verildi. Bkz.
`EXPERIMENTS.md` 2026-09-14 girişi.

Bu klasör boş bırakılmıştır — veri lisans/atıf koşulu kabulü gerektirdiği
için **kullanıcı tarafından** indirilip buraya yerleştirilmelidir.

## Kaynak 1 — Bağlantı verisi (zorunlu)

**Zenodo, DOI 10.5281/zenodo.10676866** — "FlyWire Whole-brain Connectome
Connectivity Data", Dorkenwald et al. 2024, Version 783.0
https://zenodo.org/records/10676866

Gerekli dosyalar (bu klasöre indirilecek):
- `proofread_connections_783.feather` (852 MB) — nöron-nöron bağlantıları,
  neuropil başına satır: `pre_pt_root_id`, `post_pt_root_id`, `neuropil`,
  `syn_count`, ve nörotransmitter olasılık ortalamaları
  (`gaba_avg`, `ach_avg`, `glut_avg`, `oct_avg`, `ser_avg`, `da_avg`).
- `proofread_root_ids_783.npy` (1.1 MB) — doğrulanmış tüm nöron ID'lerinin
  listesi (n_units'i bu tanımlar, sadece bağlantısı olan nöronlar değil).

İhtiyaç yok (indirme, çok büyük ve Faz 0/1 için gerekli değil):
- `flywire_synapses_783.feather` (9.5 GB, sinaps-düzeyi ham veri — sadece
  topografik/uzamsal encoding denemeleri gerekirse Faz 1'de gerekebilir)
- `per_neuron_neuropil_count_{pre,post}_783.feather` (özet istatistik,
  doğrulama için opsiyonel)

## Kaynak 2 — Sınıflandırma / hücre tipi (Faz 1 encoding için gerekli,
## Faz 0 için opsiyonel)

**GitHub, flyconnectome/flywire_annotations** (Schlegel et al. 2024)
https://github.com/flyconnectome/flywire_annotations

Gerekli dosya:
- `supplemental_files/Supplemental_file1_neuron_annotations.tsv` — flow
  (afferent/efferent/intrinsic), superclass (sensory/motor/vb.), cell
  class, side, nörotransmitter, VirtualFlyBrain ID. Faz 1'in "gerçek
  duyusal nöron sınıflarına topografik eşleme" encoding alternatifi (b)
  için gerekli — bkz. PROTOCOL.md Faz 1.

## İndirme adımları

1. Zenodo sayfasındaki atıf/kullanım koşulunu kabul et (kullanıcı hesabı
   gerekmez, ama koşulları okuyup onaylamak kullanıcı eylemi).
2. `proofread_connections_783.feather` ve `proofread_root_ids_783.npy`
   dosyalarını indir, bu klasöre (`data/raw/`) hiçbir yeniden adlandırma
   yapmadan koy.
3. (Faz 1 için, şimdi değil) `Supplemental_file1_neuron_annotations.tsv`
   dosyasını GitHub'dan indirip aynı klasöre koy.

## Doğrulama

Dosyalar buraya konduktan sonra:

```
python -m flyopt.data.validate
```

nöron sayısı, kenar sayısı, yoğunluk, derece dağılımı ve E/I oranını
dosyalardan ölçüp `data/processed/manifest.json`'a yazar. İnternetteki
referans sayılar (~139k nöron, ~50M sinaps) protokol metninde geçer ama
kopyalanmaz; her ölçüm kendi indirdiğimiz dosyadan yapılır.

## Durum

- [x] `proofread_connections_783.feather` indirildi
- [x] `proofread_root_ids_783.npy` indirildi
- [x] `python -m flyopt.data.validate` çalıştırıldı, `manifest.json` üretildi
      (2026-09-14: 139.255 nöron, 15.091.983 kenar — bkz. EXPERIMENTS.md)
- [ ] (Faz 1 öncesi) `Supplemental_file1_neuron_annotations.tsv` indirildi

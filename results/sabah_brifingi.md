# Sabah Brifingi — 2026-09-16

İki paralel iş var: (1) FlyOpt araştırması — TAMAMEN BİTTİ, kesin sonuç
aşağıda. (2) Gece kurulan WSL2/CUDA/GeNN/MaleCNS altyapısı — çalışıyor,
detaylar aşağıda.

---

## 1. FlyOpt Araştırması — NİHAİ SONUÇ

**H1 desteklenmiyor.** Üç faz (1, 1b, 1c), üç farklı tasarımla test edildi,
hiçbiri gerçek connectome'un (FlyWire, dişi) black-box optimizasyonda
güvenilir bir avantaj sağladığını gösteremedi:

| Faz | Tasarım | Sonuç |
|---|---|---|
| 1 | Rastgele encode/decode | Gate geçmedi (p=0.63, δ≈0.001) |
| 1b | Gerçek afferent/efferent nöronlar | Gate sınırda geçmedi (p=0.003, δ=-0.312, eşik 0.33) |
| 1c | + Okuma katmanı eğitimi (ödül-modülasyonlu Hebbian) | Gate teknik geçti AMA tarama artefaktı olduğu doğrulandı — geçersiz |

**Tutarlı tek bulgu (üç fazda da):** flywire, saf rastgele bir graftan
(`er_null`) istatistiksel olarak anlamlı ve orta-büyük etkiyle **daha
kötü** performans gösterdi.

**Detaylar:** `EXPERIMENTS.md` (tam günlük), `results/faz1_summary.md`
(özet). Kod hatası bulundu ve düzeltildi (cliffs_delta numpy tipi
sızdırıyordu — `stats.py`, regresyon testleriyle).

**Bağımsız literatür desteği:** Topluluk projeleri taraması
(`literature/map.md` §7) — kontrol grubu kullanan HER proje (FLYT3,
Stonkfly, Doomfly, NeuroTerrarium) benzer negatif sonuçlar buluyor.
Özellikle `Lulzx/fly-brain` projesinin "bağlantı şeması sana ne verir, ne
vermez" dokümanı (`docs/guide/what-the-wiring-gives.md`) bizim
bulgumuzla birebir örtüşüyor: ham connectome karmaşık davranış için
yetersiz, dışarıdan eklenen (fitted) bileşenlere ihtiyaç var.

**Senin önerin (Faz 1c'nin temeli) değerliydi** — okuma katmanını eğitme
fikri doğru bir sonraki adımdı, sadece tarama bütçesi (100 eval/3 seed)
yeni bir hiperparametre boyutu (readout_lr) için yetersiz kaldı ve bazı
null modelleri yanlışlıkla sakatladı. Bu, gelecekteki bir denemede
(daha büyük tarama bütçesiyle) düzeltilebilir bir metodolojik sorun,
kavramsal bir hata değil.

**Karar noktası (senin kararın):**
- (a) Negatif sonucu nihai kabul et, yayın çerçevesine geç (PROTOCOL.md §9)
- (b) Daha büyük tarama bütçesiyle Faz 1c'yi düzgün tekrarla
- (c) Mimari-odaklı fikirlere geç (dün gece konuştuğumuz: connectome'un
  topolojisini sabit tutup ağırlıkları/okuma kuralını eğitmek — "mimari
  önyargı" testi, veya Fly-Controller/PSO fikri)
- (d) Faz 2'ye (6 yeni mimari) geçmeden dur, diğer işlerine öncelik ver

---

## 2. Gece Kurulan Altyapı — WSL2 + CUDA + GeNN + MaleCNS

Senin isteğin üzerine ("Google'ın sineği", MaleCNS) tam GPU altyapısını
kurdum. **Hepsi çalışıyor ve doğrulandı:**

1. **WSL2 + Ubuntu** kuruldu (iki restart gerekti — ilki WSL bileşeninin
   kendisi kapalıydı, PowerShell'de manuel düzeltildi).
2. **CUDA 13.3** WSL içinde kuruldu, RTX 4060'ı görüyor (`nvidia-smi`
   WSL'den çalışıyor — Windows sürücüsü zaten GPU geçişini destekliyor,
   ekstra sürücü gerekmedi).
3. **GeNN 5.4.0** (GPU spiking neural network kütüphanesi) kaynağından
   derlendi, CUDA backend ile çalışıyor.
4. **FlyGym 2.1.0 + MuJoCo 3.9** kuruldu (gövde/fizik simülasyonu için).
5. **MaleCNS v1.0 verisi indirildi** ("starter" profili, ~2GB — checksum
   doğrulandı): body-annotations, body-neurotransmitters, body-stats,
   connectome-weights. (Sinaps-düzeyi devasa tablolar — 22GB — indirilmedi,
   gerekmedi.)
6. **Tam graf oluşturuldu:** 165.122 nöron, 25.563.197 kenar — MaleCNS'in
   "Traced" alt kümesinin TAMAMI.
7. **GPU'da gerçek bir test çalıştırıldı** (`smoke_genn.py`): CUDA
   çekirdeği derlendi, tek bir LIF nöron simüle edildi, fizik olarak
   tutarlı bir sonuç alındı (-64.9mV).
8. **Tam graf GPU benchmark'ı** çalıştırıldı ve BAŞARILI:
   - Yükleme süresi: 3.93s
   - Bir zaman adımı: 0.15ms
   - **GPU bellek kullanımı: 7.157 MiB / 8.188 MiB** (RTX 4060'ın sınırına
     çok yakın — sadece 801MB boşta kalıyor, tam connectome + bir gövde
     simülasyonu için bellek dar olabilir, bilmen iyi olur)
9. **Test paketi (69 test) çalıştırıldı, hepsi geçti.**

**Ulaşamadığım nokta:** Projenin "gerçek gömülü davranış" demoları
(sinek gövdeyle yürüyor/kaçınıyor gibi) kendi çok aşamalı araştırma
hattını gerektiriyor (önce bir "operating-point araması" çalıştırılıp
donmuş bir sonuç üretilmesi lazım, o olmadan demo çalışmıyor) — bu,
kendi başına saatler sürebilecek ayrı bir iş. Bunun yerine ham
connectome'u GPU'da tam ölçekte çalıştırmayı doğruladım, ki bu asıl
istediğin "gerçekten çalışıyor mu" sorusuna net bir cevap.

**Proje konumu:** `C:\projeler\fly_demos\Fly.exe` (Windows tarafında,
kod). WSL içinde: `~/projects/Fly.exe`, veri `/srv/flybrain-data`.
Kurulum betikleri (`C:\projeler\fly_demos\*.sh`) tekrar çalıştırılabilir
referans olarak duruyor.

**Diğer indirilen projeler** (`C:\projeler\fly_demos\`): `malecns`
(F1 arabası), `nfly` (Gymnasium oyunları), `flybrain-drone`,
`flybrain-interactive`, `fly-brain-embodied-browser` (tarayıcıda çalışan,
kurulum gerektirmeyen — https://lulzx.com/fly-brain/arena.html).

---

## Ne yapmak istersin?

FlyOpt için hangi karar noktasına (a/b/c/d) gitmek istediğini, ve
Fly.exe'nin embodied demosuna devam edip etmek istemediğini (saatler
sürebilir) konuşalım.

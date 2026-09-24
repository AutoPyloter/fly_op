# Faz 1c Ön Tescili — Ödül-Modülasyonlu Okuma Eğitimi

**Tarih:** 2026-09-15. PROTOCOL.md §4.5 kuralı: bu dosya deneyler
koşulmadan önce yazıldı ve koşulduktan sonra değiştirilmeyecek.

## Bağlam ve motivasyon

Faz 1 ve Faz 1b'nin ikisi de kapı testini geçemedi, ve her ikisinde de
flywire, saf rastgele bir graftan (er_null) istatistiksel olarak anlamlı
şekilde daha kötü çıktı (EXPERIMENTS.md 2026-09-14/15). Bu sonuç iki
farklı şekilde yorumlanabilir:
1. Connectome'un spike dinamiği, bu görev için gerçekten faydasız/
   dezavantajlı bilgi taşıyor, VEYA
2. Connectome'un spike dinamiği faydalı bilgi taşıyor ama **tamamen
   rastgele, hiç eğitilmemiş okuma matrisimiz** bu bilgiyi bulamıyor —
   reservoir computing'in standart pratiğinde okuma katmanı her zaman
   eğitilir, biz şimdiye kadar kasıtlı olarak eğitmedik (H0'ın en saf
   halini test etmek için).

Bu ayrımı yapmak, kullanıcının kendi önerisiyle ortaya çıktı (2026-09-15
sohbeti): "rastgele ağırlık diyoruz ama bir sonraki adıma götürecek
şekilde fine-tune edebiliriz." Bu, Fly-RL'den (PROTOCOL.md §2.5, en
pahalı/kırılgan varyant) çok daha ucuz bir ön-test: connectome'un içi
yine hiç değişmiyor, sadece okuma matrisi.

## Mekanizma

Reward-modulated üç-faktörlü Hebbian kural (PROTOCOL.md §2.5'in
bahsettiği mekanizma ailesi, ama sadece okuma matrisine uygulanıyor,
sinaptik ağırlıklara değil):

```
her propose() çağrısında: delta = readout_W @ firing_rate  (üretilen Δx)
her tell(improved) çağrısında:
    reward = +1 eğer improved, -1 değilse
    readout_W = (1 - lr) * readout_W + lr * reward * outer(delta, firing_rate)
```

`lr=0` Faz 1b'yi birebir tekrarlar (kontrol/karşılaştırma noktası zaten
elimizde). `src/flyopt/variants/fly_proposer.py::FlyProposer.tell()`.

## Tasarım

Faz 1b ile **birebir aynı**: encode=afferent havuzu (n=19.261), decode=
efferent havuzu (n=1.489), 10-D Rastrigin, T=15, 30 paylaşılan seed,
4 substrate, main_budget=1500, sphere_debug_budget=200,
max_workers=2, THREADS_PER_WORKER=4.

**Tek ek boyut:** hiperparametre tarama ızgarasına `readout_lr ∈
{0.0, 0.05, 0.2}` eklendi (step_scale × decode_scale × readout_lr = 27
kombinasyon, önceki 9'un yerine). `readout_lr=0.0`'ın taramada en iyi
çıkması mümkün — bu durumda o substrate için pratikte Faz 1b'yi tekrar
etmiş oluruz, bu da kendi başına bilgilendirici (bu substrate için
plastisite yardımcı olmuyor demektir).

## Hipotez

**Beklenti (koşum başlamadan önce yazıldı):** Emin değilim — bu,
Faz 1/1b'den farklı olarak net bir ön-beklentim olmayan bir deney.
Argüman her iki yönde de var: (a) reservoir computing literatürü
(literature/map.md §6) eğitilmiş okumanın işe yaradığını gösteriyor,
bu da lehte bir işaret; (b) ama connectome'un er_null'dan *kötü* çıkması
(bilgi eksikliği değil, potansiyel bir dezavantaj) okumayı eğitmenin
bunu telafi edemeyeceğini düşündürüyor. Sonucu görmeden tahmin
etmiyorum, bu yüzden nötr bir bekleyişle giriyorum.

## Durma/yorum kriteri

- Gate geçerse (flywire vs degree_preserving_null, p<0.05 ve |δ|>0.33):
  connectome'un dinamiği faydalı bilgi taşıyormuş ama rastgele okuma
  bunu kaçırıyormuş — Fly-RL'ye (tam sinaptik plastisite) geçmek için
  daha güçlü bir gerekçe oluşur.
- Gate yine geçmezse: hem Fly-Proposer hem "ucuz okuma eğitimi" fikri
  bu görev sınıfı için tükenmiş sayılır — Fly-RL'nin daha da pahalı
  versiyonuna yatırım yapmak için gerekçe zayıflar, negatif sonuç
  çerçevesi güçlenir.
- Her iki durumda da, flywire vs er_null karşılaştırmasının yönü
  (tutarlı şekilde flywire kötü çıkıyordu) özellikle izlenecek —
  eğitim bu yönü tersine çevirir mi, yoksa devam mı eder.

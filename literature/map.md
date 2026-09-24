# Literatür Haritası

PROTOCOL.md §8 kapsamında, Faz 0 ile paralel yürütülür. Her aile için:
girdi, ağ neyi öğreniyor, ödül, çıktı, popülasyon kullanımı, eğitim maliyeti,
hangi benchmark'ta klasik optimizerları gerçekten geçmiş, genelleme var mı,
ve "sineği buraya koyabilir miyiz / neden koyamayız".

Durum: iskelet — her aile için literatür taraması henüz yapılmadı.
Doldurulacak alanlar `TODO` ile işaretli.

---

## 1. Learned Optimizers / L2O (Learning to Optimize) — 2026-09-15 dolduruldu

- Girdi: Optimize edilen ("target") modelin gradyanı/parametre durumu
  (Adam/SGD gibi klasik optimizer'ların girdisiyle aynı yapıda).
- Ağ ne öğreniyor: Bir "meta-optimizer" (genelde RNN/LSTM) — gradyan
  geçmişinden bir sonraki parametre güncelleme adımını üretmeyi öğrenir.
  Optimizasyon, bir öğrenme problemi olarak yeniden çerçevelenir.
- Ödül: Meta-eğitim sırasında target modelin kayıp fonksiyonu (veya birkaç
  adım sonraki kaybı) — gradyan tabanlı meta-öğrenme ile.
- Çıktı: Parametre güncelleme adımı (Δθ), Fly-Proposer'ın Δx üretmesiyle
  yapısal olarak aynı rol.
- Popülasyon: Hayır — tek trafik/tek model üzerinde çalışır (bizim (1+1)
  hill-climber'ımıza yapısal olarak yakın).
- Eğitim maliyeti: Yüksek — meta-optimizer'ın kendisi, çok sayıda target
  problem üzerinde gradyan tabanlı meta-eğitim gerektirir (biz connectome'u
  hiç eğitmiyoruz, sabit substrate kullanıyoruz — bu, maliyet açısından
  L2O'dan temel bir ayrışma noktası).
- Klasik optimizerları geçtiği benchmark: Küçük/orta ölçekli, eğitim
  dağılımına yakın problemlerde (örn. küçük MLP eğitimi) Adam'ı geçebiliyor;
  büyük/farklı mimarilerde güvenilir üstünlük yok.
- Genelleme: **Bilinen en büyük zayıflık** — "meta-overfitting": öğrenilen
  optimizer eğitim dağılımına (örn. küçük bir MLP + MNIST) aşırı uyum
  sağlıyor, farklı mimari/ölçekte (örn. ResNet + ImageNet) performansı
  çöküyor (Open-L2O, JMLR 2022 "A Primer and A Benchmark"; PyLO 2025).
- Sinek değerlendirmesi: **FlyOpt'tan kategorik olarak farklı bir mekanizma.**
  L2O, gradyan tabanlı meta-eğitimle GÖREV-ÖZGÜ bir optimizer öğrenir;
  connectome-driven optimizasyon hiçbir eğitim yapmaz, sabit bir biyolojik
  substrate kullanır. Ortak zayıflık: ikisi de "bu substrate/optimizer
  genelleşiyor mu yoksa dar bir göreve mi kilitli" sorusuyla karşı karşıya
  — FlyOpt'un H0'ı tam da bunun bir versiyonu (connectome'un "genellemesi",
  aslında hiçbir şeye özel olmayan bir null-model davranışı mı).

Kaynaklar: Learning to Optimize: A Primer and A Benchmark (JMLR 23, 2022,
jmlr.org/papers/volume23/21-0308/21-0308.pdf); PyLO: Towards Accessible
Learned Optimizers in PyTorch (arXiv:2506.10315, 2025); M-L2O (arXiv:2303.00039).

## 2. Neural Combinatorial Optimization (Pointer Networks, attention-based) — 2026-09-15 dolduruldu

- Girdi: Problem örneği (örn. TSP için şehir koordinatları).
- Ağ ne öğreniyor: Bir çözümü adım adım (otoregresif) inşa etmeyi —
  Pointer Network (Vinyals ve devamı), her adımda "sıradaki hangi
  şehir/düğüm" sorusuna dikkat (attention) mekanizmasıyla cevap veriyor.
- Ödül: Genelde pekiştirmeli öğrenme (tur uzunluğu gibi bir maliyetin
  negatifi) veya supervised (optimal çözümlerden taklit).
- Çıktı: Tam bir çözüm dizisi (örn. ziyaret sırası), FlyOpt'un ürettiği
  "küçük bir Δx adımı"ndan farklı olarak **tüm çözümü bir seferde inşa
  ediyor** (constructive), FlyOpt gibi mevcut çözümü iyileştiren
  (improvement-based) değil.
- Popülasyon: Hayır (tek ağ, inşa sırasında dallanma/örnekleme olabilir).
- Eğitim maliyeti: Yüksek — çok sayıda problem örneğiyle (RL veya
  supervised) meta-eğitim gerekiyor, connectome'un hiç eğitilmemesinin
  tam tersi.
- Klasik optimizerları geçtiği benchmark: TSP/CVRP'de 1000 düğüme kadar
  ölçeklerde klasik sezgisellere yakın/rekabetçi sonuçlar (2024-2025
  çalışmaları), gerçek TSPLib/CVRPLib örneklerine de makul genelleme.
- Genelleme: Alanın **ana açık sorunu** — küçük eğitim örneklerinden
  büyük/gerçek-dünya örneklerine genelleme zayıf, 2024-2025'in çoğu
  makalesi tam olarak bunu düzeltmeye çalışıyor (test-time projection
  learning, heavy decoder, instance-conditioned adaptation).
- Sinek değerlendirmesi: **Farklı problem sınıfı** — bu aile kombinatoryal
  yapı-inşa problemleri için tasarlanmış (sıralama, rota), FlyOpt'un
  şu ana kadar test ettiği sürekli fonksiyon optimizasyonundan (Rastrigin)
  ayrı. PROTOCOL.md Faz 3'ün "Knapsack, TSP" planı burada bu aileyle
  doğrudan karşılaştırılabilir hale gelir — ama connectome'un kombinatoryal
  bir çözümü NASIL "inşa edeceği" (constructive mi, improvement-based mi)
  henüz tasarlanmadı; Faz 3'e girmeden önce netleştirilmesi gereken bir
  tasarım sorusu.

Kaynaklar: Vinyals, Fortunato, Jaitly, "Pointer Networks" (NeurIPS 2015,
temel makale); "Neural Combinatorial Optimization Algorithms for Solving
Vehicle Routing Problems: A Comprehensive Survey" (arXiv:2406.00415, 2024);
"Neural Combinatorial Optimization with Heavy Decoder" (arXiv:2310.07985);
"Improving Generalization of Neural Combinatorial Optimization... via
Test-Time Projection Learning" (NeurIPS 2025, arXiv:2506.02392).

## 3. Neural Hyper-heuristics / Adaptive Operator Selection — 2026-09-15 dolduruldu

Fly-Controller (§2.2) ile doğrudan ilişkili; rakip baseline burada.

- Girdi: Popülasyon durumu (çeşitlilik, iyileşme oranı, stagnasyon
  sayacı) — Fly-Controller'ın durum vektörüyle birebir aynı tasarım.
- Ağ ne öğreniyor: Hangi mutasyon/crossover operatörünün veya hangi F/CR
  parametre değerinin seçileceğini — pekiştirmeli öğrenme (RL, genelde
  policy gradient) veya "success-history adaptation" (SHA) gibi
  istatistiksel adaptasyon mekanizmalarıyla.
- Ödül: Bir sonraki adımda(lar)da elde edilen fitness iyileşmesi.
- Çıktı: Operatör seçimi + sürekli parametreler (F, CR gibi).
- Popülasyon: Evet — DE/GA popülasyon istatistikleri üzerinden çalışır
  (Fly-Controller'ın hedeflediği kullanım tam bu).
- Eğitim maliyeti: RL-tabanlı yaklaşımlar (policy gradient ile DE
  parametre adaptasyonu, IEEE TEVC) meta-eğitim gerektiriyor; SHA/JADE/
  SHADE gibi klasik adaptif yöntemler ise **eğitimsiz, çalışırken
  (online) istatistiksel adaptasyon** yapıyor — bu, maliyet açısından
  connectome'a en yakın rakip: hem SHADE hem Fly-Controller "önceden
  eğitim yok" kategorisinde.
- Klasik optimizerları geçtiği benchmark: **SHADE/L-SHADE, CEC2014'ten
  beri CEC yarışmalarının kazananı** — yani Fly-Controller'ın yenmesi
  gereken asıl rakip zaten kendi sınıfının en güçlü, en iyi ayarlanmış
  üyesi (PROTOCOL.md §4.6 "baseline'lar zayıf tutulmaz" kuralına tam
  uyan bir karşılaştırma noktası).
- Genelleme: RL-tabanlı adaptif operatör seçimi genelde eğitildiği
  problem ailesine iyi genelliyor ama farklı problem sınıflarında
  (örn. sürekli → kombinatoryal) yeniden eğitim gerekiyor. SHA/SHADE
  ailesi ise problem-agnostik (hiç eğitim yapmadığı için).
- Sinek değerlendirmesi: Fly-Controller'ın kritik soru işareti burada —
  PROTOCOL.md'nin kendi metninde de "connectome, aynı parametre sayılı
  MLP'yi geçiyor mu?" diye sorulmuştu (bir RL-tabanlı MLP kontrolcü
  rakip olarak). Ama asıl zor rakip SHADE/L-SHADE: **eğitimsiz olması**
  bakımından connectome ile aynı kategoride ve zaten CEC yarışmalarını
  kazanmış durumda — Fly-Controller'ın bunu geçmesi, Fly-Proposer'ın
  Faz 1'de geçemediği (p=0.63, δ≈0) barın çok daha yükseğinde bir hedef.

Kaynaklar: "Reinforcement Learning-based Self-adaptive Differential
Evolution through Automated Landscape Feature Learning" (arXiv:2503.18061);
"Dynamic operator management in meta-heuristics using reinforcement
learning" (arXiv:2408.14864); "Learning from Offline and Online
Experiences: A Hybrid Adaptive Operator Selection Framework"
(arXiv:2404.10252); SHADE/L-SHADE CEC2014 kazananı (Tanabe & Fukunaga).

## 4. Neuroevolution (NEAT, HyperNEAT, ES-based) — 2026-09-15 dolduruldu

- Girdi: Görev-özel gözlem (RL/kontrol bağlamı — örn. bir kontrol
  probleminin durum vektörü).
  Ağ ne öğreniyor: Ağın kendi **ağırlıkları VE topolojisi** birlikte
  evrimleştiriliyor (NEAT: doğrudan kodlama; HyperNEAT: CPPN ile dolaylı
  kodlama, büyük ağları kompakt bir üretici fonksiyondan türetiyor).
- Ödül: Görev-özel fitness (örn. kontrol performansı, oyun skoru).
- Çıktı: Ağın kendi davranışı (motor komutu, karar vb.) — optimize edilen
  şey AĞIN KENDİSİ, FlyOpt'ta olduğu gibi ağın ÇIKTISI başka bir
  optimizasyon probleminin girdisi değil.
- Popülasyon: Evet, temelde popülasyon-tabanlı (genetik algoritma
  iskeleti üzerine kurulu).
- Eğitim maliyeti: Orta-yüksek — topoloji arama, ağırlık öğrenmesinden
  ayrı bir arama boyutu ekliyor; GPU-hızlandırmalı yeni araçlar
  (TensorNEAT, 2025) bunu pratikleştiriyor.
- Klasik optimizerları geçtiği benchmark: Karışık — OpenAI-ES, sürekli
  kontrol görevlerinde karşılaştırılan yöntemlerin çoğunu geçiyor/eşitliyor
  (PMC7805676, 2024 karşılaştırması). HyperNEAT basit görevlerde NEAT'i
  geçiyor ama görev karmaşıklaştıkça avantajı kayboluyor — **FlyOpt'un
  "connectome avantajı boyutla/karmaşıklıkla kayboluyor mu" sorusuna
  (Faz 2.1, boyut ölçeklemesi 10-30-50D) yakın bir örüntü.**
- Genelleme: Genelde tek görev için evrimleştirilir, farklı göreve
  transfer NEAT/HyperNEAT'in odak noktası değil (transfer için ayrı bir
  literatür var, burada kapsam dışı).
- Sinek değerlendirmesi: **Mekanik olarak farklı** — neuroevolution ağın
  kendi ağırlık/topolojisini ARAR (ağ = optimize edilen nesne);
  FlyOpt'ta connectome SABİT, başka bir problemi optimize etmek için
  kullanılan bir ARAÇ. Ortak nokta: ikisi de "sinir ağı benzeri bir
  yapı, gradyansız arama" fikrini paylaşıyor, ama neuroevolution'ın asıl
  gücü (topoloji + ağırlık birlikte arama) FlyOpt'ta hiç kullanılmıyor
  — connectome'un topolojisi zaten sabit, hiç aranmıyor.

Kaynaklar: TensorNEAT (arXiv:2504.08339, ACM TELO 2025); "Efficacy of
Modern Neuro-Evolutionary Strategies for Continuous Control Optimization"
(PMC7805676); "Comparison of NEAT and HyperNEAT on a Strategic
Decision-Making Problem" (MIT, web.mit.edu/jessiehl/Public/aaai11).

## 5. Surrogate-Assisted Evolutionary Optimization — 2026-09-15 dolduruldu

Fly-Surrogate (§2.6) için teorik gerekçe protokolde zaten zayıf
işaretlenmiş; burada rakip baseline'ların gücünü belgeliyoruz.

- Girdi: Geçmiş (x, f(x)) çiftleri (değerlendirilmiş adaylar).
- Ağ ne öğreniyor: Pahalı objective fonksiyonunu TAKLİT etmeyi —
  Gaussian Process (Bayesian optimization) veya gradient boosting
  (XGBoost/LightGBM tabanlı) modeller, birkaç düzine örnekle bile
  düzgün bir yaklaşım üretebiliyor.
- Ödül: Tahmin hatası (surrogate eğitimi) + belirsizlik-farkında bir
  edinim fonksiyonu (acquisition function, örn. expected improvement)
  arama için.
- Çıktı: f(x) tahmini + belirsizlik — GP'nin en büyük avantajı bu
  (kalibre belirsizlik tahmini, connectome'un spike çıktısından böyle
  bir şey türetmek doğal değil).
- Popülasyon: Genelde hayır (sıralı, örnek-verimli arama) — az sayıda
  pahalı değerlendirmeyle çalışmak üzere TASARLANMIŞ, tam da Faz 3'ün
  ("objective evaluation 30 saniye sürüyorsa") hedeflediği rejim.
  §6'daki (Gradient-free optimization of chaotic acoustics with reservoir
  computing) makale de bu ailenin bir ESN-tabanlı örneği.
- Eğitim maliyeti: Çok düşük (GP: kapalı-form, birkaç düzine nokta;
  gradient boosting: hafif) — connectome'un LIF simülasyon maliyetiyle
  (adayı başına ~0.1-1s, bu oturumda ölçüldü) karşılaştırıldığında GP/GB
  ÇOK daha ucuz.
- Klasik optimizerları geçtiği benchmark: Bayesian optimization (GP-
  tabanlı), pahalı black-box fonksiyonlarda (hiperparametre arama,
  mühendislik simülasyonları) endüstri standardı — rastgele arama ve
  ızgara aramayı büyük farkla geçiyor, bu iyi belgelenmiş ve tartışmasız.
- Genelleme: Düşük boyutlu (genelde <20D) problemlerde güçlü, yüksek
  boyutta (>50-100D) GP'nin ölçeklenmesi zorlaşıyor — FlyOpt'un Faz 2.1
  boyut ölçeklemesi (10-30-50D) tam bu sınırın içinde kalıyor.
- Sinek değerlendirmesi (protokolün kendi değerlendirmesiyle uyumlu):
  Connectome'un GP/gradient boosting'i geçmesi için **bilinen bir
  mekanizma yok** — GP'nin kalibre belirsizlik tahmini ve düşük örnek
  maliyeti, connectome'un hiçbir avantajının telafi edemeyeceği kadar
  güçlü rakip özellikler. Bu varyant protokolde zaten "en son, en az
  bütçeyle" koşulacak şekilde sıralanmış — Faz 1'in Fly-Proposer'da bile
  gate geçilememiş olması (p=0.63), Fly-Surrogate'in bu çok daha güçlü
  rakibe karşı şansının Fly-Proposer'dan da düşük olduğunu düşündürüyor.

Kaynaklar: Bayesian Optimization standart referansları (Shahriari et al.,
"Taking the Human Out of the Loop: A Review of Bayesian Optimization",
IEEE Proceedings 2016); "Gradient-free optimization of chaotic acoustics
with reservoir computing" (arXiv:2106.09780, §6'da da geçiyor).

## 6. Reservoir Computing / Echo State Networks — 2026-09-15 dolduruldu

**Öncelikli.** Null hipotezimizin (H0) teorik temeli burada.

- Girdi: Zaman serisi / sürekli sinyal (klasik ESN kullanımında), FlyOpt'ta
  ise x ∈ ℝⁿ'in kodlanmış hali (bizim encode şemamız).
  Ağ ne öğreniyor: **Hiçbir şey** — bu, reservoir computing'in temel
  önermesi (Jaeger, "echo state property"): rezervuarın kendisi (rastgele
  ağırlıklı, sabit, seyrek rekürren ağ) hiç eğitilmez; sadece çıktı
  katmanı (lineer readout) eğitilir. FlyOpt'un Fly-Proposer'ı da AYNI
  ilkeyi kullanıyor — connectome'u da hiç eğitmiyoruz, sadece encode/decode
  (readout) sabit/rastgele. **Bu, H0'ın teorik temelinin tam burada
  olduğunu doğruluyor**: eğer "sabit rastgele rekürren ağ + eğitilmiş/sabit
  readout" formülü zaten iyi çalışıyorsa, connectome'un ekstra bir şey
  katması için topolojisinin spesifik olarak önemli olması gerekir —
  genel "seyrek rekürren dinamik" yapısı değil.
- Ödül: Yok (denetimsiz/deterministik dinamik; readout ayrı eğitilir,
  bizim tasarımımızda readout da rastgele/sabit, hiç eğitilmiyor).
- Çıktı: Rezervuarın durumundan (spike/aktivasyon) lineer bir projeksiyon.
- Popülasyon: Hayır.
- Eğitim maliyeti: Çok düşük (sadece lineer readout, kapalı-form çözüm
  mümkün) — FlyOpt'ta bu maliyet bile yok, readout da rastgele sabit.
- Klasik optimizerları geçtiği benchmark: ESN'ler zaman serisi tahmininde
  (kaotik sistemler, örn. Mackey-Glass) güçlü sonuçlar veriyor, ama bu
  **optimizasyon** değil **tahmin/dinamik modelleme** — doğrudan
  karşılaştırılabilir bir "ESN'yi arama operatörü olarak kullanan, klasik
  optimizerlarla karşılaştırılan" çalışma bulunamadı (2026-09-15 taraması).
  Bulunan tek yakın örnek — "Gradient-free optimization of chaotic
  acoustics with reservoir computing" (arXiv:2106.09780) — ESN'yi bir
  **surrogate model** (pahalı simülasyonun yerine geçen, VERİYLE EĞİTİLMİŞ
  bir tahminci) olarak kullanıyor, Bayesian optimizasyonla birleştirilmiş,
  brute-force grid search'ten ~10x hızlı yakınsıyor. Bu, FlyOpt'un
  **Fly-Surrogate** varyantına (§2.6, teorik gerekçesi zaten zayıf
  işaretlenmiş) daha yakın — Fly-Proposer'ın "eğitilmemiş, sabit substrate
  doğrudan arama üretir" mekanizmasından farklı.
- Genelleme: ESN'ler genelde tek bir dinamik sistem/görev için ayarlanır
  (reservoir boyutu, spektral yarıçap, sızıntı oranı gibi hiperparametreler
  göreve özel taranır — bkz. "Evolutionary Echo State Network" ailesi,
  ESN hiperparametrelerini GA/PSO/CMA-ES ile optimize eden çalışmalar).
  Önemli: bu literatür rezervuarı OPTİMİZE EDİYOR (ESN'nin ağırlıkları/
  topolojisi evrimleştiriliyor) — FlyOpt'un yaptığının **tersi** yönde bir
  ilişki (biz rezervuarı sabit tutup onunla başka bir şeyi optimize
  ediyoruz). Bu ayrım literatür konumlandırmasında açıkça belirtilmeli.
- Sinek değerlendirmesi / H0 için sonuç: Reservoir computing literatürü
  "sabit rastgele seyrek rekürren ağ + readout" formülünün dinamik
  modelleme için güçlü olduğunu net şekilde destekliyor. Bunun
  **optimizasyon proposal operatörü** olarak da işe yarayıp yaramadığı
  ayrı, daha az çalışılmış bir soru — ama teorik olarak, "rastgele bir
  rezervuar zaten iyi çalışıyorsa, connectome'un SPESİFİK topolojisi ne
  katar" sorusu H0'ı güçlü şekilde destekliyor. **Faz 1'in gate
  sonucuyla (p=0.63, δ≈0.001, connectome ile degree-preserving null
  ayırt edilemez) bu literatür teorik beklentisi tam örtüşüyor.**

Kaynaklar: Jaeger & Haas, "Harnessing Nonlinearity" (Science 2004, ESN'nin
temel makalesi, echo state property); Echo State Network - Scholarpedia
(scholarpedia.org/article/Echo_State_Network); "Gradient-free optimization
of chaotic acoustics with reservoir computing" (arXiv:2106.09780); "Comparison
of Reservoir Computing topologies using the Recurrent Kernel approach"
(ScienceDirect, 2024); "Evolutionary Echo State Network: evolving reservoirs
in the Fourier space" (arXiv:2206.04951).

## 7. Connectome-Constrained Neural Networks (2024+, fly connectome mimarisi) — 2026-09-15 dolduruldu

**Shiu et al. 2024 (Nature, "A leaky integrate-and-fire computational
model based on the connectome of the entire adult Drosophila brain"):**
127.400+ nöron, 50M+ sinaps, LIF + α-sinaps dinamiği, nörotransmitter
kimliği önceki büyük ölçekli tahminlerden alınmış (bizim
`NT_TO_SIGN`/`nt_avg` yaklaşımımızla aynı aile). **Optimizasyon/arama
içermiyor** — besleme devresi ve anten-temizleme (grooming) devresi gibi
bilinen, spesifik devreleri simüle edip davranışsal/elektrofizyolojik
veriyle doğruluyorlar. FlyOpt'un "connectome'u bir arama operatörü olarak
kullanma" sorusuyla **doğrudan çakışmıyor** — bu, projenin özgünlüğünü
tehdit etmiyor, aksine "connectome'un ciddi bilimsel kullanımı nasıl
görünür" için bir referans noktası.

**Önemli bulgu — `annel0/flybrain` (GitHub, topluluk projesi, hakemli
değil, 2026-09-15 taraması):** GPU'da connectome-kısıtlı LIF simülasyonu,
Shiu et al.'ı ±1-3% hassasiyetle tekrar üretiyor. **Kendi degree-preserving
rewiring null model'iyle** (bizim kullandığımız aynı null model ailesi)
şunu buluyor: gerçek connectome'da aktivite belirli hücrelerde
yoğunlaşırken (örn. APL nöronu belirli bir gain-control rolü oynuyor),
degree-preserving rewire edilmiş versiyonda **aktivite 8 kat daha fazla
hücreye yayılıyor ve APL nöronu tamamen susuyor.** Yani bu spesifik
devre-seviyesi hesaplama (koku işleme devresinde gain kontrolü) için
**connectome'un gerçek topolojisi, degree-preserving null'dan
ölçülebilir şekilde farklı davranıyor.**

**Bunun FlyOpt için anlamı — çelişki değil, tamamlayıcı:** Bu bulgu bizim
Faz 1 sonucumuzla (connectome, Rastrigin optimizasyonunda degree-preserving
null'dan ayırt edilemez, p=0.63) ÇELİŞMİYOR, aksine onu daha inandırıcı
kılıyor: (1) Aynı null model ailesi (degree-preserving rewiring) burada
**gerçek bir farkı yakalayabiliyor** — yani metodoloji "kör" değil, connectome
yapısı gerçekten önemli olduğunda bunu tespit edebiliyor. (2) Fark, connectome'un
**evrimleştiği görevle ilgili** bir devrede (koku işleme, gain kontrolü)
ortaya çıkıyor — protokolün §10'da öngördüğü tam senaryo: "sinek beyni
koku/görme/lokomotor devreleri için evrimleşti... Rastrigin'e dair
bilinen bir mekanizma yok." Connectome'un yapısı ÖNEMLİ, ama SADECE
kendi evrimleştiği görev sınıfında — soyut bir optimizasyon görevinde değil.
Bu, Faz 1b'nin (anatomik/duyusal encode-decode) neden önemli bir sonraki
adım olduğunu da güçlendiriyor: belki connectome'un yapısı, kendi doğal
girdi/çıktı yollarına (duyusal→motor) daha yakın bir görev kurgusunda
iş görür, tamamen rastgele enjeksiyon/okumada değil.

Diğer topluluk projeleri (`vshapenko/flypoke`, `eonsystemspbc/fly-brain`,
`philshiu/Drosophila_brain_model`) benzer whole-brain LIF simülasyon/
emülasyon amaçlı, optimizasyon/arama iddiası taşımıyorlar — özgünlük
tehdidi değil.

Kaynaklar: Shiu et al., Nature 2024 (biorxiv.org/content/10.1101/
2023.05.02.539144, ön-baskı sürümü); `github.com/annel0/flybrain`
(topluluk projesi, hakemli değil — bulgu burada bilgilendirici olarak
kullanılıyor, birincil kaynak olarak değil); "Neuromorphic Simulation of
Drosophila Melanogaster Brain Connectome on Loihi 2" (arXiv:2508.16792).

**Ek — 2026-09-16 taraması: MaleCNS v1.0 ve ilgili topluluk projeleri**

FlyWire'dan (bizim kullandığımız, dişi, Princeton) ayrı bir connectome:
**MaleCNS v1.0** (HHMI Janelia FlyEM + Cambridge + MRC LMB + Google
Research, 8 Haziran 2026, CC-BY) — erkek *Drosophila*, 165.122 nöron,
25.563.197 kenar (Traced alt kümesi). 11 Eylül 2026'da bir geliştiricinin
(Nick Walton) bu connectome'u kullanarak "Rubik küpü çözdürdüğü" viral bir
X (Twitter) videosu yayıldı — Doom/Minecraft/Mario 64/Mini Cooper sürme
gibi bir "viral gösteri dalgası"nın parçası. **Bu bir bilimsel iddia
değil**: hiçbir kontrol grubu/karşılaştırma yok, "çözdü" demek "geçerli
hamleler üretti" demek olabilir, connectome'un katkısını ayırt eden hiçbir
şey gösterilmiyor.

**`Ibtisam-Mohammad/Fly.exe`** — bulunan en titiz topluluk projesi.
MaleCNS'in TAMAMINI (165.122 nöron, 25.563.197 kenar) GeNN/CUDA ile
GPU'da çalıştırıyor, MuJoCo/NeuroMechFly gövdesine bağlıyor. Metodolojik
disiplini FlyOpt'a çok yakın: her parametrenin kaynağı sınıflandırılmış
(M=ölçülmüş, P=popülasyon önseli, F=uydurulmuş/fitted, E=mühendislik
yaması, I=kurtarılamaz), ön-tescilli kabul kriterleri var, başarısız
sonuçlar saklanıyor (`docs/STATUS.md`: "Track A's 0-of-30
grooming-displacement failure"). On iki sinekli sürü demosu bile açıkça
şunu yazıyor: **"The walking is engineered. No part of the simulated
ventral nerve cord contributes to leg movement"** ve kontrol grubu
eksikliğini itiraf ediyor: "The controls that would make this a swarm
result do not exist yet." 2026-09-16'da bu proje FlyOpt'un kendi
makinesinde (RTX 4060) tekrar üretildi: WSL2+CUDA 13.3+GeNN 5.4.0
kuruldu, tam graf inşa edildi ve GPU'da çalıştırıldı (7,16GB/8GB VRAM,
0.15ms/adım) — bağımsız bir doğrulama, ayrıntı `results/sabah_brifingi.md`
2026-09-16.

**`Lulzx/fly-brain`** — tarayıcıda çalışan, en dürüst proje
(`docs/guide/what-the-wiring-gives.md`): "bağlantı şeması sana ne verir,
ne vermez" ayrımını açıkça belgeliyor. **Bağlantı şemasından çıkanlar:**
yürüme komutları, kaçış/kalkış refleksleri, çiftleşme algısı, seyrek
mantar-cisimciği kodlaması. **Çıkmayanlar, dışarıdan eklenenler:**
koordineli yürüme (elle uydurulmuş bir adım üreteci gerekiyor),
kendiliğinden davranış istatistikleri (etoloji makalelerinden), bir
"reafference gain" (yoksa sinek kendi ayak seslerinden sonsuza kadar
yürüyor). **Oktopamin sürprizi:** literatüre göre kurulan açlık zinciri,
saf connectome üzerinde denendiğinde aç sineği beklenenin TERSİNE **daha
az** yürütüyor.

**FlyOpt için anlamı:** Bu üç kaynak (viral video, Fly.exe, Lulzx),
alanın tam spektrumunu gösteriyor — kontrolsüz spektakülden en titiz
uca. En titiz olanlar (Fly.exe, Lulzx) bizim bulgumuzla aynı yöne işaret
ediyor: **ham connectome karmaşık/hedefli davranış için yetersiz,
dışarıdan eklenen (fitted/engineered) bileşenlere ihtiyaç var**, ve
biyolojik olarak makul bir mekanizma (oktopamin) bile beklenenin tersi
sonuç verebiliyor — tıpkı bizim Faz 1c'de bir null modelin plastisiteyle
kötüleşmesi gibi.

Kaynaklar: male-cns.janelia.org/release/; Janelia haber duyurusu
(janelia.org/news/researchers-reveal-connectome-of-the-male-fruit-fly-
central-nervous-system); `github.com/Ibtisam-Mohammad/Fly.exe`;
`github.com/Lulzx/fly-brain`; `github.com/cobanov/awesome-fly` (52
projelik derlenmiş liste — kontrol grubu kullanan projelerin negatif
sonuç raporladığı gözlem için bkz. "Özgünlük tehdidi taraması" bölümü,
2026-09-16 eki).

## 8. Metafor-Temelli Metasezgisel Eleştirisi (Sörensen 2015 ve devamı) — 2026-09-15 dolduruldu

**Bu bölüm "ne yapmamamız gerektiğini" belirler.**

- Ana eleştiri (Sörensen, "Metaheuristics—the metaphor exposed",
  International Transactions in Operational Research, 2015): optimizasyon
  alanı, çoğu doğal/yapay bir süreçten esinlenen "yeni" metasezgisel
  yöntemlerin bir "tsunamisi"yle boğulmuş durumda — bunların çoğu, yazarın
  yayın yapma isteği dışında net bir gerekçe olmadan öneriliyor. Devamı
  ("Metaphor-based metaheuristics, a call for action: the elephant in the
  room", Swarm Intelligence, 2022) bu eleştirinin literatürde yankı
  bulmasına rağmen aynı tür makalelerin yayınlanmaya devam ettiğini
  belgeliyor.
- Ampirik kanıt: "Large-scale Benchmarking of Metaphor-based Optimization
  Heuristics" (arXiv:2402.09800, 2024) — düzinelerce farklı isim/metaforla
  pazarlanan algoritmanın, aslında birbirinin yeniden-parametrelenmiş
  (reparametrized) hali olduğunu, farklı isimler altında **aynı
  matematiksel operatörleri** kullandığını sistematik olarak gösteriyor.
- **Fruit Fly Optimization Algorithm (FOA)** tam olarak bu aile içinde:
  sinek yiyecek arama davranışından "esinlenen" bir metasezgisel, gerçek
  ölçülmüş bir sinek beyni/connectome verisiyle hiçbir ilişkisi yok — saf
  metafor. FOA'nın kendi literatüründe de bilinen zayıflıkları var
  (erken yakınsama, yerel optimuma takılma) ve bunları gidermek için
  sürekli "geliştirilmiş FOA" varyantları üretiliyor (chaotic FOA, quasi-
  affine transformation FOA, vb.) — Sörensen'in eleştirdiği tam örüntü.
- **PROTOCOL.md §1.3'teki adlandırma disiplininin gerekçesi burada:**
  FlyOpt, "sinekten esinlenen" bir metasezgisel DEĞİL — gerçek ölçülmüş
  connectome topolojisini sabit bir substrat olarak kullanıyor, hiçbir
  metafor/esinlenme iddiası yok. Bu ayrımı net tutmak için proje boyunca
  "fly-inspired algorithm" dili kullanılmıyor; "connectome-driven
  optimization" / "biological substrate as proposal operator" tercih
  ediliyor. Faz 1'in negatif sonucu (gate geçmedi) bu ayrımı daha da
  önemli kılıyor: eğer connectome, null modellerinden ayırt edilemiyorsa,
  "sinek" ismini kullanmak tam da Sörensen'in eleştirdiği duruma
  (isim/metaforun kendisinin bir katkı iddiası taşıması) düşme riski taşır
  — bulgularımızı raporlarken bu riske özellikle dikkat edilmeli.

Kaynaklar: Sörensen, "Metaheuristics—the metaphor exposed" (ITOR, 2015,
onlinelibrary.wiley.com/doi/abs/10.1111/itor.12001); "Metaphor-based
metaheuristics, a call for action: the elephant in the room" (Swarm
Intelligence, 2022, link.springer.com/article/10.1007/s11721-021-00202-9);
"Large-scale Benchmarking of Metaphor-based Optimization Heuristics"
(arXiv:2402.09800, 2024); "A systematic review on fruit fly optimization
algorithm and its applications" (Artificial Intelligence Review, 2023).

---

## Özgünlük tehdidi taraması

Bu bölüm sürekli güncellenir: connectome-driven optimizasyonu doğrudan
deneyen bir yayın bulunursa buraya not düşülür ve PROTOCOL.md §7.3 gereği
kullanıcıya haber verilir (durma/dur-sor durumu).

**2026-09-15 taraması (WebSearch, bölüm 1-8'i doldururken paralel
yürütüldü):** Connectome'u soyut bir black-box optimizasyon probleminde
(sürekli fonksiyon minimizasyonu gibi) arama/proposal operatörü olarak
kullanıp klasik optimizerlarla veya null modellerle karşılaştıran bir
yayın/proje **bulunamadı**. En yakın bulgular:
- Shiu et al. 2024 ve türevleri (§7): connectome-kısıtlı LIF simülasyonu,
  ama optimizasyon değil, devre/davranış doğrulama amaçlı.
- `annel0/flybrain` (§7): degree-preserving null ile karşılaştırma yapıyor
  ama devre-seviyesi hesaplama (gain kontrolü) için, soyut optimizasyon
  için değil — FlyOpt'un sorusuyla örtüşmüyor, tamamlayıcı.
- Reservoir computing (§6) ve L2O (§1): kavramsal olarak yakın ("sabit/
  öğrenilmiş bir dinamik sistemi arama/tahmin için kullanma") ama connectome
  spesifik değiller.

**Sonuç: Özgünlük tehdidi tespit edilmedi.** FlyOpt'un sorusu ("gerçek
ölçülmüş bir connectome, soyut optimizasyonda null modellerden ayırt
edilebilir mi") literatürde doğrudan test edilmiş görünmüyor — bu hem
projenin özgünlüğünü koruyor hem de Faz 1'in negatif sonucunun
"zaten bilinen bir şeyi tekrarlamak" olmadığını, gerçek bir bilgi
katkısı olduğunu gösteriyor (PROTOCOL.md §9 negatif senaryo, hâlâ
yayınlanabilir).

**Ek — 2026-09-16 taraması (`cobanov/awesome-fly`, 52 topluluk projesi):**
Kontrol grubu/baseline kullanan HER proje bizim bulgumuzla aynı yönde
negatif sonuç raporluyor: **FLYT3** (tic-tac-toe + 5-sınıflı yangın
sınıflandırması) connectome modeliyle %60.8 doğruluk, basit bir **MLP
%85.7** — connectome kaybediyor. **Stonkfly/OpenFly** (borsa/trading):
"kârlı öğrenme gösterilemedi" / "kanıtlanmış kâr avantajı yok".
**Doomfly/FlyPong**: "negatif dopamin plastisite sonuçları". **NeuroTerrarium**:
"negatif topoloji-kontrol sonucu". Viral/kontrolsüz gösteri projeleri
(araç sürme, oyun oynama) ise hiçbir baseline içermiyor — "çalışıyor"
göstermek "null modelden iyi" göstermekle aynı şey değil. Bu, özgünlük
tehdidi değil ama **bağımsız doğrulama**: FlyOpt'un titiz null-model
metodolojisi olmadan yapılan gösterimler bilgilendirici değil, metodolojiyle
yapılanlar bizim sonucumuzu destekliyor. Detay: bölüm 7 (Ek — 2026-09-16).

Not: Bu tarama WebSearch ile yapıldı, sistematik bir literatür taraması
(örn. Google Scholar/Semantic Scholar ile kapsamlı sorgu) değil —
Faz 2'ye geçilirse (veya yayın hazırlığı başlarsa) daha sistematik bir
tarama tekrarlanmalı.

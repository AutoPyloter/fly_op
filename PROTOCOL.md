# FlyOpt — Connectome-Driven Optimization Araştırma Protokolü
## Claude Code için agentik görev tanımı

> Ön tescil disiplini bu dosyaya da uygulanır: bu, kullanıcının 2026-09-14
> tarihinde verdiği görev tanımının birebir kaydıdır. İçerik sonradan
> değiştirilmez; revizyon gerekirse yeni bir bölüm eklenir ve nedeni belirtilir.

---

## 0. BU DOSYA NE DEĞİL

Bu bir "şunu kodla" talebi değil. Bu, **falsifiye edilebilir bir araştırma
hipotezinin sistematik olarak test edilmesi** için bir protokoldür. Senden
beklenen şey, hipotezi doğrulamak değil; onu **öldürmeye çalışmak** ve
öldüremezsen neden öldüremediğini kanıta bağlamaktır.

Eğer bu protokolü uygularken hipotezin yanlış olduğuna dair kanıt bulursan,
bu bir başarısızlık değil, **projenin en değerli çıktısıdır**. Negatif sonucu
gizleme, yumuşatma veya "belki şu parametreyle olur" diye kovalama. Raporla ve
durma kapısına git.

---

## 1. ÇIKIŞ NOKTASI VE FİKRİN DOĞUŞU

### 1.1 Nereden geldi

Kullanıcı (geoteknik + optimizasyon araştırmacısı) FlyWire
projesinin yayınladığı tam yetişkin *Drosophila melanogaster* connectome'unu
(~139.000 nöron, ~15M yönlü ağırlıklı bağlantı, ~50M sinaps) kullanan açık
kaynak projeleri inceledi. Bunların bir kısmı ciddi bilimsel temele dayanıyor
(Shiu et al., Nature 2024 — tüm beynin LIF nöronlarıyla simülasyonu), bir kısmı
ise connectome'u OCR, trading, hatta kripto token mekanizmalarına bağlayan
deneysel/spekülatif işler.

Oradan şu soru doğdu:

> Sinek beynini bir **sınıflandırıcı** olarak kullanmak yerine, bir
> **optimizatör** olarak kullanabilir miyiz?

Yani connectome'u sabit bir hesaplama grafiği (fixed neural substrate) kabul
edip, onun seyrek rekürren dinamiğini bir arama operatörü olarak çalıştırmak.

### 1.2 Fikir patlamasının ürettiği varyantlar

Beyin fırtınası şu yedi mimariyi ortaya çıkardı:

| # | Ad | Sinek ne yapıyor |
|---|---|---|
| 1 | Fly-Proposer | Mevcut çözümden yeni aday çözüm (Δx) üretiyor |
| 2 | Fly-Controller | GA/PSO/DE operatörlerini seçen hyper-heuristic |
| 3 | Fly-Agent | PSO'daki her particle yerine bir beyin |
| 4 | Fly-Swarm | Beyinler birbirine kuplajlı, kolektif arama |
| 5 | Fly-RL | Plastisite açık; ödülle arama politikası öğreniyor |
| 6 | Fly-Surrogate | Pahalı objective function'ı taklit ediyor |
| 7 | Fly-Hybrid | Sinek lokal karar + PSO/DE/CMA-ES global koordinasyon |

Ayrıca kullanıcının kendi HSDS (Harmony Search with Dependent Variable Spaces /
Domain-Aware Search Spaces) çalışmasıyla birleştirme fikri var: kısıt-temelli
tasarım uzayı indirgeme + connectome-driven arama. Bkz. Bölüm 8.

### 1.3 Adlandırma disiplini

Literatürde zaten **Fruit Fly Optimization Algorithm (FOA)** var ve bu,
metafor-temelli metasezgisel eleştirisinin (Sörensen 2015 ve devamı) tam
ortasında duran bir aile. Bizim yaptığımız şey ondan **kategorik olarak
farklı**: biz sinekten esinlenmiyoruz, gerçek ölçülmüş connectome topolojisini
substrat olarak kullanıyoruz.

Bu yüzden proje boyunca **asla** "fly-inspired algorithm" veya "yeni
metasezgisel" dili kullanma. Kullanılacak terminoloji:

- Connectome-driven optimization
- Connectome-constrained search dynamics
- Biological substrate as proposal operator

Metafor dili kullanırsan hakem sürecinde proje baştan kaybeder.

---

## 2. MERKEZİ HİPOTEZ VE NULL HİPOTEZ

### H1 (ana hipotez)
Gerçek FlyWire connectome topolojisinin seyrek rekürren dinamiği, black-box
optimizasyonda, **aynı istatistiksel özelliklere sahip rastgele graflardan
ölçülebilir şekilde daha iyi** bir proposal dağılımı / arama politikası üretir.

### H0 (null hipotez — yenmen gereken şey)
Gözlenen her performans, connectome'un *spesifik* topolojisinden değil,
sadece şunlardan kaynaklanır:
- ağın seyrekliği (sparsity),
- ağın boyutu,
- rekürren dinamiğin kendisi,
- derece dağılımı,
- eksitatör/inhibitör oranı.

**Bu ayrımı yapmadan üretilen hiçbir sonuç yayınlanabilir değildir.**
Reservoir computing / echo state network literatürü, rastgele seyrek rekürren
ağların da benzer işleri gördüğünü zaten gösteriyor. Bizim tek bilimsel
katkımız, connectome'un bunun *ötesinde* bir şey kattığını göstermek ya da
gösterememektir.

### Falsifikasyon kriteri (önceden tescilli)
Faz 1 sonunda, degree-preserving rewired null model ile gerçek connectome
arasında 30 seed üzerinden Mann-Whitney U testinde p < 0.05 **ve** Cliff's
delta |δ| > 0.33 (orta etki) bulunamazsa, H1 bu görev sınıfı için
reddedilmiştir.

---

## 3. TEMEL TASARIM KARARI: TEK ARAYÜZ

Yedi varyantı ayrı ayrı kodlarsan proje ölür. Hepsi tek bir çekirdeğin
sarmalayıcısı olmalı:

```python
class Substrate(Protocol):
    def reset(self, seed: int) -> State: ...
    def step(self, stimulus: np.ndarray, state: State,
             reward: float | None = None) -> tuple[np.ndarray, State]: ...
    @property
    def n_units(self) -> int: ...
```

Ve optimizasyon tarafı:

```python
class Proposer(Protocol):
    def propose(self, x: np.ndarray, ctx: SearchContext) -> np.ndarray: ...
    def tell(self, x: np.ndarray, fx: float, improved: bool) -> None: ...
```

`Substrate` implementasyonları:
- `FlyWireSubstrate` (gerçek connectome)
- `ERNullSubstrate` (eşleşmiş yoğunluklu Erdős–Rényi)
- `DegreePreservingNullSubstrate` (Maslov–Sneppen rewire)
- `WeightShuffleNullSubstrate` (topoloji aynı, ağırlıklar karıştırılmış)
- `SignFlipNullSubstrate` (E/I etiketleri karıştırılmış)
- `RandomMLPSubstrate` (aynı parametre sayılı ileri beslemeli kontrol)

**Kural:** Her deney, gerçek connectome ile birlikte en az
`DegreePreservingNull` ve `WeightShuffleNull` üzerinde de otomatik koşar.
Null'ları "sonra koşarız" diye erteleme imkânı kodda bulunmasın — koşum
fonksiyonu bir substrate listesi alsın, tek substrate kabul etmesin.

---

## 4. ADİL KARŞILAŞTIRMA KURALLARI (İHLAL EDİLEMEZ)

Bunlar protokolün en kritik kısmı. Bu kurallar ihlal edilirse üretilen tüm
sayılar çöptür.

1. **Eşit ayar bütçesi.** Gerçek connectome için harcanan her hiperparametre
   arama denemesi, her null model için de aynen harcanır. Sineği ayarlayıp
   null'ı varsayılan parametrelerle koşmak, sonucun tamamını açıklayabilecek
   bir yapaylıktır.

2. **Eşit değerlendirme bütçesi.** Karşılaştırmalar objective function
   evaluation sayısına göre yapılır (10k / 50k / 100k). Ayrıca wall-clock
   süresi **ayrıca** raporlanır — çünkü 139k LIF nöronunu her adayda simüle
   etmek PSO'nun velocity güncellemesinden ~10⁴ kat pahalı ve bu gerçeği
   gizlemek dürüstsüzlük olur.

3. **Seed disiplini.** Her konfigürasyon minimum 30 bağımsız seed.
   Seed'ler substrate ve optimizer arasında paylaşılır (paired comparison).

4. **İstatistik.** Mann-Whitney U veya Wilcoxon signed-rank (eşleştirilmiş
   seed'ler için), Holm-Bonferroni çoklu karşılaştırma düzeltmesi, etki
   büyüklüğü olarak Cliff's delta veya Vargha-Delaney Â₁₂. Sadece ortalama
   karşılaştırması **yasak**.

5. **Ön tescil.** Her faz başlamadan önce `preregistration/faz_N.md` dosyası
   yazılır: hangi hipotez, hangi metrik, hangi eşik, hangi durma kararı.
   Deney koşulduktan sonra bu dosya değiştirilemez. Değiştirmek gerekirse yeni
   dosya açılır ve nedeni yazılır.

6. **Baseline'lar zayıf tutulmaz.** CMA-ES, DE (SHADE/L-SHADE), PSO ve pure
   random search, kendi literatürdeki önerilen ayarlarıyla koşulur. Kasten
   kötü ayarlanmış baseline kullanmak en yaygın sahtekârlık biçimidir.

---

## 5. FAZLAR

### FAZ 0 — Altyapı (hedef: 1-2 hafta)

**Amaç:** Veriyi yükle, simülatörü kur, null modelleri üret, hiçbir iddiada
bulunma.

Görevler:
- FlyWire v783 verisini indir (`neurons.csv.gz`, `classification.csv.gz`,
  `connections_princeton.csv.gz`, `consolidated_cell_types.csv.gz`).
  Lisans kabulü ve hesap gerekiyor — bunu kullanıcı yapacak, sen indirme
  adımını dokümante et ve veriyi `data/raw/` altında bekle.
- Adjacency'yi `scipy.sparse.csr_matrix` olarak kur. Nöron sayısı, bağlantı
  sayısı, yoğunluk, derece dağılımı, E/I oranını **kendi verinden ölç** ve
  raporla. İnternetteki sayıları kopyalama; doğrula.
- LIF simülatörü: önce `brian2` ile referans implementasyon, sonra
  PyTorch/JAX ile vektörize sparse matmul versiyonu. İkisinin aynı spike
  train'i ürettiğini bir birim testiyle kanıtla (tolerans dahilinde).
- Null model üreticileri + her birinin gerçek ağın hangi istatistiğini
  koruduğunu gösteren doğrulama testleri.
- Deney koşum çerçevesi: config → run → artifact. Her koşumun git commit
  hash'i, config hash'i, seed'i ve ortam bilgisi kaydedilsin.

**Çıkış kriteri:** Gerçek connectome ve 5 null model, aynı arayüzden, aynı
uyarana karşı çalışıyor ve tekrarlanabilir spike aktivitesi üretiyor.

**Bu fazda hiçbir optimizasyon deneyi yapma.**

---

### FAZ 1 — KAPI DENEYİ (hedef: 1 hafta)

**Amaç:** Projeyi devam ettirmeye değer mi, sadece bunu öğren.

Kapsam kasten dar:
- Tek mimari: **Fly-Proposer**
- İki fonksiyon: 10-D Sphere (debug) ve 10-D Rastrigin (asıl test)
- Substrate'ler: FlyWire + DegreePreserving + WeightShuffle + ER
- 30 seed, 10k evaluation bütçesi

Tasarım kararları (bunları açıkça belgele, keyfî seçim yapma):
- **Encoding:** x ∈ ℝⁿ nasıl uyarana dönüşüyor? En az iki alternatif dene:
  (a) rastgele seçilmiş n adet duyusal nörona akım enjeksiyonu,
  (b) gerçek duyusal nöron sınıflarına (görsel/olfaktör) topografik eşleme.
- **Decoding:** spike aktivitesi → Δx. En az iki alternatif:
  (a) motor nöron popülasyonunun ateşleme hızından lineer readout,
  (b) rastgele n nöron alt kümesinden readout.
- (b) seçeneklerinin (a)'dan farksız çıkması, connectome'un anatomik
  yapısının işe yaramadığının doğrudan kanıtıdır — bu bilgiyi topla.

**DURMA KAPISI:**
- Gerçek connectome, degree-preserving null'ı istatistiksel anlamlılık **ve**
  orta etki büyüklüğüyle geçemezse → H1 bu görev sınıfı için reddedilir.
- Bu durumda proje **durmaz, ama dönüşür**: hat "biyolojik connectome
  avantajı" değil, "seyrek rekürren reservoir'lar proposal operatörü olarak"
  olur. Bu da meşru bir çalışmadır ama farklı bir makale ve farklı bir
  literatür konumlandırmasıdır. Kullanıcıya bu dönüşümü öner, kendi başına
  karar verme.

---

### FAZ 2 — MİMARİ TARAMASI (kapı geçilirse)

Sıra kasten ucuzdan pahalıya. Her varyant kendi mini durma kapısına sahip.

**2.1 Fly-Proposer (genişletilmiş)**
Faz 1'in devamı. CEC2017 / BBOB alt kümesine genişlet. Boyut ölçeklemesi:
10-D, 30-D, 50-D. Boyut arttıkça avantaj korunuyor mu, yoksa kayboluyor mu?

**2.2 Fly-Controller**
DE veya PSO'nun operatör/parametre seçimini connectome'a devret.
- Durum vektörü: popülasyon çeşitliliği, iyileşme oranı, stagnasyon sayacı
- Çıktı: mutation stratejisi seçimi, F ve CR değerleri
- Rakip: aynı işi yapan küçük bir MLP, ve adaptive DE (JADE/SHADE) varsayılanı
- **Kritik soru:** connectome, aynı parametre sayılı MLP'yi geçiyor mu?
  Geçmiyorsa bu varyant ölür.

**2.3 Fly-Agent**
PSO'daki her particle bir beyin. Burada hesaplama maliyeti patlar
(N particle × 139k nöron). Önce popülasyon boyutunu küçük tut (N=10) ve
maliyeti ölç, sonra ölçeklenebilirlik kararı ver.
- Ablasyon: tüm sinekler aynı connectome ama farklı internal state mi, yoksa
  farklı alt-graf mı kullanmalı?

**2.4 Fly-Swarm**
Beyinler arası kuplaj (bir sineğin çıktısı diğerinin uyaranına giriyor).
Bu varyantın Fly-Agent'tan farkı sadece kuplaj terimi olmalı — ayrı kod
tabanı değil.

**2.5 Fly-RL (en pahalı)**
Plastisiteyi aç: reward-modulated STDP veya three-factor Hebbian kural.
- Seviye A: tek problemde öğrenme
- Seviye B: problem ailesi üzerinde meta-öğrenme, sonra görülmemiş probleme
  transfer
- Seviye B'deki transfer, projenin en güçlü potansiyel sonucu. Ama en pahalı
  ve en kırılgan olanı. Buraya ancak 2.1–2.3 pozitif çıkarsa gel.

**2.6 Fly-Surrogate**
Açıkça söylüyorum: bu varyantın başarılı olması için **teorik bir gerekçe
yok**. Fonksiyon yaklaşımında Gaussian process ve gradient boosting güçlü,
iyi anlaşılmış ve ucuz. Connectome'un onları geçmesi için bir mekanizma
göremiyorum. Yine de tam kapsama için koş, ama en son ve en az bütçeyle.
Negatif sonucu rahatça raporla.

**2.7 Fly-Hybrid**
Ayakta kalan en iyi varyantı klasik optimizer'la birleştir.
Ablasyon zorunlu: sinek-yok, klasik-yok, ikisi-var.

---

### FAZ 3 — GERÇEK PROBLEMLER

Buraya sadece Faz 2'den sağ çıkan 1-2 varyant gelir.

Problem sınıfları (ucuzdan pahalıya):
- Knapsack, TSP (kombinatoryal davranış kontrolü)
- Truss / kafes sistem boyutlandırma
- Betonarme istinat duvarı optimizasyonu (kullanıcının mevcut veri seti var)
- İksa sistemi + Hardening Soil model tabanlı black-box

**Burada avantaj tersine döner:** objective evaluation 30 saniye sürüyorsa,
beynin 200 ms'i önemsizleşir. Faz 2'de sineği yenen wall-clock argümanı
burada geçersizleşir. Bu, projenin gerçek uygulama alanıdır ve makalede
böyle konumlandırılmalıdır.

---

## 6. HSDS İLE BİRLEŞTİRME

Kullanıcının HSDS çalışması, geometrik/mühendislik kısıtları nedeniyle
imkânsız olan bölgeleri tasarım uzayından **önceden** çıkarıyor. Bu, sinek
tarafıyla dikey olarak birleşiyor:

```
Orijinal uzay
     ↓
HSDS: kısıt-temelli uzay indirgeme
     ↓
İndirgenmiş uzay
     ↓
Connectome-driven proposal
     ↓
Objective değerlendirme → ödül → (opsiyonel) plastisite
     ↺
```

**Zorunlu 2×2 ablasyon tasarımı:**

| | HSDS yok | HSDS var |
|---|---|---|
| **Klasik optimizer** | baseline | HSDS'nin tek katkısı |
| **Fly proposer** | sineğin tek katkısı | birleşik |

Etkileşim terimi olmadan "birleşim işe yarıyor" denemez. İki bileşenin
katkıları toplanabilir mi (additive) yoksa etkileşiyor mu (synergistic) —
asıl ilginç soru bu.

**Uyarı:** HSDS kullanıcının aktif yayın hattı. Bu birleşimi HSDS'nin ana
makalesine sokma. Ayrı, sonraki bir çalışma olarak konumlandır. HSDS'nin
kendi katkısını deneysel bir eklentiyle bulandırmak yayın açısından risk.

---

## 7. SENİN ÇALIŞMA KURALLARIN (AGENT DAVRANIŞI)

1. **Her koşumdan önce** ne beklediğini `EXPERIMENTS.md`'ye yaz. Sonucu
   gördükten sonra beklentini değiştirme.

2. **Sonuç sürpriz çıkarsa önce kodu şüphelen.** Beklenmedik derecede iyi bir
   sonuç, neredeyse her zaman bir sızıntı veya bug'dır. Şunları kontrol et:
   objective function'a fazladan çağrı, null model üretiminde hata, seed
   sızıntısı, bütçe muhasebesi hatası.

3. **Kullanıcıya şu durumlarda dur ve sor:**
   - Bir durma kapısına ulaşıldığında (kararı sen verme)
   - Bir tasarım kararı sonucun yorumunu değiştirecekse
   - Hesaplama maliyeti öngörünün 2 katını aşarsa
   - Literatürde çalışmanın özgünlüğünü tehdit eden bir yayın bulursan

4. **Kullanıcıya şu durumlarda sorma, devam et:**
   - Rutin implementasyon kararları
   - Test yazma, refactor
   - Ön tescil edilmiş deneyleri koşma

5. **Kod tabanını küçük tut.** Her varyant ≤ 150 satır yeni kod olmalı.
   Bundan fazlası gerekiyorsa arayüz soyutlaması yanlış demektir — durup
   arayüzü düzelt.

6. **Negatif sonuçları eşit ayrıntıyla raporla.** `results/negative/` klasörü
   `results/positive/` kadar dolu olmalı. Doluysa dürüst çalışıyorsundur.

---

## 8. LİTERATÜR HARİTASI (Faz 0 ile paralel yürüsün)

Şu ailelerin her biri için: girdi nedir, ağ neyi öğreniyor, ödül nedir, çıktı
nedir, popülasyon kullanıyor mu, eğitim maliyeti nedir, hangi benchmark'ta
klasik optimizerları gerçekten geçmiş mi, genelleme yapıyor mu?

- Learned optimizers / L2O (learning to optimize)
- Neural Combinatorial Optimization (Pointer networks, attention-based)
- Neural hyper-heuristics / adaptive operator selection
- Neuroevolution (NEAT, HyperNEAT, ES-based)
- Surrogate-assisted evolutionary optimization
- Reservoir computing ve echo state networks — **özellikle bu**, çünkü null
  hipotezimizin teorik temeli burada
- Connectome-constrained neural networks (2024 sonrası fly connectome
  mimarisini NN olarak kullanan çalışmalar)
- Metafor-temelli metasezgisel eleştirisi (Sörensen ve devamı) — bizim ne
  yapmamamız gerektiğini anlamak için

Çıktı: `literature/map.md`, her aile için yukarıdaki soruların cevabı ve
"sineği buraya koyabilir miyiz / neden koyamayız" değerlendirmesi.

---

## 9. YAYIN HEDEFLERİ (gerçekçi)

- **Pozitif senaryo:** Connectome, null modelleri anlamlı şekilde geçiyor.
  Makale: "Do biological connectome dynamics provide a useful inductive bias
  for black-box optimization?" — hedef: Swarm and Evolutionary Computation,
  Applied Soft Computing, veya IEEE TEVC.
- **Negatif senaryo:** Geçmiyor. Makale: "Sparse recurrent substrates as
  proposal operators: connectome topology provides no advantage over
  degree-matched nulls" — bu da yayınlanabilir ve alan için değerli. Negatif
  sonuç makalesi yazmaktan çekinme.
- **Yazılım:** `flyopt` paketi — SoftwareX veya JOSS.

---

## 10. RİSKLER VE DÜRÜST DEĞERLENDİRME

Protokolü uygularken bunları unutma:

- **En olası sonuç H0'ın reddedilememesi.** Sinek beyni koku, görme ve
  lokomosyon için evrimleşti; Rastrigin'e veya istinat duvarı boyutlandırmaya
  dair bir tümevarımsal önyargı taşıması için bilinen bir mekanizma yok.
- **Hesaplama maliyeti gerçek bir engel.** 139k LIF nöronu, adayı başına
  yüzlerce milisaniye. Ucuz benchmark'larda bu tek başına yenilgi demek.
- **Kapsam riski.** Kullanıcının aynı anda birden fazla açık manuscript ve iki
  yüksek lisans tezi var. Bu proje onların önüne geçerse net zarar üretir.
  Faz 1 kapısı kasten ucuz ve hızlı — bir hafta harcayıp cevabı öğrenmek,
  bir yıl harcayıp öğrenmekten kıyaslanamaz derecede iyidir.

Bu riskleri projenin aleyhine kanıt çıktıkça kullanıcıya hatırlat. Senin işin
projeyi yaşatmak değil, doğru cevabı bulmak.

---

## 11. EK BÖLÜM (2026-09-20) — H2 spektrum taramasının resmi sonucu

> Bu, dosyanın başındaki ön tescil disiplinine uygun bir EKLEMEDİR — yukarıdaki
> hiçbir madde değiştirilmedi. Tam türetim `EXPERIMENTS.md` (2026-09-17 ila
> 2026-09-20 girişleri) ve `PROTOCOL_V2_TASK_SPECTRUM.md`'de.

### 11.1 Orijinal Faz 1 kapısı (Fly-Proposer / Rastrigin) zaten reddedildi

Bölüm 5'in Faz 1 kapı deneyi (Fly-Proposer, 10-D Rastrigin) daha önce
koşuldu ve **H1'i bu görev sınıfı için reddetti** (soyut optimizasyonda
δ, ~0 ile -0.34 arası — hiçbir varyant eşiği geçemedi). Bölüm 5.235'in
öngördüğü gibi proje durmadı, "seyrek rekürren reservoir'lar" hattına
(H2 — görev-mimari eşleşmesi hipotezi, ayrı belgede) döndü.

### 11.2 H2 taramasında metodolojik bir tuzağa düşüldü ve düzeltildi

H2 taramasının bir kısmı, hız kaygısıyla, Bölüm 3'ün "her deney en az
DegreePreservingNull ve WeightShuffleNull üzerinde de koşar" kuralını
İHLAL ETTİ — birçok yeni deney (malecns F1-sürüş demosu, CX-modulated
arama) sadece ER-null'a karşı test edildi. İki çarpıcı "pozitif" sonuç
(malecns δ=1.000; CX-modulated δ=-0.562) bu şekilde ortaya çıktı.

Bunlar sonradan `degree_preserving_rewire`'a (Bölüm 2'nin asıl
belirlediği null) karşı test edildiğinde **ikisi de tamamen kayboldu**
(malecns: -0.156; CX-modulated: +0.125). Kök neden: FlyWire alt-graflarının
heterojen/çok-kuyruklu derece dağılımı (hub nöronlar) ER-null'da
tamamen yok oluyor — bu tek başına, gerçek bağlantı özgüllüğü olmadan,
çoğu farkı açıklıyor. Bölüm 3'ün kuralı bundan sonra sıkı biçimde
uygulanacak; hiçbir yeni sonuç sadece ER-null'a karşı rapor edilmeyecek.

### 11.3 Tek hayatta kalan bulgu: resmi Faz 1 kriterine göre DOĞRULANDI

Şev stabilitesi (Fellenius yöntemi, tek-atış toplu-regresyon — Bölüm
5'in Faz 3 "gerçek problemler" ailesine yakın, burada ucuz/basitleştirilmiş
biçimde test edildi) Bölüm 2'nin **TAM ön tescilli falsifikasyon
kriteriyle** test edildi:

- Null model: `degree_preserving_rewire` (Bölüm 2'nin belirlediği null)
- n = 30 bağımsız tohum (Bölüm 4.3'ün seed disiplini)
- İstatistik: Mann-Whitney U (Bölüm 4.4)

**Sonuç: p ≈ 0.000000, Cliff's δ = 0.884.** Kriter (p<0.05 VE |δ|>0.33)
açıkça karşılanıyor. Bu, projenin kuruluşundan beri **Bölüm 2'nin
harfine tam uyan ilk ve tek pozitif sonuçtur.**

### 11.4 Kapsam — bu H1'in genel kabulü DEĞİL

Bölüm 10'un "en olası sonuç H0'ın reddedilememesi" öngörüsü genel
black-box optimizasyon için (Rastrigin, §11.1) hâlâ geçerli ve
doğrulandı. Buradaki PASS, dar ve spesifik bir görev sınıfıyla
(uzamsal, tek-atış toplu-regresyon) sınırlı — Fly-Proposer/Rastrigin
gibi genel-amaçlı iteratif arama görevlerinde H1 reddedilmiş durumda.
Yani proje hem H0'ı hem de dar bir H1 istisnasını aynı anda destekleyen
bir kanıt tabanına ulaştı. Bölüm 9'un "pozitif senaryo" yayın hedefi
artık bu dar görev sınıfı için, "negatif senaryo" ise genel-amaçlı
optimizasyon için geçerli — muhtemelen tek bir makale ikisini de
(görev-mimari eşleşmesi çerçevesinde) taşıyabilir.

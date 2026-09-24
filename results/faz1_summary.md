# Faz 1 + Faz 1b Sonuç Özeti (TAMAMLANDI — 2026-09-15)

**Durum:** Tamamlandı. Hem Faz 1 hem Faz 1b bitti, ikisi de gate'i
geçemedi (Faz 1b çok sınırda). Aşağıdaki özet nihaidir.

## Soru

Gerçek FlyWire *Drosophila* connectome'unun (139.255 nöron, 15.091.983
yönlü bağlantı) seyrek rekürren dinamiği, black-box optimizasyonda aynı
istatistiksel özelliklere sahip null modellerden (rastgele graf,
derece-korumalı karıştırılmış graf, ağırlık-karıştırılmış graf) ölçülebilir
şekilde daha iyi bir "proposal operatörü" üretiyor mu?

## Yöntem (kısaca)

Fly-Proposer mimarisi: mevcut çözüm x, seçilen nöronlara akım olarak
enjekte edilir, connectome T=15 adım LIF dinamiğiyle koşulur, seçilen bir
nöron alt kümesinin ateşleme hızı lineer bir projeksiyonla Δx'e dönüştürülür.
Bu, (1+1) elitist hill-climber içinde çalıştırılır. 10-D Rastrigin, 30
paylaşılan seed, 1500 evaluation bütçesi, eşit hiperparametre tarama
bütçesi her substrate için. İstatistik: Wilcoxon signed-rank (eşleştirilmiş
seed), Cliff's delta, eşik p<0.05 ve |δ|>0.33 (PROTOCOL.md §2).

## Sonuç — Faz 1 (rastgele encode, rastgele decode)

**Gate GEÇMEDİ.** flywire vs degree_preserving_null, 30/30 seed:
Wilcoxon p=0.6309, Cliff's δ=0.001. Ortalamalar neredeyse özdeş
(159.55 vs 158.02). Bu, "belki biraz daha veriyle anlamlı çıkar" türü
zayıf bir null değil — dağılımlar pratik olarak örtüşüyor.

**İkincil bulgu (sürpriz):** flywire, er_null'dan (p=4.3×10⁻⁶, δ=0.608,
büyük etki) ve weight_shuffle_null'dan (p=0.028, δ=0.241) istatistiksel
olarak anlamlı şekilde **daha kötü** performans gösteriyor (medyanlar:
er_null 123.75 < weight_shuffle_null 143.93 < flywire 153.56 <
degree_preserving_null 157.27). Yani connectome sadece "null'dan farksız"
değil, bazı basit null'lardan ölçülebilir şekilde daha kötü. Olası
mekanizma (doğrulanmamış hipotez): connectome'un ağır-kuyruklu derece
dağılımı (hub nöronlar) ve gerçek ağırlık-topoloji eşleşmesi, bu soyut
görev için rastgele/homojen alternatiflerden daha az elverişli dinamikler
üretiyor olabilir. Detay: EXPERIMENTS.md 2026-09-15.

## Sonuç — Faz 1b (afferent/efferent encode-decode)

Aynı tasarım, tek fark: encode=afferent havuzundan (n=19.261, gerçek
duyusal giriş nöronları) 10 nöron, decode=efferent havuzundan (n=1.489,
gerçek motor çıkış nöronları) 50 nöron. Sağlık kontrolü temiz (havuzlar
doğru boyutta, kesişim yok, seed disiplini korunmuş).

**Gate YİNE GEÇMEDİ, ama sınırda.** flywire vs degree_preserving_null,
30/30 seed: Wilcoxon p=0.003162 (güçlü anlamlı), Cliff's δ=-0.312
(pre-tescilli |δ|>0.33 eşiğinin **hemen altında** — 0.018 fark). Yön
Faz 1'den farklı: burada flywire, degree_preserving_null'dan **iyi**
çıkıyor (istatistiksel olarak anlamlı ama etki büyüklüğü eşiği
karşılamıyor). Ön tescil kuralı harfiyen uygulandı — "neredeyse geçti"
yuvarlanmadı.

**Ama flywire, HÂLÂ er_null'dan (saf rastgele graf) anlamlı ölçüde kötü**
(p=0.000479, δ=0.392) — bu, Faz 1'le tutarlı, tekrarlanan bir bulgu.

**Faz1 vs Faz1b flywire karşılaştırması (Mann-Whitney):** p=0.24, anlamlı
fark yok (148.86 vs 159.55) — yani **anatomik encode/decode, flywire'ın
kendi performansını istatistiksel olarak değiştirmedi.** Faz1b'deki
gate farkının kaynağı, esas olarak degree_preserving_null'un bu koşumda
biraz daha kötü çıkmasından geliyor (166.93 vs Faz1'in 158.02, kendisi
de anlamlı değil, p=0.42) — yüksek gürültülü bir açıklama ama veriyle
tutarlı; her koşumda degree-preserving null yeniden (farklı rastgele
rewiring ile) üretiliyor.

## Birleşik tablo (Faz 1 + Faz 1b)

| | Faz 1 (rastgele) | Faz 1b (anatomik) |
|---|---|---|
| flywire vs degree_preserving_null | p=0.631, δ=0.001 (geçmedi) | p=0.003, δ=-0.312 (sınırda geçmedi) |
| flywire vs er_null | p=4.3e-6, δ=0.608 (flywire kötü) | p=0.0005, δ=0.392 (flywire kötü) |
| flywire vs weight_shuffle_null | p=0.028, δ=0.241 (flywire kötü) | p=0.381, δ=-0.014 (fark yok) |
| flywire kendisi (iki faz arası) | — | fark yok (p=0.24) |

**Not:** weight_shuffle_null karşılaştırması iki fazda tutarsız çıktı
(Faz 1'de flywire anlamlı kötü, Faz 1b'de fark yok) — bu null model her
koşumda yeniden, farklı rastgele ağırlık karıştırmasıyla üretiliyor;
tutarsızlık gürültü payının bu karşılaştırmada yüksek olduğunu gösteriyor,
güvenilir bir bulgu olarak sayılmamalı. **Tek tutarlı, iki fazda da
tekrarlanan bulgu: flywire vs er_null.**

## Yorum ve literatür bağlamı

- Reservoir computing teorisi (Jaeger 2004, "echo state property") zaten
  "sabit rastgele seyrek rekürren ağ + eğitilmemiş/sabit readout"
  formülünün dinamik modellemede güçlü olduğunu gösteriyor —
  Faz 1'in null sonucu bu teorik beklentiyle tam örtüşüyor
  (literature/map.md §6).
- Bağımsız bir topluluk projesi (`annel0/flybrain`, GitHub) AYNI null
  model ailesiyle (degree-preserving rewiring) connectome'un **kendi
  evrimleştiği bir devrede** (koku işleme, gain kontrolü) null'dan
  ölçülebilir şekilde farklı davrandığını gösteriyor — yani metodoloji
  "kör" değil, connectome yapısı önemli olduğunda bunu yakalayabiliyor.
  Bizim negatif sonucumuz, connectome'un yapısının Rastrigin gibi
  evrimsel bağlamı olmayan soyut bir görevde İŞE YARAMADIĞINI gösteriyor
  — "her şeyde işe yaramıyor" değil, "kendi doğal görev sınıfının dışında
  işe yaramıyor" (literature/map.md §7).
- Bu, Faz 1b'nin (gerçek duyusal/motor nöron popülasyonlarıyla) neden
  bilimsel olarak gerekli bir sonraki adım olduğunu güçlendiriyor:
  belki connectome'un yapısı, kendi doğal girdi/çıktı yollarına daha
  yakın bir görev kurgusunda iş görür.

## Bilinen sınırlamalar

1. Evaluation bütçesi 1500 (protokolün istediği 10k değil) — maliyet
   aşımı nedeniyle küçültüldü, ama adil karşılaştırma (her substrate aynı
   bütçe) korundu; etki büyüklüğü (δ≈0) zaten örneklem büyüklüğüyle
   değişmeyecek bir bulgu.
2. Tek bir encode/decode kombinasyonu test edildi Faz 1'de (Faz 1b ikinci
   kombinasyonu ekliyor); protokolün istediği 4 kombinasyondan 2'si
   (encode=görsel/olfaktör topografik, decode=motor okuma ile tam
   yapılandırılmış versiyon) henüz test edilmedi.

## Genel sonuç (iki faz birlikte)

**H1 desteklenmiyor.** İki farklı encode/decode tasarımında da, gerçek
connectome degree-preserving null'dan güvenilir bir avantaj göstermedi
(bir sefer tamamen farksız, bir sefer istatistiksel olarak anlamlı ama
pre-tescilli etki büyüklüğü eşiğinin altında — yani "kanıtlanmış avantaj"
diyemeyiz). Daha çarpıcısı: **her iki fazda da flywire, saf rastgele
bir graftan (er_null) anlamlı ve orta-büyük etkiyle daha kötü** —
bu tutarlı, tekrarlanan, connectome'un aleyhine olan tek net bulgu.

## Önerilen sonraki adımlar (kullanıcı kararı gerektirir)

Bkz. EXPERIMENTS.md 2026-09-14/15 girişleri ve PROTOCOL.md Faz 1 durma
kapısı metni. Seçenekler:
- **(a) Faz 1c:** okuma katmanını (readout) eğitmek — connectome'un içi
  sabit kalır ama rastgele okuma yerine basit bir regresyonla eğitilmiş
  okuma denenir (reservoir computing'in standart pratiği; LIF simülasyon
  maliyetinin yanında neredeyse bedava). Kullanıcının kendi önerisi
  (2026-09-15 sohbeti) — mevcut negatif sonucun "connectome'da bilgi yok"
  mu yoksa "rastgele okuma bilgiyi bulamadı" mı olduğunu ayırt eder.
- **(b) Negatif sonucu nihai kabul et**, yayın çerçevesine geç
  (PROTOCOL.md §9 negatif senaryo — yayınlanabilir, alan için değerli).
- **(c) Faz 2'ye (6 yeni mimari) geçmeden dur**, kapsamı ve kullanıcının
  diğer aktif projelerini (manuscript/tez) önceliklendirmeyi yeniden
  değerlendir.

Nihai karar kullanıcıya ait.

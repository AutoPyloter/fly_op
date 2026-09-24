# Optimizasyon Nedir? Sezgiden Algoritmaya

## Bir tanım problemiyle başlayalım

"Optimizasyon algoritması nedir?" diye aratırsanız karşınıza şuna benzer bir cümle çıkar: *"Bu algoritmalar, belirli bir probleme en iyi çözümü bulmak için kullanılır."*

Bu tanımı sevmiyorum. Çünkü sayısal, matematiksel bir eylemi tarif ederken "en iyi" gibi bir ifadeye başvurmak bana tuhaf geliyor. İyilik muallak ve öznel bir kavramdır. Matematik bilen ama hayatında hiç optimizasyon duymamış birine bu tanımı okutun; aklına ilk gelecek sorular şunlar olacaktır: *İyiden kasıt ne? İyi olmak nasıl ölçülür? En iyiye kim karar veriyor?*

Tanım, cevap vermek yerine yeni sorular doğuruyor. "A, B'den büyüktür" demek matematiksel olarak anlamlıdır; "A, B'den iyidir" demenin ise matematiksel bir karşılığı yoktur.

İşin aslı şu: öznellik "en iyi" kelimesinde değil, **amaç fonksiyonunun ve kısıtların kim tarafından seçildiğinde** saklı. Standart tanımlar bu modelleme tercihini görünmez bırakıyor. Kriter belirlendikten sonra ise ortada öznel hiçbir şey kalmaz — geriye sadece sayılar arasında bir karşılaştırma kalır.

Bu yüzden benim tanımım şu olurdu:

> **Optimizasyon, tanımlanmış bir amaç fonksiyonunun, uygun çözüm kümesi üzerinde uç değerini aldığı noktaların aranmasıdır.**

Cümlenin her parçası bir iş yapıyor:

**"Tanımlanmış"** — Öznellik burada durur ve burada biter. Amaç fonksiyonunu ve kısıtları biri seçmek zorundadır; o seçim yapılmadan optimizasyon diye bir şey yoktur.

**"Uygun çözüm kümesi üzerinde"** — Kısıtları sağlayan çözümlerin kümesi. Kısıt sağlamak ile uç değer aramak farklı işlerdir: ilki bu kümeyi tanımlar, ikincisi kümenin içinde gerçekleşir.

**"Uç değer"** — "En iyi" değil. Reel bir eksende ekstremum aramak nesnel bir işlemdir.

**"Noktaların"** — Çoğul. Birden fazla optimum olabilir.

**"Aranmasıdır"** — Bulunması değil. Bu fiil tercihinin sebebi yazının ilerleyen kısmında netleşecek; şimdilik şu kadarını söyleyeyim: "bulunur" demek, çoğu yöntemin veremeyeceği bir söz vermektir.

---

## Peki bu arama neye benziyor?

Tanımı bir kenara bırakıp somut bir sahneye geçelim.

Bize bir fonksiyonun minimum noktası sorulsa ne yapardık? Türevini alır, sıfıra eşitler, çıkan noktalardaki değerleri karşılaştırır, en düşüğünü seçerdik. Lisede öğrendiğimiz şey bu.

Peki ya matematik yeteneğimiz körelmişse? Ya fonksiyon fazla karmaşıksa, hatta formülünü hiç bilmiyorsak?

Yine de bir çaremiz var: GeoGebra gibi bir programda grafiğini çizdirir, ortaya çıkan kabartma haritaya bakıp en dip noktayı parmağımızla gösteririz.

Ama burada gözden kaçan bir şey var. GeoGebra o grafiği çizerken aslında ne yaptı? Fonksiyonu belirli aralıklarla tek tek hesapladı ve sonuçları bize gösterdi. Yani bize çözümü sunmadı — **bütün noktaları hesaplayıp önümüze serdi**. Yaptığı şeyin adı var: ızgara araması (*grid search*).

Demek ki bir fonksiyonun en düşük değerini bulmak için bütün değerlerini hesaplamak gerekiyor gibi görünüyor. Fonksiyonun bir yüzeyi var, ama biz onu göremiyoruz. Gözlerimiz bağlı. Her hesap yaptığımızda, önümüzdeki kabartma haritaya bir kez dokunmuş oluyoruz. Haritanın bütün şeklini anlamak için her noktasına dokunmamız gerekiyor sanki.

Ama her zaman öyle değil.

---

## Boncuk

Bir havzanın kabartma haritasını alalım ve üzerine küçük bir boncuk bırakalım. Boncuk, yerçekimi ve bulunduğu yerin eğimi ne diyorsa o yöne yuvarlanır; suyun yolunu bulması gibi havzanın dibine iner.

Peki boncuk ne yaptı? Bütün haritayı gezdi mi? Bütün noktaları hesapladı mı? Hayır. Sadece bulunduğu yerden dibe giden patikaya dokundu ve yalnızca o patikadaki değerleri hesaplamış oldu. Nereye gideceğini söyleyen tek şey, o an bastığı yüzeyin eğimiydi.

İşte bu davranışı taklit edersek, bütün yüzeyi taramadan, çok daha az hesapla minimuma ulaşırız. Anlattığım şeyin adı **gradyan inişi**. Ve bu mantıkla çalışan, aramayı körü körüne taramak yerine akıllı hâle getiren yöntemlere **optimizasyon algoritmaları** diyoruz.

Buradan tanımı yeniden okuyabiliriz: optimizasyon algoritması, uygun çözüm kümesinin tamamını değerlendirmeden, **topladığı bilgiyi kullanarak nereye bakacağına karar veren** bir arama sürecidir.

---

## Neden gerekli: sayılarla

Bu "akıllılık" bir konfor meselesi değil, zorunluluk. Somut bir örnekle görelim.

Bir betonarme istinat duvarı tasarladığınızı düşünün. Altı tasarım değişkeniniz var: taban genişliği, ökçe uzunluğu, gövde üst ve alt kalınlığı, temel kalınlığı, duvar yüksekliği. Her birini 100 kademeye bölelim.

- Kaba kuvvetle tüm uzayı taramak: 100⁶ = **10¹² değerlendirme**
- Her değerlendirme 1 milisaniye sürse: yaklaşık **31 yıl**
- Metasezgisel bir algoritma, 5000 iterasyon: **5000 değerlendirme**, birkaç dakika

Aradaki oran 10⁹. Ve değişken sayısı arttıkça uçurum katlanarak büyüyor. Bunun adı **boyutun laneti** (*curse of dimensionality*) ve optimizasyon algoritmalarının varlık sebebi tam olarak bu tek satırdır.

---

## "Akıllı" ne demek?

Bir arama, daha önce baktığı noktalardan gelen bilgiyi sonraki adımda kullanıyorsa akıllıdır. Izgara araması bunu yapmaz — 500.000'inci noktayı hesaplarken önceki 499.999 noktadan hiçbir şey öğrenmemiştir. Boncuk ise her adımda öğrenir.

Bu bilginin iki kaynağı vardır:

**Yerel bilgi.** Gradyan, eğim, Hessian. Nereye doğru inileceğini söyler. Hızlıdır, ama boncuğun sorunu buradadır: boncuk en yakın çukura iner ve orada durur. O çukurun havzanın en derin yeri olup olmadığını bilemez. Buna **yerel minimum** tuzağı denir.

**Toplu bilgi.** Tek bir boncuk yerine yüzlerce boncuk bıraksaydık ve bunlar birbirlerinin bulduğu iyi bölgeleri paylaşsaydı? Popülasyon tabanlı yöntemlerin fikri budur. Harmony Search'ün hafıza havuzu (*harmony memory*) tam olarak bunu yapar: iyi çözümlerin bileşenlerini saklar ve yeni adaylar üretirken bunları harmanlar.

Her algoritma **keşif** (bilinmeyen bölgelere bakma) ile **sömürü** (iyi bulunan bölgeyi derinleştirme) arasında bir denge kurar. Algoritmalar arasındaki farkların çoğu, bu dengeyi nasıl ayarladıklarından ibarettir.

---

## Gözler gerçekten bağlıyken: kara kutu problemleri

Boncuk analojisinin gizli bir varsayımı var: **yüzeyin eğimini hissedebiliyoruz.**

Ya hissedemiyorsak?

Amaç fonksiyonuna yalnızca bir kâhin gibi erişebildiğiniz durumlar vardır: x verirsiniz, f(x) alırsınız, başka hiçbir şey bilmezsiniz. Buna **kara kutu (black-box) problemi** denir.

Bilmedikleriniz:
- Analitik bir ifade yok, dolayısıyla türev alınamaz
- Fonksiyon konveks mi, kaç yerel minimumu var, bilinmiyor
- Her değerlendirme pahalı olabilir (bir sonlu elemanlar koşusu, bir simülasyon)
- Bazen gürültülü: aynı x, farklı f(x)

Karşıtı **beyaz kutu**tur: doğrusal programlama, konveks karesel problemler. Orada yapıyı bilirsiniz, polinom zamanda çözersiniz ve elinizde optimallik ispatı olur.

İstinat duvarı problemi kara kutuya çok yakındır. Her aday geometri için deprem yönetmeliği kontrollerini çalıştırırsınız — devrilme, kayma, taşıma gücü, kesit kontrolleri. Çıkan sayıların tasarım değişkenlerine göre türevi yoktur.

Burada boncuk çöker. Eğim hissedilemiyorsa yuvarlanacak yön de yoktur.

---

## Metasezgiseller: eğim olmadan arama

Çözüm, eğimi *tahmin etmek* değil, eğime ihtiyaç duymayan arama stratejileri kurmaktır.

Simulated Annealing metalin tavlanmasını taklit eder: başta rastgele ve cesur adımlar atar, zamanla soğuyup yerleşir. Genetik algoritmalar iyi çözümleri çaprazlar ve mutasyona uğratır. Parçacık sürüsü, sürüdeki bireylerin hem kendi en iyisini hem sürünün en iyisini takip etmesiyle çalışır. Harmony Search ise müzisyenlerin doğaçlamasını model alır: hafızadan bir nota seçer, bazen onu biraz kaydırır, bazen tamamen yeni bir nota dener.

Bu yöntemlerin ortak yanı şudur: hiçbiri türev istemez, hepsi keşif-sömürü dengesini kendi mekanizmasıyla kurar ve **hiçbiri global optimumu garanti etmez.**

---

## Bedeli: neden "aranması"

Şimdi tanımın en çok üzerinde durduğum kelimesine geri dönebiliriz.

Uzayın tamamına bakmıyorsanız, elinizdeki en iyi noktanın global optimum olduğunu **ispatlayamazsınız**. Bu bir yöntem kusuru değil, mantıksal bir zorunluluktur: garanti için tam tarama gerekir, o da zaten kaçmaya çalıştığımız şeydir.

**No Free Lunch teoremi** (Wolpert & Macready, 1997) bunu genelleştirir: tüm olası problemler üzerinde ortalandığında hiçbir algoritma diğerinden üstün değildir. Bir algoritma ancak problem hakkında doğru varsayımlar yaptığı ölçüde kazanır. "Her işte en iyi optimizasyon yöntemi" diye bir şey yoktur.

Bu yüzden "optimum çözüm bulunur" demek, yöntemin veremeyeceği bir söz vermektir. Tam algoritmalar — konveks bir problemde iç nokta yöntemi gibi — optimumu bulmayı gerçekten garanti eder. Sezgisel ve metasezgisel yöntemler ise aday çözümler sunar. Tanımın "aranması" demesi, bu iki sınıfı aynı cümlede dürüstçe barındırabilmek içindir.

---

## Tek bir "en iyi" olmadığında

Şimdiye kadar tek bir amaç fonksiyonu varsaydık. Gerçek tasarımda nadiren öyledir.

İstinat duvarında maliyeti düşürmek istersiniz, ama güvenlik paylarını da yükseltmek istersiniz. Bu ikisi çelişir. Hangisi "en iyi"?

Bu soru cevapsızdır, çünkü sorunun kendisi hatalıdır. Çok amaçlı problemlerde tek bir uç değer yoktur; onun yerine **baskınlık ilişkisi** devreye girer. Bir çözüm, başka bir çözümü ancak hiçbir amaçta kötü olmayıp en az birinde daha iyiyse baskılar. Hiçbir çözüm tarafından baskılanmayan çözümlerin kümesine **Pareto cephesi** denir.

Burada matematik size tek bir nokta vermez, bir **menü** verir. Menüden seçmek artık matematiğin değil, karar vericinin işidir — ve öznellik, tanımın başında söylediğimiz yere, kriteri belirleyen tarafa geri döner.

Tanımın çok amaçlı hâli bu yüzden şöyle genişler: *amaç fonksiyonu birden çok bileşen taşıdığında uç değer yerini baskınlık ilişkisine bırakır ve arama tekil bir çözümü değil, baskılanmayan çözümler kümesini hedefler.*

---

## Kara kutuyu gri kutuya çevirmek

Kara kutunun çaresizliği şudur: hiçbir şey bilmiyorsanız her yeri örneklemeniz gerekir.

Kaçış yolu, problemin aslında **tamamen kara kutu olmadığını** fark etmektir.

İstinat duvarında siz biliyorsunuz ki ökçe genişliği taban genişliğini aşamaz. Gövde kalınlığı belli bir orandan ince olamaz. Temel derinliği donma derinliğinin altına inmek zorundadır. Bu bilgi amaç fonksiyonunun içinde yazmaz — **alan bilgisidir**.

Arama uzayını bu ilişkilere göre kurduğunuzda algoritma, geçersiz bölgeleri hiç ziyaret etmez. Kara kutu, gri kutuya dönüşür. Aynı bütçeyle çok daha fazla anlamlı nokta denenmiş olur.

Benzer bir fikir **vekil modellerde** (*surrogate models*) vardır: pahalı değerlendirmeyi ucuz bir yaklaşıklayıcıyla — bir makine öğrenmesi modeliyle — değiştirip arama bütçenizi büyütürsünüz. Bayesian optimizasyon bu ailenin en bilinen üyesidir.

Yani mühendislik bilgisi, optimizasyonun rakibi değil; aramayı akıllı kılan bilginin ta kendisi.

---

## Bir kelime meselesi

TDK optimizasyon için "eniyileme" karşılığını önermiş. Pratik ve anlaşılır bir tercih, ama dikkat edin: bu kelime de doğrudan "en iyi" eksenine oturuyor — yani bu yazının başında rahatsız olduğumuz vurguyu Türkçeye taşıyor.

Başka diller aynı kavramı başka eksenlerde kodlamış. Japonca 最適化 (*saitekika*) "en **uygun** hâle getirme" demek — üstünlük değil, uygunluk. Çince 优化 (*yōuhuà*) "nitelikli kılma". Arapçada التحسين "iyileştirme", الأمثلة ise "ideale yaklaştırma".

Arapçada bu kavram için boşta duran güzel bir kelime var: **tahyîr** (تخيير). خ-ي-ر kökünden — "hayır" kelimesinin de kökü — ve "seçenekleri ayırmak, seçim sunmak" anlamına geliyor. Tanımdaki vurguya "eniyileme"den çok daha yakın: en iyisini yapmak değil, kriterlere göre ayrıştırıp önünüze koymak.

Nitekim bir optimizasyon algoritmasının yaptığı gerçekten budur. Özellikle Pareto durumunda algoritma bir cephe verir; üzerinden birini seçmez.

---

## Toparlarsak

Optimizasyon, tanımlanmış bir amaç fonksiyonunun, uygun çözüm kümesi üzerinde uç değerini aldığı noktaların aranmasıdır.

Bu aramanın anlamlı olmasının sebebi, uygun çözüm kümesinin tamamının değerlendirilmesinin çoğu gerçek problemde imkânsız olmasıdır. Optimizasyon algoritmaları, önceki değerlendirmelerden elde ettikleri bilgiyi kullanarak nereye bakacaklarına karar eder — gözü bağlı birinin haritanın her noktasına dokunmak yerine, eğimi takip ederek dibe inmesi gibi.

Bunun bedeli, bulunan çözümün gerçekten uç değeri verdiğinin ispatlanamamasıdır. Bu yüzden dürüst fiil "bulmak" değil, "aramak"tır.

Ve "en iyi" dediğimiz şey, çoğu zaman bir keşif değil; başta yaptığımız bir seçimin kaçınılmaz sonucudur.

---

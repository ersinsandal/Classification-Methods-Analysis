<div align="center">

# Örüntü Tanımada Sınıflandırma Yöntemlerinin Nicel Karşılaştırmalı Analizi
### Yapısal, Gerçek Dünya ve Doku Veri Setleri Üzerinde Farklı Makine Öğrenmesi Sınıflandırıcılarının HOG Tabanlı Değerlendirilmesi

<br/>

[![English Documentation](https://img.shields.io/badge/Documentation-English-blue?style=for-the-badge&logo=readme)](README.md)

[🇬🇧 **Click here to read the documentation in English**](README.md)

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Alan](https://img.shields.io/badge/Alan-Örüntü%20Tanıma%20%7C%20Bilgisayarlı%20Görü-success?style=flat-square)](#)
[![Lisans](https://img.shields.io/badge/Lisans-MIT-green?style=flat-square)](LICENSE)

**Yazar:** Ersin Sandal  
*Bilgisayar Mühendisi*

</div>

---

## 📑 İçindekiler

- [Yönetici Özeti](#-yönetici-özeti)
- [Motivasyon ve Problem Tanımı](#-motivasyon-ve-problem-tanımı)
- [Deneysel Çerçeve ve Metodoloji](#-deneysel-çerçeve-ve-metodoloji)
  - [Sistem Hattı Genel Bakışı (Pipeline)](#sistem-hattı-genel-bakışı-pipeline)
  - [Kontrol Değişkeni: Sabit HOG Öznitelik Çıkarımı](#kontrol-değişkeni-sabit-hog-öznitelik-çıkarımı)
  - [Bağımsız Değişken: Makine Öğrenmesi Sınıflandırıcıları](#bağımsız-değişken-makine-öğrenmesi-sınıflandırıcıları)
  - [Değerlendirme Metrikleri](#değerlendirme-metrikleri)
- [Kıyaslama Veri Setleri (Benchmark Datasets)](#-kıyaslama-veri-setleri-benchmark-datasets)
- [Nicel Deneysel Sonuçlar](#-nicel-deneysel-sonuçlar)
- [Kapsamlı Tartışma ve Derinlemesine Bulgular](#-kapsamlı-tartışma-ve-derinlemesine-bulgular)
  - [1. Fashion-MNIST: Doğrusal Ayrılabilir Manifoldlar Rejimi](#1-fashion-mnist-doğrusal-ayrılabilir-manifoldlar-rejimi)
  - [2. CIFAR-10: Karmaşık Görsel Bağlam ve Doğrusal Olmayan Dağılımlar](#2-cifar-10-karmaşık-görsel-bağlam-ve-doğrusal-olmayan-dağılımlar)
  - [3. KTH-TIPS: Doku Sınıflandırmasında Boyut Laneti (Curse of Dimensionality)](#3-kth-tips-doku-sınıflandırmasında-boyut-laneti-curse-of-dimensionality)
  - [4. Hesaplama Karmaşıklığı: Eğitim Süresi vs. Çıkarım (Test) Hızı](#4-hesaplama-karmaşıklığı-eğitim-süresi-vs-çıkarım-test-hızı)
- [Görselleştirmeler ve Hata Matrisleri (Confusion Matrices)](#-görselleştirmeler-ve-hata-matrisleri-confusion-matrices)
- [Sentez: 'No Free Lunch' Teoreminin Doğrulanması](#-sentez-no-free-lunch-teoreminin-doğrulanması)
- [Proje Mimarisi ve Dizin Yapısı](#-proje-mimarisi-ve-dizin-yapısı)
- [Kurulum ve Tekrarlanabilirlik Kılavuzu](#-kurulum-ve-tekrarlanabilirlik-kılavuzu)
- [Sunum Dosyası İçeriği](#-sunum-dosyası-içeriği)
- [Alıntı ve Atıf](#-alıntı-ve-atıf)

---

## 🔬 Yönetici Özeti

Bilgisayarlı görü ve örüntü tanıma sistemlerinde nihai sınıflandırma performansı iki temel aşamaya bağlıdır: **öznitelik temsili (feature representation)** ve **karar yüzeyinin öğrenilmesi (classification)**. Literatürdeki deneysel çalışmalarda çoğu zaman öznitelik kalitesi ile sınıflandırıcının algoritma yeteneği birbirine karışmakta, hangi başarının öznitelikten hangi başarının sınıflandırıcıdan kaynaklandığı belirsiz kalmaktadır.

Bu araştırma projesinde, dört temel sınıflandırma paradigması nicel olarak karşılaştırılmıştır:
1. **Destek Vektör Makineleri (Linear SVM)**
2. **K-En Yakın Komşu (KNN)**
3. **Rastgele Orman (Random Forest - RF)**
4. **Çok Katmanlı Algılayıcı (Multi-Layer Perceptron - MLP)**

Sınıflandırıcıların algoritmik davranışlarını, tümevarımsal önyargılarını (inductive bias) ve genelleme kapasitelerini tamamen izole edebilmek adına, öznitelik çıkarım aşaması **Yönlendirilmiş Gradyanlar Histogramı (HOG - Histogram of Oriented Gradients)** kullanılarak tüm deneylerde **kesin olarak sabit tutulmuştur**. Çalışma, üç farklı veri karakteristiğine sahip veri seti üzerinde yürütülmüştür:
- **Fashion-MNIST** (Belirgin silüet ve biçim odaklı, gri tonlamalı, düşük karmaşıklık)
- **CIFAR-10** (Gerçek dünya nesneleri, karmaşık arka plan, orta-yüksek karmaşıklık)
- **KTH-TIPS** (Aşırı yüksek boyutlu doku sınıflandırması, $d=34.596$, yüksek karmaşıklık)

### Temel Deneysel Çıkarımlar:
- **Evrensel Olarak En İyi Model Yoktur:** Elde edilen veriler, makine öğrenmesindeki *No Free Lunch (Bedava Öğle Yemeği Yok)* teoremini ampirik olarak doğrulamaktadır; sınıflandırıcının başarısı doğrudan öznitelik uzayının geometrisine ve veri setinin iç yapısına bağlıdır.
- **Doğrusal Ayrılabilirlikte Modellerin Eşitliği:** Basit ve düzenli silüet verisinde (Fashion-MNIST), dört sınıflandırıcının tamamı neredeyse birebir aynı başarıyı göstermiştir (~%85.5 – %86.2 Doğruluk, Macro F1 ~0.858 – 0.865).
- **Karmaşık Sahnelerde Model Kapasitesi:** Karmaşık gerçek dünya görüntülerinde (CIFAR-10), **MLP** doğrusal olmayan yapay sinir ağı katmanları sayesinde en yüksek doğruluğu (%47.80) elde etmiş, doğrusal SVM ve ağaç tabanlı modelleri geride bırakmıştır.
- **Boyut Laneti ve Uzaklık Metriğinin Çöküşü:** Yüksek çözünürlüklü doku verisinde (KTH-TIPS, $N=648$ eğitime karşılık $d=34.596$ boyut), **Lineer SVM %64.81 doğruluk** (%70.89 kesinlik) ile açık ara zirvede yer alırken, Öklid uzaklığına dayanan **KNN aşırı boyutlanma nedeniyle çökerek %23.46 doğruluğa** gerilemiştir.

---

## 🎯 Motivasyon ve Problem Tanımı

Örüntü tanıma ve bilgisayarlı görü projelerinde en sık karşılaşılan soru şudur: *“Belirli bir görsel sınıflandırma görevi için hangi algoritma seçilmelidir?”* Çoğu zaman geliştiriciler, veri seti boyutu, öznitelik uzayı ve algoritma teorisi arasındaki etkileşimi hesaba katmadan rastgele model seçimi yapmaktadır.

### Karşılaşılan Sorunlar:
- Aynı öznitelik vektörü üzerinde farklı sınıflandırıcılar radikal biçimde farklı karar sınırları çizer.
- Yalnızca tek bir metriğe (örneğin ham doğruluk oranına) odaklanmak; kesinlik, duyarlılık ve çıkarım gecikmesi (inference latency) arasındaki kritik ödünleşimleri gizler.
- Yanlış model seçimi, özellikle yüksek boyutlu veya karmaşık veri setlerinde ciddi performans kayıplarına yol açar.

### Araştırmanın Amacı:
Öznitelik temsili kaynaklı varyansı ortadan kaldırmak için **HOG parametrelerini sabit tutmak**, böylece farklı veri karakteristikleri altında sınıflandırıcıların saf algoritmik tepkilerini ve matematiksel davranışlarını nicel olarak ortaya koymaktır.

```
+--------------------------------------------------------------------------------------------------+
|                                    DENEYSEL KONTROL PARADİGMASI                                  |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   [ Giriş Görüntüsü ] ---> [ SABİT Öznitelik: HOG ] ---> [ DEĞİŞKEN Sınıflandırıcı ] ---> Metrik|
|                                                                                                  |
|   Sabit Tutulan Parametreler:                               Algoritmalar:                        |
|   * orientations = 9                                         1. Linear SVM                       |
|   * pixels_per_cell = (4, 4)                                 2. K-Nearest Neighbors (k=5)        |
|   * cells_per_block = (2, 2)                                 3. Random Forest (100 ağaç)         |
|   * block_norm = 'L2-Hys'                                    4. Multi-Layer Perceptron (128 birim|
+--------------------------------------------------------------------------------------------------+
```

---

## 🛠 Deneysel Çerçeve ve Metodoloji

### Sistem Hattı Genel Bakışı (Pipeline)

Sistem mimarisi; veri yükleme, gri tonlama dönüşümü, parlaklık normalizasyonu, HOG öznitelik çıkarımı, model eğitimi/tahmini ve metrik raporlamasını uçtan uca otomatikleştirir.

```mermaid
flowchart TD
    A["Ham Veri Seti<br/>(Fashion-MNIST / CIFAR-10 / KTH-TIPS)"] --> B["Görüntü Ön İşleme<br/>- Gri Seviye Dönüşümü (Ağırlıklı Parlaklık Formülü)<br/>- Piksel Normalizasyonu [0.0, 1.0]<br/>- Boyutlandırma (KTH-TIPS: 128x128)"]
    B --> C["Öznitelik Çıkarımı (KESİNLİKLE SABİT)<br/>Histogram of Oriented Gradients (HOG)<br/>- Yönelim: 9 açı aralığı [0°-180°]<br/>- Hücre Boyutu: 4x4 piksel<br/>- Blok Boyutu: 2x2 hücre<br/>- Normalizasyon: L2-Hys"]
    C --> D{"Sınıflandırıcı Karşılaştırması"}
    D --> E1["Destek Vektör Makineleri<br/>(Linear Kernel SVM)"]
    D --> E2["K-En Yakın Komşu<br/>(k = 5, Öklid Uzaklığı)"]
    D --> E3["Rastgele Orman<br/>(100 Karar Ağacı, Bagging)"]
    D --> E4["Çok Katmanlı Algılayıcı<br/>(128 Gizli Birim, ReLU, Adam)"]
    E1 --> F["Kapsamlı Değerlendirme Paketi<br/>- Accuracy, Precision (Macro), Recall (Macro), Macro F1<br/>- Eğitim Süresi & Test/Çıkarım Süresi (Saniye)<br/>- Karışıklık Matrisleri & Karşılaştırma Grafikleri"]
    E2 --> F
    E3 --> F
    E4 --> F
```

---

### Kontrol Değişkeni: Sabit HOG Öznitelik Çıkarımı

Histogram of Oriented Gradients ([Dalal & Triggs, 2005](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)), yerel kenar ve gradyan yönelimlerini yakalarken aydınlatma değişimlerine karşı blok normalizasyonu ile dayanıklılık sağlar.

#### Algoritmik Parametreler:
$$\text{Yönelim Sayısı (Orientations)} = 9 \quad ([0^\circ, 180^\circ] \text{ aralığında işaretsiz})$$
$$\text{Hücre Başına Piksel (Pixels per cell)} = (4, 4)$$
$$\text{Blok Başına Hücre (Cells per block)} = (2, 2)$$
$$\text{Blok Normalizasyonu} = \text{L2-Hys (0.2 eşik kırpmalı L2 normalizasyonu)}$$

#### Elde Edilen Öznitelik Boyutları:
- **Fashion-MNIST** ($28 \times 28$): $\left(\frac{28}{4} - 2 + 1\right) \times \left(\frac{28}{4} - 2 + 1\right) \times 2 \times 2 \times 9 = 6 \times 6 \times 36 = \mathbf{1.296} \text{ boyut}$
- **CIFAR-10** ($32 \times 32$): $\left(\frac{32}{4} - 2 + 1\right) \times \left(\frac{32}{4} - 2 + 1\right) \times 2 \times 2 \times 9 = 7 \times 7 \times 36 = \mathbf{1.764} \text{ boyut}$
- **KTH-TIPS** ($128 \times 128$): $\left(\frac{128}{4} - 2 + 1\right) \times \left(\frac{128}{4} - 2 + 1\right) \times 2 \times 2 \times 9 = 31 \times 31 \times 36 = \mathbf{34.596} \text{ boyut}$

---

### Bağımsız Değişken: Makine Öğrenmesi Sınıflandırıcıları

Dört farklı makine öğrenmesi yaklaşımı, farklı tümevarımsal önyargıları temsil edecek şekilde yapılandırılmıştır:

| Sınıflandırıcı | Model Türü | Hiperparametreler | Temel Özellik ve Davranış |
| :--- | :--- | :--- | :--- |
| **SVM** | Maksimum Marjinli Doğrusal Ayrıcı | `kernel='linear'`, $C=1.0$ | Sınıflar arası geometrik mesafeyi (marjini) maksimize eden hiper-düzlemi bulur. Yüksek boyutlu uzaylarda ($d \gg n$) son derece dayanıklıdır. |
| **KNN** | Örnek Tabanlı Bellek Öğrenicisi | `n_neighbors=5`, metric=`minkowski` ($p=2$) | Tembel öğrenicidir (lazy learner); eğitim maliyeti sıfırdır ancak test esnasında tüm eğitim verisiyle Öklid uzaklığı hesaplar. Boyut lanetine karşı çok hassastır. |
| **Random Forest** | Torbalanmış Karar Ağacı Topluluğu | `n_estimators=100`, criterion=`gini` | Rastgele öznitelik alt uzayları üzerinde eğitilen 100 karar ağacının oylamasıyla çalışır. Aşırı öğrenmeye dayanıklıdır. |
| **MLP** | İleri Beslemeli Yapay Sinir Ağı | `hidden_layer_sizes=(128,)`, `max_iter=300`, solver=`adam`, activation=`relu` | Geriye yayılım (backpropagation) ile ağırlıkları optimize eden katmanlı sinir ağıdır. Doğrusal olmayan karmaşık örüntüleri modelleyebilir. |

---

### Değerlendirme Metrikleri

Sınıf dengesizliklerinden bağımsız, tarafsız bir değerlendirme için tüm metriklerde makro ortalama (macro-average) kullanılmıştır:

$$\text{Doğruluk (Accuracy)} = \frac{\sum_{c=1}^C TP_c}{N_{\text{toplam}}}$$

$$\text{Kesinlik (Precision)}_{\text{makro}} = \frac{1}{C}\sum_{c=1}^C \frac{TP_c}{TP_c + FP_c}, \quad \text{Duyarlılık (Recall)}_{\text{makro}} = \frac{1}{C}\sum_{c=1}^C \frac{TP_c}{TP_c + FN_c}$$

$$\text{Makro } F_1 = 2 \cdot \frac{\text{Precision}_{\text{makro}} \cdot \text{Recall}_{\text{makro}}}{\text{Precision}_{\text{makro}} + \text{Recall}_{\text{makro}}}$$

Ayrıca modellerin pratik uygulanabilirliğini ölçmek amacıyla **Eğitim Süresi (s)** ve **Test/Çıkarım Süresi (s)** yüksek çözünürlüklü zamanlayıcılarla kaydedilmiştir.

---

## 📊 Kıyaslama Veri Setleri (Benchmark Datasets)

Çalışmada üç farklı görsel karmaşıklık düzeyini temsil eden veri setleri kullanılmıştır:

```
+---------------------------------------------------------------------------------------------+
| Veri Seti     | Giriş Boyutu     | Kanallar | Eğitim Örnek  | Test Örnek   | Sınıf   | Karakter    |
+---------------+------------------+----------+---------------+--------------+---------+---------+
| Fashion-MNIST | 28 x 28          | 1 (Gri)  | 10.000*       | 2.000*       | 10      | Şekil   |
| CIFAR-10      | 32 x 32          | 3 -> 1   | 10.000*       | 2.000*       | 10      | Doğal   |
| KTH-TIPS      | 128 x 128        | 1 (Gri)  | 648 (%80)     | 162 (%20)    | 10      | Doku    |
+---------------------------------------------------------------------------------------------+
*Deneylerin eşit şartlarda ve makul sürede çalışması için dengeli olarak 10.000 eğitim / 2.000 test ile sınırlandırılmıştır.
```

1. **Fashion-MNIST:** 10 giyim kategorisi (Tişört, Pantolon, Kazak, Elbise, Palto, Sandalet, Gömlek, Spor Ayakkabı, Çanta, Yarım Bot). Arka plansız, net silüet hatlarına sahip gri seviye görüntüler.
2. **CIFAR-10:** 10 gerçek dünya nesnesi (Uçak, Otomobil, Kuş, Kedi, Geyik, Köpek, Kurbağa, At, Gemi, Kamyon). Karmaşık arka planlar, değişken açılar ve ışık koşulları. Standart parlaklık formülü ile griye dönüştürülmüştür:
   $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
3. **KTH-TIPS (Farklı Aydınlatma, Poz ve Ölçekte Dokular):** 10 malzeme sınıfı (Alüminyum folyo, Esmer ekmek, Fitilli kadife, Pamuk, Kraker, Keten, Portakal kabuğu, Zımpara kağıdı, Sünger, Köpük). Az örnek sayısına ($N_{\text{eğitim}} = 648$) karşın devasa öznitelik boyutu ($d = 34.596$).

---

## 📈 Nicel Deneysel Sonuçlar

`results/metrics/classifier_summary_results.csv` dosyasından derlenen resmi deney sonuçları tablosu:

| Veri Seti | Sınıflandırıcı | Doğruluk (Acc) | Makro Precision | Makro Recall | Makro F1 | Eğitim Süresi (s) | Test Süresi (s) | Öznitelik Boyutu |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fashion-MNIST** | **SVM (Linear)** | 0.8580 | 0.8608 | 0.8599 | 0.8600 | 12.12 s | 4.04 s | $(10000, 1296) \to (2000, 1296)$ |
| **Fashion-MNIST** | **KNN (k=5)** | 0.8555 | 0.8616 | 0.8569 | 0.8581 | **0.02 s** | 2.73 s | $(10000, 1296) \to (2000, 1296)$ |
| **Fashion-MNIST** | **Random Forest** | 0.8575 | 0.8607 | 0.8591 | 0.8597 | 31.58 s | 0.08 s | $(10000, 1296) \to (2000, 1296)$ |
| **Fashion-MNIST** | **MLP (128)** | **0.8620** | **0.8663** | **0.8636** | **0.8647** | 28.20 s | **0.01 s** | $(10000, 1296) \to (2000, 1296)$ |
| | | | | | | | | |
| **CIFAR-10** | **SVM (Linear)** | 0.4375 | 0.4428 | 0.4357 | 0.4369 | 71.10 s | 12.78 s | $(10000, 1764) \to (2000, 1764)$ |
| **CIFAR-10** | **KNN (k=5)** | 0.3990 | 0.4960 | 0.3913 | 0.3805 | **0.03 s** | 1.08 s | $(10000, 1764) \to (2000, 1764)$ |
| **CIFAR-10** | **Random Forest** | 0.4195 | 0.4124 | 0.4159 | 0.4113 | 57.91 s | 0.11 s | $(10000, 1764) \to (2000, 1764)$ |
| **CIFAR-10** | **MLP (128)** | **0.4780** | **0.4750** | **0.4748** | **0.4740** | 66.38 s | **0.01 s** | $(10000, 1764) \to (2000, 1764)$ |
| | | | | | | | | |
| **KTH-TIPS** | **SVM (Linear)** | **0.6481** | **0.7089** | **0.6493** | **0.6576** | 60.75 s | 8.44 s | $(648, 34596) \to (162, 34596)$ |
| **KTH-TIPS** | **KNN (k=5)** | 0.2346 | 0.4126 | 0.2320 | 0.1561 | **0.04 s** | 0.46 s | $(648, 34596) \to (162, 34596)$ |
| **KTH-TIPS** | **Random Forest** | 0.4198 | 0.4513 | 0.4206 | 0.4167 | 13.03 s | 0.03 s | $(648, 34596) \to (162, 34596)$ |
| **KTH-TIPS** | **MLP (128)** | 0.5926 | 0.6440 | 0.5938 | 0.6024 | 102.76 s | **0.03 s** | $(648, 34596) \to (162, 34596)$ |

---

## 🔍 Kapsamlı Tartışma ve Derinlemesine Bulgular

### 1. Fashion-MNIST: Doğrusal Ayrılabilir Manifoldlar Rejimi

```
Fashion-MNIST Doğruluk Dağılımı:
MLP           [===================================================] %86.20
SVM           [================================================== ] %85.80
Random Forest [================================================== ] %85.75
KNN           [================================================== ] %85.55
```

- **Gözlem:** En düşük sonuç veren model (KNN: %85.55) ile en yüksek sonuç veren model (MLP: %86.20) arasındaki fark **yalnızca %0.65**'tir.
- **Teorik Açıklama:** Fashion-MNIST veri setinde nesneler beyaz/siyah zemin üzerinde izole biçimdedir. HOG gradyanları sınıfları birbirinden ayıran güçlü ve doğrusal bir öznitelik uzayı oluşturur. Öznitelikler sınıfları temiz bir şekilde ayırdığında, **sınıflandırıcı algoritmasının karmaşıklığı performansı neredeyse hiç etkilemez**.
- **Hata Dağılımı:** Karışıklık matrisleri incelendiğinde, tüm modellerin benzer şekilde biçimsel olarak birbirine çok benzeyen sınıflarda (örneğin `Gömlek` ile `Tişört` ve `Palto`) kararsız kaldığı görülür. Bu da kısıtın sınıflandırıcıdan değil, biçim benzerliğinden kaynaklandığını kanıtlar.

---

### 2. CIFAR-10: Karmaşık Görsel Bağlam ve Doğrusal Olmayan Dağılımlar

```
CIFAR-10 Doğruluk Dağılımı:
MLP           [===================================                 ] %47.80
SVM           [===============================                     ] %43.75
Random Forest [==============================                      ] %41.95
KNN           [============================                        ] %39.90
```

- **Gözlem:** Veri setinin karmaşıklığı arttığında tüm modellerin doğruluğu belirgin biçimde düşmüştür (%40 - %48 aralığı). Ancak **MLP diğer tüm modellerin önüne geçmiş (+%4.05 SVM'e karşı, +%5.85 RF'ye karşı, +%7.90 KNN'e karşı)**.
- **Teorik Açıklama:** CIFAR-10 doğal fotoğraflardan oluşur; araba, at veya kedi gibi sınıflar çok farklı açılardan çekilmiş ve karmaşık arka planlara sahiptir. Düşük çözünürlükte el yapımı HOG gradyanları bu karmaşık anlamsal bağlamı tam olarak ifade edemez.
- **MLP Neden Kazandı?:** Öznitelik uzayı gürültülü ve doğrusal olmayan ilişkiler barındırdığında, MLP'nin gizli katmanlarındaki doğrusal olmayan aktivasyon fonksiyonları (ReLU) karmaşık karar sınırları inşa edebilme kapasitesi sunar. Doğrusal SVM veya eksene paralel bölmeler yapan Rastgele Orman bu dağılımı modellemekte yetersiz kalmıştır.

---

### 3. KTH-TIPS: Doku Sınıflandırmasında Boyut Laneti (Curse of Dimensionality)

```
KTH-TIPS Doğruluk Dağılımı (d = 34.596):
SVM           [===============================================     ] %64.81
MLP           [==========================================          ] %59.26
Random Forest [==============================                      ] %41.98
KNN           [=================                                   ] %23.46 (ÇÖKTÜ)
```

- **Gözlem:** Modeller arasındaki en dramatik performans uçurumu bu veri setinde yaşanmıştır. **Lineer SVM %64.81 doğruluk ve %70.89 kesinlik ile açık ara en başarılı model** olurken, **KNN %23.46 doğruluğa ve %15.61 Macro F1 değerine çakılmıştır**.
- **Matematiksel Açıklama — Boyut Laneti (Curse of Dimensionality):**
  $d = 34.596$ öznitelik boyutuna karşılık yalnızca $N = 648$ eğitim örneği bulunmaktadır ($d \gg N$). Çok yüksek boyutlu uzaylarda en yakın komşu ile en uzak komşu arasındaki mesafe oranı 1'e yaklaşır:
  $$\lim_{d \to \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \to 0$$
  Öklid uzaklığı tüm noktalar arasında neredeyse eşit hale geldiği için KNN'in en yakın komşuyu seçme mantığı çöker ve model rastgele tahmine yakın sonuçlar üretir.
- **Lineer SVM Neden Zirvede?:**
  Lineer SVM, sınıflar arasındaki marjini maksimize eden konveks bir optimizasyon çözer:
  $$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i$$
  $d \gg N$ durumlarında veriler hemen her zaman yüksek boyutlu uzayda doğrusal olarak ayrılabilirdir. SVM marjin maksimizasyonu sayesinde aşırı öğrenmeye (overfitting) düşmeden global optimumu bulur ve yüksek genelleme başarısı sergiler.

---

### 4. Hesaplama Karmaşıklığı: Eğitim Süresi vs. Çıkarım (Test) Hızı

```
+---------------+------------------------+------------------------+
| Sınıflandırıcı| Eğitim Süresi          | Test / Çıkarım Süresi  |
+---------------+------------------------+------------------------+
| KNN           | O(1) [0.02s - 0.04s]   | O(N*d) [0.46s - 4.04s] |
| Random Forest | O(T * N * log N * d')  | O(T * derinlik) [0.03s]|
| SVM (Lineer)  | O(N^2 * d) [12s - 71s] | O(N_sv * d) [4s - 13s] |
| MLP           | O(E * N * d * h) [28s] | O(d * h) [< 0.01s]     |
+---------------+------------------------+------------------------+
```

- **KNN'in Çıkarım Darboğazı:** KNN modeli neredeyse 0 saniyede eğitilmesine ($0.02$ saniye) karşın, test esnasında her bir sorgu için tüm eğitim kümesiyle $O(N \cdot d)$ uzaklık hesaplamak zorundadır. Fashion-MNIST test kümesinde ($N=2000$) KNN'in tahmini **2.73 saniye** sürerken, MLP yalnızca **0.008 saniyede** tahmin üretmektedir (**>300 kat daha hızlı çıkarım**).
- **Pratik Sistemler İçin Çıkarım:** Gerçek zamanlı veya gömülü sistemlerde **MLP ve Rastgele Orman**, eğitim süresi gerektirmelerine rağmen mikrosaniye düzeyindeki çıkarım hızlarıyla en ideal üretim modelleridir.

---

## 🖼 Görselleştirmeler ve Hata Matrisleri (Confusion Matrices)

### Metrik Karşılaştırma Grafikleri

| Metrik | Fashion-MNIST | CIFAR-10 | KTH-TIPS |
| :--- | :---: | :---: | :---: |
| **Doğruluk (Accuracy)** | ![Fashion Accuracy](results/figures/fashion_mnist_accuracy.png) | ![CIFAR10 Accuracy](results/figures/cifar10_accuracy.png) | ![KTH Accuracy](results/figures/kth_tips_accuracy.png) |
| **Makro F1** | ![Fashion F1](results/figures/fashion_mnist_macro_f1.png) | ![CIFAR10 F1](results/figures/cifar10_macro_f1.png) | ![KTH F1](results/figures/kth_tips_macro_f1.png) |
| **Makro Kesinlik** | ![Fashion Precision](results/figures/fashion_mnist_precision.png) | ![CIFAR10 Precision](results/figures/cifar10_precision.png) | ![KTH Precision](results/figures/kth_tips_precision.png) |
| **Makro Duyarlılık** | ![Fashion Recall](results/figures/fashion_mnist_recall.png) | ![CIFAR10 Recall](results/figures/cifar10_recall.png) | ![KTH Recall](results/figures/kth_tips_recall.png) |
| **Eğitim Süresi** | ![Fashion Train Time](results/figures/fashion_mnist_train_time.png) | ![CIFAR10 Train Time](results/figures/cifar10_train_time.png) | ![KTH Train Time](results/figures/kth_tips_train_time.png) |
| **Test Süresi** | ![Fashion Test Time](results/figures/fashion_mnist_test_time.png) | ![CIFAR10 Test Time](results/figures/cifar10_test_time.png) | ![KTH Test Time](results/figures/kth_tips_test_time.png) |

---

### Karışıklık Matrisleri (Confusion Matrices)

#### 1. Fashion-MNIST Karışıklık Matrisleri
| Linear SVM | K-Nearest Neighbors |
| :---: | :---: |
| ![Fashion SVM CM](results/confusion_matrices/fashion_mnist_svm_cm.png) | ![Fashion KNN CM](results/confusion_matrices/fashion_mnist_knn_cm.png) |
| **Random Forest** | **Multi-Layer Perceptron** |
| ![Fashion RF CM](results/confusion_matrices/fashion_mnist_random_forest_cm.png) | ![Fashion MLP CM](results/confusion_matrices/fashion_mnist_mlp_cm.png) |

#### 2. CIFAR-10 Karışıklık Matrisleri
| Linear SVM | K-Nearest Neighbors |
| :---: | :---: |
| ![CIFAR SVM CM](results/confusion_matrices/cifar10_svm_cm.png) | ![CIFAR KNN CM](results/confusion_matrices/cifar10_knn_cm.png) |
| **Random Forest** | **Multi-Layer Perceptron** |
| ![CIFAR RF CM](results/confusion_matrices/cifar10_random_forest_cm.png) | ![CIFAR MLP CM](results/confusion_matrices/cifar10_mlp_cm.png) |

#### 3. KTH-TIPS Karışıklık Matrisleri
| Linear SVM | K-Nearest Neighbors |
| :---: | :---: |
| ![KTH SVM CM](results/confusion_matrices/kth_tips_svm_cm.png) | ![KTH KNN CM](results/confusion_matrices/kth_tips_knn_cm.png) |
| **Random Forest** | **Multi-Layer Perceptron** |
| ![KTH RF CM](results/confusion_matrices/kth_tips_random_forest_cm.png) | ![KTH MLP CM](results/confusion_matrices/kth_tips_mlp_cm.png) |

---

## 💡 Sentez: 'No Free Lunch' Teoreminin Doğrulanması

Bu çalışma, örüntü tanımada Wolpert'in **No Free Lunch Teoremi**'ni deneysel verilerle net biçimde ortaya koymaktadır:

$$\text{Performans} = f(\text{Öznitelik Önyargısı}, \text{Sınıflandırıcı Önyargısı}, \text{Veri Manifold Geometrisi})$$

```
+--------------------------------------------------------------------------------------------------+
|                                  UYGULAYICILAR İÇİN KARAR MATRİSİ                                |
+-------------------+-----------------------------------+------------------------------------------+
| Veri Tipi         | Önerilen Sınıflandırıcı           | Temel Teorik Gerekçe                     |
+-------------------+-----------------------------------+------------------------------------------+
| Yüksek Boyutlu    | Destek Vektör Makineleri          | Konveks marjin maksimizasyonu; boyut     |
| (d >> N, Doku)    | (Lineer SVM)                      | lanetine ve uzaklık çöküşüne bağışık.    |
+-------------------+-----------------------------------+------------------------------------------+
| Karmaşık / Doğrusal| Çok Katmanlı Algılayıcı          | Doğrusal olmayan evrensel yaklaşıklık;   |
| Olmayan (Nesne)   | (MLP)                             | hiyerarşik dahili öznitelikler üretir.   |
+-------------------+-----------------------------------+------------------------------------------+
| Temiz Silüet      | Rastgele Orman / Lineer SVM /     | Öznitelikler sınıfları zaten doğrusal    |
| (Düzenli Şekil)   | KNN (Herhangi bir temel model)    | olarak ayırmaktadır; karmaşıklık gereksiz|
+-------------------+-----------------------------------+------------------------------------------+
| Düşük Gecikmeli   | Çok Katmanlı Algılayıcı (MLP) /   | Matris çarpımına dayalı hızlı ileri yayılım|
| Kenar Çıkarımı    | Rastgele Orman (RF)               | milisaniyenin altında tahmin süresi.     |
+-------------------+-----------------------------------+------------------------------------------+
```

---

## 📁 Proje Mimarisi ve Dizin Yapısı

```
pr-classifier-project/
│
├── config.py                     # Merkezi konfigürasyon (yollar, limitler, sabitler)
├── main.py                       # Tüm hattı tek komutla çalıştıran ana giriş noktası
├── requirements.txt              # Proje Python kütüphane bağımlılıkları
├── .gitignore                    # Önbellek ve ham yığın verileri hariç tutan git kuralları
├── README.md                     # Ana İngilizce akademik rapor ve dokümantasyon
├── README_TR.md                  # Kapsamlı Türkçe akademik dokümantasyon
│
├── src/                          # Modüler Python kaynak kodları
│   ├── __init__.py
│   ├── classifiers.py            # SVM, KNN, RF ve MLP modellerinin tanımlanması
│   ├── data_loader.py            # Veri setlerinin yüklenmesi, griye çevrilmesi ve ölçeklenmesi
│   ├── features.py               # Standartlaştırılmış sabit HOG öznitelik çıkarımı
│   ├── train_eval.py             # Model eğitimi, süre ölçümü ve metrik hesaplama
│   ├── visualization.py          # Karşılaştırma grafikleri ve karışıklık matrisi çizimleri
│   └── experiments.py            # Tüm deney döngülerini yöneten orkestratör
│
├── data/                         # Veri seti dizin yapısı
│   ├── raw/
│   │   └── KTH_TIPS/             # 10 doku sınıfı (810 görüntü)
│   └── processed/                # Ara işlem tensörleri için yer tutucu (.gitkeep)
│
├── results/                      # Üretilen deneysel çıktılar
│   ├── metrics/
│   │   └── classifier_summary_results.csv   # Birleştirilmiş resmi metrik tablosu (CSV)
│   ├── figures/                  # 18 adet metrik karşılaştırma grafiği (PNG)
│   └── confusion_matrices/       # 12 adet yüksek çözünürlüklü karışıklık matrisi (PNG)
│
├── notebooks/                    # İnteraktif keşif ve analiz Jupyter Defterleri
│   ├── 01_data_check.ipynb
│   ├── 02_hog_preview.ipynb
│   └── 03_result_analysis.ipynb
```

---

## 🚀 Kurulum ve Tekrarlanabilirlik Kılavuzu

### Gereksinimler
- Python 3.10, 3.11, 3.12 veya 3.13
- Git

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/<kullanici-adiniz>/pr-classifier-project.git
cd pr-classifier-project
```

### 2. Sanal Ortam Oluşturun ve Aktif Edin
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Tüm Deneyleri Çalıştırın
```bash
python main.py
```
*Bu komut sırasıyla:*
1. Fashion-MNIST, CIFAR-10 ve KTH-TIPS veri setlerini hazırlar.
2. Sabit HOG parametreleri ile öznitelik vektörlerini çıkarır.
3. SVM, KNN, Random Forest ve MLP modellerini eğitir ve test eder.
4. Sonuçları `results/metrics/classifier_summary_results.csv` dosyasına kaydeder.
5. Tüm metrik karşılaştırma grafiklerini `results/figures/` içerisine oluşturur.
6. Karışıklık matrislerini `results/confusion_matrices/` dizinine kaydeder.

---

## 📑 Akademik Sunum ve Savunma Özeti

Bu araştırma çalışması, lisans eğitimi **Örüntü Tanıma (Pattern Recognition)** dersi kapsamında 37 slaytlık kapsamlı bir akademik sözlü savunma ve sunum olarak gerçekleştirilmiştir. Sunumun ana başlıkları ve konu dağılımı aşağıda özetlenmiştir:

### Sunum Dosyası Mimarisi ve Konu Dağılımı:
- **Slayt 1–6:** Araştırmaya Giriş, Problem Tanımı, Amaçlar ve HOG ile Kontrollü Deney Mimarisi
- **Slayt 7–12:** Veri Setleri Özellikleri, Sınıflandırıcı Matematiği ve Değerlendirme Metrikleri
- **Slayt 13–20:** 1. Deney: Fashion-MNIST Tahminleri, Metrik Grafikleri ve Karışıklık Matrisleri
- **Slayt 21–27:** 2. Deney: CIFAR-10 Beklentileri, Derin Analiz ve Model Kapasitesi Bulguları
- **Slayt 28–34:** 3. Deney: KTH-TIPS Doku Analizi, Metrik Çöküşü ve Boyut Laneti (Curse of Dimensionality)
- **Slayt 35–37:** Veri Setleri Arası Karşılaştırmalı Bulgular, Sonuç ve Algoritmik Sentez

---

## 📜 Alıntı ve Atıf

Bu çalışmayı, kaynak kodlarını veya deneysel metodolojiyi akademik bir yayında veya teknik bir projede kullanırsanız, lütfen aşağıdaki şekilde atıfta bulunun:

```bibtex
@misc{sandal2026prclassifier,
  author       = {Ersin Sandal},
  title        = {Quantitative Comparative Analysis of Classification Methods in Pattern Recognition: HOG-based Evaluation of Diverse Classifiers},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/ersinsandal/Classification-Methods-Analysis}},
  note         = {Örüntü Tanıma Araştırma Çalışması}
}
```

---

<div align="center">
  <sub><b>Ersin Sandal</b>, Bilgisayar Mühendisi tarafından geliştirilmiş ve belgelenmiştir. MIT Lisansı ile sunulmaktadır.</sub>
</div>

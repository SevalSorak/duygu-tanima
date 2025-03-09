# Duygu Tanıma Projesi

Bu proje, derin öğrenme modellerini kullanarak yüz ifadelerini algılayıp sınıflandıran bir duygu tanıma sistemidir.

## 📌 Proje Hakkında
- Yüz ifadelerini algılayarak **mutlu, üzgün, kızgın, korkmuş** gibi duygu sınıflarına ayırır.
- **Veri seti:** FER2013 veri setinden faydalanılmıştır.
- **Model:** VGG, ViT ve farklı CNN modelleri denenmiş, en iyi sonuç VGG modeli ile alınmıştır (%72.55 test doğruluğu).
- **Veri işleme:** Normalizasyon uygulandı ve veri arttırma (data augmentation) teknikleri kullanıldı.
- **Gerçek zamanlı tahmin:** Model, Streamlit tabanlı bir arayüz üzerinden çalıştırılarak kullanıcının yüklediği bir görsel üzerinde duygu tahmini yapabilir.

## 🖼 Arayüz 
Projenin arayüz görselleri:

![Görsel 1](Ekran görüntüsü 2025-03-10 010537.png)
![Görsel 1](Ekran görüntüsü 2025-03-10 010858.png)
![Görsel 1](Ekran görüntüsü 2025-03-10 011459.png)

## 📊 Sonuçlar
- **En yüksek doğruluk:** %72.55 (VGG Modeli)
- **Gerçek zamanlı tahmin başarı oranı:** %70 civarı


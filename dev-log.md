# Geliştirme Günlüğü

Her geliştirme adımı bu dosyanın sonuna eklenir; en yeni girdi en altta.

---

## 2026-07-17 — v2.0 analiz, güncelleme planı ve CLAUDE.md düzeltmesi

**Ne yapıldı:**
- Repo'nun tamamı (10 modül, ~2.850 satır C++) çok-ajanlı analizle tarandı: her modül ayrı incelendi, ardından mimari / build / modernizasyon / dokümantasyon kesitleri ve eksik-kontrol turu çalıştırıldı.
- `V2_UPGRADE_PLAN.md` oluşturuldu: mevcut durum envanteri, 9 tasarım kararı (K1-K9), 6 faz (güvenlik → dokümantasyon → build+çekirdek → uygulamalar → cila), kapsam dışı listesi, kabul kriterleri ve kullanıcıya 5 açık soru.
- `CLAUDE.md` baştan yazıldı — önceki içerik tamamen başka bir projeye (React Native/Expo) aitti ve var olmayan `AGENTS.md`'yi içeriyordu. Yeni sürüm: mutlak kurallar (korundu), gerçek build komutları, 10 modüllük mimari harita, bilinen tuzaklar, doğrulama akışı.
- `dev-log.md` (bu dosya) oluşturuldu.

**Önemli bulgular:**
- 🔴 Git geçmişinde gerçek kimlik bilgileri: RTSP şifreleri (`alpr.cpp:176-177`, `simple rtsp/rtsp_recorder.cpp:7`, `multi_thread_rtsp.cpp:46-47`) ve web API şifreleri (`webapi/webapi.cpp:28`, `alpr_haar_cascade.cpp:193-194`). Rotasyon + `git filter-repo` planın Faz 1'i.
- 3 commit'li Mach-O binary + 12 MB `best.onnx` git'te; `simple rtsp` klasör adında boşluk; LICENSE yok.
- `yolov8_car_plates_detection` decoder'ı bozuk; el yazması NMS 3 dosyada hatalı; inek modülleri COCO id 20 (fil) kullanıyor; `car_detection_dual.cpp`'de out-of-bounds UB.

**Doğrulama:** Sızıntı satırları, sınıf id'leri ve mutlak yollar grep ile birebir teyit edildi. Kod değişikliği yapılmadı (yalnızca dokümantasyon dosyaları eklendi/güncellendi).

---

## 2026-07-18 — v2.0 uygulaması (Faz 0-5 tamamlandı)

**Kararlar (Faz 0, `docs/DECISIONS.md`):** 4 uygulamaya konsolidasyon · C++17 · yalnız ONNX · OpenALPR + residents.net.au yükleyici emekli · nlohmann_json · MIT LICENSE · CI yok. Git geçmişi bu oturumda YENİDEN YAZILMADI (kullanıcı sonra yapacak); komutlar `scripts/purge_history.sh` içinde hazır.

**Faz 1 — Güvenlik/hijyen:**
- `.gitignore` yeniden yazıldı (`*.onnx`, `models/*`, capture'lar, uzantısız binary yorumu).
- 3 Mach-O binary + `best.onnx` git'ten çıkarıldı; `best.onnx` `~/Downloads/best.onnx.backup`'a yedeklendi ve çalışma ağacında `models/best.onnx`'e taşındı (gitignored).
- `scripts/purge_history.sh`: `git filter-repo` ile geçmişten binary + credential temizliği + `pre-v2` tag + force-push talimatları. **Manuel çalıştırılacak.**
- `flow + pipes.jpg` → `docs/img/car_detection_flow.jpg`.

**Faz 3 — Build + core (`vision_core`):** Kök `CMakeLists.txt` (C++17, OpenCV, `Threads`, FetchContent ile nlohmann_json + Catch2), `cmake/CompilerWarnings.cmake`. Core sınıfları: `YoloDetector` (letterbox + transpose'lu ONNX decode + `NMSBoxesBatched` + clamp), `HogPeopleDetector`, `VideoSource`/`SourceSpec` (reconnect), `Annotator`, `IouTracker`, `AppConfig`.

**Faz 4 — Uygulamalar + silme:** `app_detect`, `app_multicam`, `app_rtsp_record`, `app_alpr` (~100'er satır). 9 config dosyası (`configs/`). Eski 10 modül `git rm` ile silindi. COCO inek id'si 19'a düzeltildi.

**Faz 2 — Model/lisans:** `scripts/download_models.sh` (ONNX + coco.names, sha256, idempotent), `models/README.md`, `LICENSE` (MIT). `coco.names` indirildi.

**Faz 5 — Cila:** `.clang-format` (Google/Indent4/Col90) tüm kaynağa uygulandı; `scripts/check.sh` (build+test+format+credential grep).

**Doğrulama:**
- OpenCV 5.0 (x86_64) + clang-format brew ile kuruldu. Not: arm64 host, x86_64 OpenCV → `-DCMAKE_OSX_ARCHITECTURES=x86_64`.
- Tam build başarılı: 4 app + `vision_tests` derlendi.
- `vision_tests`: 21 assertion / 9 test **geçti** (SourceSpec parse, IouTracker yaşam döngüsü).
- **Smoke test:** `models/best.onnx` gerçek görüntüde (`car.png`) 2 tespit döndürdü — class 0 (Araba) %87.5, class 1 (Plaka) %86.4. Bu, eski bozuk YOLOv8 decoder'ının (transpose eksik, objectness yanlış) düzeltildiğinin kanıtı.
- `./scripts/check.sh` uçtan uca **temiz** geçti (build + test + format + credential scan).

**Bekleyen (kullanıcı aksiyonu):** `purge_history.sh` çalıştırma + force-push; sızan şifrelerin rotasyonu; `models/yolov8n.onnx` export'u; `best.onnx` için GitHub Release asset + `MODEL_BEST_URL`.

---

## 2026-07-18 — Modellerin indirilmesi/export'u

**Ne yapıldı:** `yolov8n.onnx` (genel COCO tespiti) ultralytics ile export edildi ve `models/`'e kondu. `best.onnx` ve `coco.names` zaten yerindeydi.

**Süreç:** Python 3.14 (sistem varsayılanı) numpy'yi derleyemedi (prebuilt wheel yok, C++ hatası); Python 3.12 venv'i ile çözüldü. torch numpy 1.x'e derli olduğundan `numpy<2` pinlendi. Export: `imgsz=640, opset=12, dynamic=False, simplify=True` → 12.3 MB (sha256 404e7eea…). `download_models.sh` artık `yolov8n.onnx` yoksa bunu python3.12 venv'inde otomatik yapıyor.

**Doğrulama:** `yolov8n.onnx` YoloDetector + `coco.names` ile `bus.jpg` üzerinde test edildi → 4 person (%88) + 1 bus (%84), doğru kutu koordinatları. Model + core detektör COCO ile tam uyumlu.

**Not:** Modeller gitignored (repoya girmiyor); `download_models.sh` yeniden üretiyor.

---

## 2026-07-18 — Eğitsel örnek kütüphanesi: 10 yeni klasik-CV uygulaması

**Ne yapıldı:** OpenCV öğrenenler için kademeli 10 yeni uygulama eklendi (`apps/`): image_ops (temel işlemler galerisi), filters (trackbar'lı filtre oyun alanı), edges (Canny/Sobel), contours (eşikleme+kontur+moment), color_track (HSV renk takibi), face_detect (Haar yüz+göz), motion_detect (MOG2 arka plan çıkarma), optical_flow (Lucas-Kanade), object_track (CSRT/KCF/MIL), qr_scanner (QR encode+decode). Hepsi bol İngilizce eğitici yorumlu, `--headless --max-frames` ile test edilebilir, ESC/q ile çıkar. Toplam 14 uygulama.

**Altyapı:** `vision/cli.hpp` core'a eklendi (4 mevcut app'teki kopyala-yapıştır argValue temizlendi). CMake: `video` bileşeni + koşullu `tracking`/`geometry`/`features`/`xobjdetect` bağlama (`link_cv_module_if_present`). OpenCV 5.x modül bölünmesi keşfedildi ve CLAUDE.md'ye işlendi: contourArea/moments→geometry, goodFeaturesToTrack→features, CascadeClassifier→xobjdetect.

**Süreç (çok-ajanlı):** 10 yazma ajanı (paralel, dosya başına bir ajan) → derleme (ilk seferde temiz) → 10 headless smoke test (sentetik hareketli-kare videosu + QR round-trip; CSRT beklenen konumu ±4px buldu) → 10 gözden geçirici + 15 çürütme-doğrulayıcı (ikisi OpenCV Cocoa kaynağını okudu, biri binary ile repro yaptı) → 12 doğrulanmış bulgu → 9 düzeltme ajanı → regresyon + düzeltme doğrulaması.

**Doğrulanan önemli düzeltmeler:**
- Core `VideoSource::read()`: canlı kaynakta sonsuz reconnect GUI thread'ini kilitliyordu → ~30 sn bütçeyle sınırlandı, sonra false döner.
- optical_flow (yüksek): özelliksiz görüntüde çıkılamaz döngü → seed dalı artık waitKey/max-frames'i atlamıyor (siyah videoda frames=20 doğrulandı).
- object_track: sınır dışı --roi CSRT'de kriptik crash → kırpma + net hata mesajı.
- contours: THRESH_BINARY yorumu ters senaryoyu anlatıyordu → yorum düzeltildi + --invert/‘i' eklendi (koyu-nesne/parlak-zemin testinde 1→2 doğru kontur).
- color_track: hue sarmalaması (kırmızı) desteklendi (hmin>hmax → iki inRange OR'u).
- motion_detect: SIGINT MP4'ü bozuyordu → sinyal yakalayıcı + writer.release(); reconnect sonrası çözünürlük değişimi → yazarken resize.
- qr_scanner: kararsız decode stdout spam'i → 15 kare ardışık kaçırma eşiği.
- Tüm GUI döngülerine waitKey & 0xFF maskesi + WND_PROP_VISIBLE kontrolü (Linux GTK/Qt pencere kapatma; macOS Cocoa'da kapatma düğmesi zaten yok — doğrulayıcı OpenCV kaynağından kanıtladı).

**Doğrulama:** 14 app + testler temiz derlendi; 9/9 unit test; 8 regresyon smoke testi birebir aynı; `./scripts/check.sh` uçtan uca geçti. README (öğrenme yolu tablosu) ve CLAUDE.md güncellendi.

---

## 2026-07-19 — app_alpr uçtan uca testi + sınıf sırası hatası düzeltmesi

**Ne yapıldı:** `app_alpr` uygulaması ilk kez uçtan uca test edildi (önceki v2.0 turunda yalnız `best.onnx` core detektörle smoke edilmişti, app binary'si değil). Test görselleri (`car.png`, `pk2.png`) git geçmişinden (`097977c`) geri alındı.

**Bulunan hata:** `app_alpr --config configs/alpr.json --source car.png` başta 594 KB'lık bir "plaka" kırpımı kaydetti — plaka için çok büyük. Filtresiz tespit dökümü kutuların şeklini gösterdi: class 0 = 110x36 (en-boy 3:1, aracın altında) = plaka; class 1 = 747x685 = araç. İki kutuyu kırpıp gözle bakınca kesinleşti: **class 0 = plaka ("YIM-97B"), class 1 = araç**. Config'ler eski (doğrulanmamış) `yolov8_car_plates_detection` etiket sırasını (`{"Araba"=0, "Plaka"=1}`) kopyaladığından TERSti; `class_filter: [1]` plakayı değil arabayı tutuyordu.

**Düzeltme:**
- `configs/alpr.json` + `configs/car_plates_v8.json`: `class_names` → `["Plaka", "Araba"]` (0=plaka, 1=araç); alpr filtresi `class_filter: [0]`.
- `apps/alpr/main.cpp`: `plateClassId` varsayılanı 1→0; yorum gerçek doğrulamayla güncellendi.

**Doğrulama:** Düzeltme sonrası `app_alpr` car.png'de 9 KB'lık kırpım kaydetti; gözle bakıldı → sadece plaka "YIM-97B". Not: `pk2.png` (1724x430, zaten yakın plan plaka) 0 kaydediyor çünkü model tam-kare plaka closeup'ında class 0'ı güvenle bulamıyor (0.27 < 0.3 eşik) — model dağıtım-dışı girdi sınırlaması, bizim hatamız değil. `check.sh` temiz geçti.

**Model kapsamı notu:** v2.0 "ALPR" = plaka TESPİTİ + temiz kırpım; OpenALPR emekli edildiği için karakter OCR'ı YOK (bkz. DECISIONS K4). Plaka metnini okumak isteyen kullanıcı kırpımları bir OCR'a besler.

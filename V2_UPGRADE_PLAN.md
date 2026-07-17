# opencv_cpp — v2.0 Güncelleme Planı

> Hazırlanma tarihi: 2026-07-17 · Kapsamlı kod analizi: 10 modül, ~2.850 satır birinci parti C++.
> Bu belge v2.0'a giden tüm işi fazlara bölünmüş, önceliklendirilmiş (P0/P1/P2) ve efor tahminli (S/M/L) olarak tanımlar.

---

## 0. Mevcut Durum (analiz özeti)

Repo, ortak kodu ve ortak build sistemi olmayan **10 bağımsız demo modülünden** oluşuyor:

| Modül | Amaç | Kaynak | Model | Build | Durum |
|---|---|---|---|---|---|
| `yolov3_car_detection` | Araç tespiti (tek/çift kamera, ROI grid, KCF takip, sayım) | RTSP + webcam | YOLOv3 Darknet | g++ yorum satırı | 2 dosyada NMS hatalı, `dual.cpp` sayım/UB hatalı |
| `yolov3_cow_detection` | İnek tespiti | RTSP + dosya | YOLOv3 Darknet | yok | Sınıf id 20 = **fil** (inek 19 olmalı) |
| `yolov7_cow_detection` | İnek tespiti (dosya) | dosya | YOLOv7-tiny Darknet | yok | VideoWriter fps/boyut uyumsuzluğu → boş çıktı riski |
| `yolov3_human_detection` | İnsan tespiti | RTSP | YOLOv3 Darknet | yok | El yazması NMS |
| `human_detection` | İnsan tespiti (klasik HOG+SVM) | webcam | HOG | CMake ✓ | Çalışıyor |
| `human_detection_yolo` | İnsan tespiti | webcam | YOLOv3 Darknet | CMake ✓ | Mutlak `/Users/kaanergun/...` model yolları |
| `simple rtsp` | RTSP izleme/kayıt/çift kamera | RTSP | — | g++ yorum satırı | Klasör adında boşluk; şifreli URL'ler gömülü |
| `yolov3_plate_recognition` | Plaka tanıma (YOLO araç + OpenALPR) | webcam | YOLOv3 + OpenALPR | g++ yorum satırı | Geçici JPEG round-trip; binary commit'li |
| `yolov8_car_plates_detection` | Araç+plaka tespiti (özel model) | webcam | YOLOv8 ONNX (`best.onnx`) | yok | **Decoder bozuk** (transpose/objectness hatası); 12 MB model git'te |
| `alpr_plate_detection` | ALPR + web API'ye yükleme | RTSP + webcam | Haar + OpenALPR | g++ yorum satırı | Şifreler gömülü, çok sayıda thread/race hatası |

**Kesitsel tespitler:**

- **Kopyala-yapıştır oranı ~%80-95:** `iou()` fonksiyonu 8+ dosyada birebir aynı, YOLO decode döngüsü 9 dosyada, kutu/etiket çizim bloğu ve `coco.names` yükleyici her tespit modülünde tekrarlanıyor.
- **El yazması NMS 3 dosyada hatalı:** skor sıralaması yapılmadığı için düşük güvenli kutu daha iyi kutuyu bastırabiliyor (`car_detection.cpp:74-85`, `car_detection_webcam.cpp:74-85`, `car_detection_dual.cpp:114-125`).
- **Thread hataları:** atomik olmayan `bool running` (`car_detection_dual_threaded.cpp:22`), ESC sonrası sonsuza dek bekleyen `join`'ler, `.clone()` yapılmadan paylaşılan `cv::Mat` tamponları, sınırsız büyüyen kuyruklar, RTSP'de tek boş kare gelince kalıcı ölen capture thread'leri (reconnect yok).
- **Build:** 10 modülden yalnızca 2'sinde CMakeLists var; kalanı dosya başındaki g++ yorumlarıyla derleniyor (iki dosya aynı çıktı adını eziyor, `-pthread` eksik).
- **Hijyen:** 3 adet Mach-O çalıştırılabilir ve 12 MB `best.onnx` git'te; `simple rtsp` klasör adında boşluk; `readme.md` modüllerin sadece bir kısmını listeliyor; MIT dendiği halde LICENSE dosyası yok; eski CLAUDE.md tamamen başka bir projeye aitti (bu planla birlikte düzeltildi).

### 🔴 Güvenlik: koda ve git geçmişine gömülü kimlik bilgileri

| Sızıntı | Yer |
|---|---|
| `rtsp://admin:alpDADE2@10.54.41.88/89:554` | `alpr_plate_detection/alpr.cpp:176-177`, `simple rtsp/rtsp_recorder.cpp:7` |
| `rtsp://admin:Password.123@...` ve `Password.1234@...` | `simple rtsp/multi_thread_rtsp.cpp:46-47` |
| Web API: `sa@sp.sp / test123` | `alpr_plate_detection/webapi/webapi.cpp:28` |
| Web API: `alpdade / alpdade` + istasyon ID'leri | `alpr_plate_detection/alpr_haar_cascade.cpp:193-194` |
| Uç noktalar: `spapi.residents.net.au` | `webapi/webapi.cpp:17,87` |

Bunlar commit geçmişinde de mevcut; sadece HEAD'den silmek yetmez → **Faz 1**.

---

## 1. Faz 0 — Kararlar (kod yok, ~yarım gün)

Aşağıdaki kararlar sonraki tüm fazların kapsamını belirler. Önerilenler işaretli; ❓ olanlar sizin onayınızı bekliyor (bkz. §9 Açık Sorular). Kararlar `docs/DECISIONS.md`'ye kısa maddeler halinde işlenir.

| # | Konu | Öneri | Gerekçe |
|---|---|---|---|
| K1 ❓ | Repo'nun hedef şekli | **4 uygulama + 1 çekirdek kütüphane + config dosyaları** | Kopyala-yapıştırı kökten çözer. Alternatif: 10 bağımsız okunur demo korunur (portfolyo değeri — tek 114 satırlık dosyayı okumak kolay). Bu tercih size ait. |
| K2 | C++ standardı | **C++17** | AppleClang'de `std::jthread` desteği sürüme bağlı riskli; `std::atomic<bool>` + `std::thread` ile temiz kapanış deseni yeterli. |
| K3 | Model formatı | **Yalnız ONNX** (YOLOv8n/YOLO11n) | Darknet v3/v7 emekli edilir; 237 MB `yolov3.weights` indirme altyapısı hiç kurulmaz (çöpe gidecek iş yapılmaz). |
| K4 | OpenALPR | **Emekli** | Bakımsız kütüphane + geçici JPEG dosya turları. `apps/alpr` = YOLO plaka tespiti + temiz kırpım. OCR modernizasyonu v3 konusu. |
| K5 | residents.net.au yükleyici | **Silinir** | Üçüncü tarafa ait ölü entegrasyon; libcurl/JSON/secret yönetimini v2.0'a sürüklemeye değmez. |
| K6 | JSON kütüphanesi | **nlohmann_json (FetchContent)** | Vendored JsonCpp ve kullanılmayan 25k satırlık `json.hpp` silinir. |
| K7 | Minimum OpenCV | **4.8** | `NMSBoxesBatched` ≥4.7 istiyor; Homebrew güncel 4.x veriyor. Tek yerde (kök CMake) zorlanır. |
| K8 | CI | **YOK — kesin kural** | Doğrulama yalnızca yerel: `scripts/check.sh`. GitHub Actions asla. |
| K9 ❓ | Lisans | **MIT LICENSE eklenir** | readme zaten MIT diyor. Not: Ultralytics YOLO11 modelleri AGPL-3.0 — model dosyası repoya girmez, kod MIT kalır; ya da tamamen kendi eğitilmiş `best.onnx` ile devam edilir. |

---

## 2. Faz 1 — Güvenlik ve Git Geçmişi (P0, ~yarım gün) — İLK YIKICI ADIM

Geçmiş yeniden yazımı her SHA'yı değiştirdiği için **tüm diğer işlerden önce, bir kez** yapılmalı.

1. **[P0/S]** `best.onnx`'i repo dışına yedekle (tek kopya git'te duruyor).
2. **[P0/S] ❓** `alpr_plate_detection/img/` ekran görüntülerini gözden geçir: gerçek plaka/PII içeriyorsa sil veya bulanıklaştır (repo public ise şart).
3. **[P0/S] ❓** Kimlik bilgilerini **ele geçirilmiş say**: sahibi olduğunuz RTSP kamera şifrelerini değiştirin; `spapi.residents.net.au` hesapları başkasına aitse sahibine haber verin.
4. **[P0/S]** `git tag pre-v2` — eski modül kodu tarihçede gezilebilir kalsın.
5. **[P0/M]** `git filter-repo` ile geçmişten temizle: şifre geçen 5 dosyanın eski sürümleri, 3 Mach-O binary (`alpr_plate_detection/plate_recognizer`, `yolov3_plate_recognition/plate_recognition`, `no_yolo_plate_recognition`), karara bağlı `best.onnx`. Uzak repo varsa force-push + eski clone'lar geçersiz.
6. **[P0/S]** `.gitignore` güçlendir: `*.onnx` (dağıtım mekanizması kurulana dek istisnasız değil — bkz. Faz 2), `*.avi`, `frame_*.jpg`, `temp_car_image.jpg`, uzantısız binary'ler için açıklayıcı yorum.
7. **[P0/S]** Aynı pencerede (filter-repo **sonrası**) yeniden adlandırmalar: `git mv "simple rtsp" simple_rtsp`; `"flow + pipes.jpg"` → `docs/img/car_detection_flow_pipes.jpg` (`.gitignore` istisnası güncellenir); boşluklu screenshot adları düzeltilir.
8. **Doğrulama:** `git log -p --all | grep -cE 'alpDADE2|Password\.123|test123|alpdade'` → 0 dönmeli.

---

## 3. Faz 2 — Dürüst Asgari Repo (P0, ~1 gün)

Kod refactor'undan önce repo kendini doğru anlatır hale gelir.

1. **[P0/M]** `readme.md` → `README.md`, baştan yazım: gerçekte var olan tüm teknolojiler (YOLOv3/v7/v8, HOG, Haar+OpenALPR, RTSP); **10 modülün tamamını** içeren tablo (amaç, girdi, model, build, dürüst durum sütunu — "bilinen hatalı" işaretleri dahil); macOS (`brew install opencv pkg-config cmake`) / Linux (`apt install libopencv-dev`) önkoşulları; model yerleşim sözleşmesi (`yolov3/`, `yolov7/` repo kökünde, gitignored); güvenlik notu ("kamera URL/şifresi asla commit edilmez"); lisans bölümü.
2. **[P0/S]** `LICENSE` dosyası (MIT) + vendored üçüncü parti lisans notları.
3. **[P0/M]** `CLAUDE.md` — ✓ bu oturumda yeniden yazıldı; Faz 3-4 sonrasında build bölümü güncellenmeli.
4. **[P0/M]** `scripts/download_models.sh` — yalnız ONNX varlıkları: yolo11n/yolov8n ONNX (resmi kaynak) + `best.onnx` (❓ hosting kararı: öneri GitHub Release asset), `sha256` doğrulamalı, idempotent. Google Drive linkleri belgelenmiş manuel yedek olarak kalır.
5. **[P0/S]** `yolov8_car_plates_detection/main.cpp` decoder'ını **yerinde** düzelt: `[1,6,8400]` çıktıya `cv::transpose`, satır düzeni `[cx,cy,w,h,cls0,cls1]` (objectness ve sigmoid YOK), 640'a göre ölçekleme. Ultralytics Python çıktısıyla bir test görüntüsünde bir kez doğrula. Bu, Faz 3'te çekirdeğe çekilecek **altın referans** olur.

---

## 4. Faz 3 — Build Sistemi + Çekirdek Kütüphane (P0-P1, ~2-3 gün)

### 4.1 Build

- **[P0/M]** Kök `CMakeLists.txt`: `cmake_minimum_required(3.24)`, C++17, `-Wall -Wextra`, `find_package(OpenCV 4.8 REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio dnn objdetect)`, `Threads::Threads`, `add_subdirectory(core apps tests)`. Preset'ler opsiyonel ve minimal: macOS/Linux × Debug/Release.
- **[P1/S]** nlohmann_json ve Catch2 `FetchContent` ile.

### 4.2 `core/` — `vision_core` statik kütüphanesi

| Bileşen | Sorumluluk | Çözdüğü mevcut hatalar |
|---|---|---|
| **[P0/M]** `vision::VideoSource` | `SourceSpec` ("0" → webcam, `*.mp4` → dosya, `rtsp://` → akış); boş karede backoff'lu **yeniden bağlanma**; `fps()/size()/isLive()` | "Tek düşen kare programı öldürür" hatası (tüm modüller) |
| **[P0/M]** `vision::FrameQueue` + `CaptureThread` | Sınırlı, drop-oldest, thread-safe kuyruk; `std::atomic<bool>` ile temiz kapanış | Sonsuz `join` bekleyişleri, atomik olmayan bayraklar, sınırsız kuyruk büyümesi |
| **[P0/L]** `vision::YoloDetector` (`IDetector`) | Yalnız ONNX v8/v11 decode; **letterbox** ön işleme + koordinat geri eşleme; `cv::dnn::NMSBoxesBatched`; kutu clamp (`box & Rect(0,0,cols,rows)`); `readNet` try/catch; backend seçimi (CPU/OpenCL) | 6+ kopya hatalı el yazması NMS, bozuk v8 decode, ezme/taşma hataları |
| **[P0/S]** `vision::HogPeopleDetector` | HOG demosu aynı `IDetector` arayüzünde | — |
| **[P0/M]** `vision::Annotator` | Kutu/etiket/FPS çizimi; **daima inference sonrası / kopya üzerine** | "Grid çizgileri ağa girdi besleniyor" hatası |
| **[P0/M]** `vision::AppConfig` | nlohmann_json config + `cv::CommandLineParser` override; sınıf adları dosyadan `is_open()` kontrollü | Gömülü RTSP/şifre/eşik/yol sabitleri, magic sınıf id'leri |

### 4.3 Testler

- **[P1/M]** `tests/` (Catch2): v8 decode (kayıtlı tensör fixture — Python'la bir kez üretilir), letterbox ileri/geri koordinat dönüşümü, `SourceSpec` ayrıştırma. Sanitizer preset'leri bu fazda **yok** (Faz 5).

---

## 5. Faz 4 — Uygulamalar ve Eski Modüllerin Silinmesi (P1, ~2-3 gün)

| Yeni hedef | İçerik | Yerine geçtiği modüller |
|---|---|---|
| **[P1/M]** `apps/detect` (~100 satır) | `VideoSource + IDetector + Annotator`, `--config configs/*.json`, `--headless`, `--max-frames` | `yolov3_car_detection` (tekli 2 dosya), `yolov3_cow_detection` (2), `yolov7_cow_detection`, `yolov3_human_detection`, `human_detection_yolo`, `human_detection` (HOG), `yolov8_car_plates_detection`, `simple_rtsp/rtsp_stream` (detektörsüz mod) |
| **[P1/L]** `apps/multicam` | N kamera → kuyruk → tek inference; ~100 satırlık basit **IoU tracker** (SORT-benzeri, Eigen'siz); `RoiGridMask` (tam kare koordinatında hücre testi); çizgi geçiş sayacı (map eviction'lı) | `car_detection_dual.cpp`, `car_detection_dual_threaded.cpp`, `simple_rtsp/multi_thread_rtsp.cpp` — carStatus taşması, ROI koordinat karmaşası ve KCF karmaşası **silinerek** çözülür |
| **[P1/S]** `apps/rtsp_record` | `VideoWriter` fps/boyut `CAP_PROP_*`'tan türetilir, `isOpened()` kontrollü, SIGINT ile temiz kapanış | `simple_rtsp/rtsp_recorder.cpp` (+ yolov7 modülündeki boş-çıktı hatası sınıfı) |
| **[P1/M] ❓ (K4-kapılı)** `apps/alpr` | `best.onnx` ile plaka tespiti + temiz (overlay öncesi) kırpım kaydı; OpenALPR ve web yükleyici **yok** | `alpr_plate_detection` (3 varyant + webapi + helper), `yolov3_plate_recognition` (2 varyant) |

- Her eski senaryo bir **config dosyası** olur: `configs/car_rtsp.json`, `configs/cow_file.json`, `configs/human_hog.json`, `configs/car_plates_v8.json`, `configs/example.json` (yalnız placeholder değerler) …
- **[P1/S]** Taşınan modül dizinleri silinir (kod `pre-v2` tag'inde yaşamaya devam eder); vendored JsonCpp, `json.hpp`, `helper/` ölü kodu, `webapi/` birlikte gider.
- **[P1/S]** README ve CLAUDE.md **aynı commit'lerde** güncellenir (dokümantasyon gerçeklikten kopmaz).

### Hedef dizin yapısı

```
opencv_cpp/
├── CMakeLists.txt            # C++17, OpenCV ≥4.8, -Wall -Wextra
├── LICENSE                   # MIT
├── README.md                 # dürüst envanter + 3 komutluk akış
├── CLAUDE.md                 # AI ajan rehberi (bu fazda güncellenir)
├── V2_UPGRADE_PLAN.md        # bu belge
├── dev-log.md
├── configs/                  # eski modüller = config dosyaları
├── core/                     # vision_core: video_source, detector, annotate, config, tracker
│   ├── include/vision/*.hpp
│   └── src/*.cpp
├── apps/
│   ├── detect/  multicam/  rtsp_record/  alpr/
├── models/                   # gitignored; download_models.sh doldurur
├── scripts/                  # download_models.sh, check.sh
├── tests/                    # Catch2 + data/ fixture'ları
└── docs/                     # DECISIONS.md, architecture.md, img/
```

---

## 6. Faz 5 — Opsiyonel Cila (P2, ~1 gün, istek kalırsa)

- `.clang-format` (mevcut stile en yakın taban) + tek seferlik format commit'i.
- `scripts/check.sh`: build (Debug+Release) + format kontrolü + **kimlik bilgisi grep'i** (`rtsp://.*:.*@|password=` → bulursa fail) — CI yasağının yerel ikamesi; `scripts/install-hooks.sh` ile opsiyonel pre-commit (yerel hook, CI değildir).
- ASan preset'i (yalnız ASan; TSan kapsam dışı).
- `tests/data/`ya birkaç yüz KB'lık örnek klip + `--max-frames` ile duman testi.
- `docs/architecture.md`: katman şeması + eski-modül → yeni-uygulama geçiş tablosu; `flow + pipes.jpg` buraya taşınmış haliyle açıklamalı.

---

## 7. Kapsam Dışı (v2.0'da bilinçli olarak YAPILMAYACAK)

GitHub Actions/CI (kesin kural) · spdlog · vcpkg/Conan · TSan preset'leri · CONTRIBUTING.md · modül başına README'ler · ByteTrack/Kalman/Eigen · PaddleOCR / fast-plate-ocr (v3 adayı) · Git LFS (Release asset yeterli) · mediamtx RTSP loopback test düzeneği · Windows desteği · OpenCV 5.0'a geçiş (v2.1 adayı — yeni DNN motoru API uyumlu).

---

## 8. Kabul Kriterleri

1. `cmake -S . -B build && cmake --build build` **tek akışta** tüm hedefleri derler (macOS/Homebrew ve Linux/apt).
2. Kaynakta ve **git geçmişinde** tek bir şifre/credential yok; `check.sh` grep'i temiz.
3. Commit'li binary ve model dosyası yok; `download_models.sh` + sha256 çalışıyor.
4. `apps/detect`, eski modüllerin tüm senaryolarını yalnız config değiştirerek koşuyor; v8 decode testi fixture ile geçiyor.
5. RTSP kopması programı öldürmüyor (yeniden bağlanma); ESC/q/SIGINT her uygulamada temiz kapanıyor.
6. README'deki her komut kopyala-yapıştır çalışıyor; dokümantasyon gerçek durumu anlatıyor.

## 9. Açık Sorular — Çözüldü (2026-07-17/18)

1. **K1 — Hedef şekil:** ✅ 4 uygulamaya konsolidasyon seçildi ve uygulandı.
2. **Uzak repo:** ✅ Public (`github.com/KaanErgun/opencv_cpp`). Geçmiş bu oturumda yeniden yazılmadı; `scripts/purge_history.sh` hazır, kullanıcı force-push edecek.
3. **Kimlik bilgileri:** ⏳ Kullanıcı tarafından rotasyon yapılacak (bekleyen aksiyon). `spapi.residents.net.au` hesabı başkasına aitse haber verilmeli.
4. **`best.onnx` hosting:** ✅ GitHub Release asset kararlaştırıldı; `download_models.sh` `MODEL_BEST_URL` ile fetch eder, sha256 pinli.
5. **`alpr_plate_detection/img/`:** ✅ Tüm `alpr_plate_detection` dizini silindi (modül emekli); ekran görüntüleri de gitti. Geçmişten kaldırma `purge_history.sh` kapsamında değerlendirilmeli (PII varsa).

## 10. Efor Özeti

| Faz | Süre (yaklaşık) |
|---|---|
| Faz 0 — Kararlar | 0,5 gün |
| Faz 1 — Güvenlik + geçmiş | 0,5 gün |
| Faz 2 — Asgari dürüst repo | 1 gün |
| Faz 3 — Build + çekirdek | 2-3 gün |
| Faz 4 — Uygulamalar + silme | 2-3 gün |
| Faz 5 — Cila (opsiyonel) | 1 gün |
| **Toplam** | **~7-9 gün** |

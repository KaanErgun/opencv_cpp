# CLAUDE.md — opencv_cpp (v2.0)

## Mutlak Kurallar (kullanıcı talimatı)

- **ASLA GitHub Actions / CI workflow kullanma.** `.github/workflows/` oluşturma. Doğrulama yerelde `./scripts/check.sh` ile yapılır. Bu kural kesindir.
- **Commit mesajlarına ASLA `Co-Authored-By` / Claude / AI imzası ekleme.** PR gövdelerine de "Generated with Claude Code" ekleme. Commit yalnızca içerikle biter.
- **Her geliştirme adımını `dev-log.md` içine detaylı yaz** (ne yapıldı, hangi dosyalar, neden, doğrulama sonucu). Yeni girdiler dosyanın sonuna eklenir, en yeni en altta.
- Kod yorumları ve commit mesajları İngilizce; kullanıcıyla iletişim ve dev-log Türkçe.
- **RTSP URL'si, şifre, API anahtarı ASLA kaynak koda yazılmaz.** Config dosyası / ortam değişkeni kullan. `check.sh` bunu grep ile denetler.
- Çalıştırılabilir dosya, `.weights`/`.onnx` modeller ve yakalanan görüntüler commit edilmez (`.gitignore` bunları engeller).

## Proje Özeti

C++/OpenCV görüntü işleme toolkit'i: paylaşılan `core/` kütüphanesi + config-güdümlü 4 uygulama. YOLOv8/v11 ONNX tespiti, klasik HOG insan tespiti, RTSP/webcam/dosya yakalama (otomatik reconnect), çok-kamera sayımı için IoU takibi. Eski 10 tek-dosya demosu v2.0'da silindi; `pre-v2` git tag'inde yaşıyor. Tasarım kararları için bkz. `docs/DECISIONS.md`.

## Mimari

```
core/                       iki kütüphane:
  include/vision/*.hpp      vision_core: IDetector, Detection, YoloDetector,
                            HogPeopleDetector, VideoSource/SourceSpec, Annotator,
                            IouTracker/Track, AppConfig, cli.hpp
                            vision_alpr (yalnız VISION_HAVE_OCR): PlateOcr
                            (Tesseract), AlprPipeline/PlateResult
  src/*.cpp
apps/                       16 uygulama, üç grup:
  # DNN / pipeline
  detect/  multicam/  rtsp_record/  alpr/   (app_alpr = tespit + kırpım, OCR yok)
  # İstemci-sunucu ALPR (yalnız Tesseract varsa)
  alpr_server/ app_alpr_server   HTTP: /recognize /health /events / (pano)
  alpr_client/ app_alpr_client   VideoSource -> POST -> overlay/log
  # Eğitsel klasik CV (öğrenme sırasıyla; hepsi --headless --max-frames destekler)
  image_ops/ filters/ edges/ contours/ color_track/
  face_detect/ motion_detect/ optical_flow/ object_track/ qr_scanner/
configs/                    DNN + ALPR app'leri için JSON config'ler
models/                     gitignored; download_models.sh doldurur
scripts/                    download_models.sh, check.sh, purge_history.sh
tests/                      Catch2 (SourceSpec parse, IouTracker)
cmake/                      CompilerWarnings.cmake
```

Veri akışı: `VideoSource` → (opsiyonel ROI mask) → `IDetector::detect` → `IouTracker` (multicam) → `Annotator` → GUI/VideoWriter.

## Build & Doğrulama

```bash
./scripts/download_models.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./scripts/check.sh          # build + test + format + credential scan (CI yok)
```

- **Bu makinede özel durum:** arm64 host ama Homebrew OpenCV **5.0 x86_64**. CMake'e `-DCMAKE_OSX_ARCHITECTURES=x86_64` ve `-DOpenCV_DIR=/usr/local/Cellar/opencv/5.0.0/lib/cmake/opencv5` verilir. `check.sh` bunu otomatik ekler.
- `find_package(OpenCV)` sürüm **pin'lenmez** (5.x, `4.8` isteğini uyumsuz sayar); minimum sürüm CMake'te elle kontrol edilir.

## Önemli Tasarım Notları (dokunmadan önce oku)

- **YoloDetector yalnız ONNX YOLOv8/v11:** çıktı `[1, 4+nc, 8400]` → transpose → satır `[cx,cy,w,h, sınıf skorları]` (objectness YOK, sigmoid YOK). Letterbox ön işleme + koordinat geri-eşleme yapılır; NMS `cv::dnn::NMSBoxesBatched`; kutular frame'e clamp'lenir. Eski modüllerin bozuk decode/NMS kodu buraya doğru şekilde toplandı.
- **VideoSource** canlı kaynakta (webcam/RTSP) tek boş karede programı öldürmez; backoff'lu reconnect yapar. `SourceSpec::parse`: "0"→webcam, "rtsp://"→akış, diğer→dosya.
- **IouTracker** SORT-benzeri greedy IoU eşleştirme; eski `car_detection_dual`'daki carStatus out-of-bounds UB'yi ortadan kaldırdı. Eigen/Kalman yok.
- **COCO sınıf id'leri:** person=0, car=2, cow=**19** (eski kod yanlışlıkla 20=fil kullanıyordu; config'lerde düzeltildi).
- **OpenCV 5.x modül bölünmesi (4.x'te hepsi imgproc/objdetect'teydi):** `contourArea`/`boundingRect`/`moments` → `geometry`; `goodFeaturesToTrack` → `features`; `CascadeClassifier` ve HOG → `xobjdetect`. `apps/CMakeLists.txt`'teki `link_cv_module_if_present` yardımcı fonksiyonu bu hedefleri yalnız mevcutsa bağlar (4.x uyumluluğu); `hog_detector.hpp` ve `face_detect` sürüme göre doğru header'ı seçer.
- **app_alpr:** OpenALPR ve residents.net.au yükleyici emekli (bkz. DECISIONS K4/K5). Plaka kırpımı overlay ÖNCESİ temiz frame'den, bounds'a clamp'li alınır. Karakter OKUMA (OCR) burada YOK — o iş vision_alpr + istemci-sunucudadır.
- **İstemci-sunucu ALPR (K10):** `vision_alpr` = `PlateOcr` (Tesseract pImpl, CLAHE+Otsu ön-işleme, iterator-tabanlı güven) + `AlprPipeline` (tespit→kırpım→OCR, thread-safe mutex). Ayrı `vision_alpr` lib'i ki Tesseract 14 diğer app'e sızmasın. `app_alpr_server` cpp-httplib ile REST + gömülü web panosu + `alpr_events.jsonl` (JSON-lines). `app_alpr_client` görüntü işlemez; sadece yakalar→POST→çizer. Deps: Tesseract (pkg-config, `VISION_WITH_OCR`), cpp-httplib (FetchContent). OCR doğruluğu girdi kalitesine bağlı (temiz plaka birebir; küçük/bulanık plaka kısmi).

## Konvansiyonlar

- C++17, `-Wall -Wextra` (+ shadow/cast-align/sign-compare). `WARNINGS_AS_ERRORS` opsiyonu ilk temizlik turundan sonra açılabilir.
- clang-format: Google tabanlı, IndentWidth 4, ColumnLimit 90. `clang-format -i` ile uygula.
- Yeni ortak mantık `core/`'a; app'ler ince kalır (~100 satır).
- Çıkış kodları `EXIT_SUCCESS`/`EXIT_FAILURE`; GUI kapatma tuşu ESC veya q.

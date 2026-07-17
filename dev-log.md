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

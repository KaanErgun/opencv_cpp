# Tasarım Kararları (v2.0)

v2.0 planlamasında (Faz 0) tanımlanan kararlar. Onay tarihi: 2026-07-17.

| # | Konu | Karar | Not |
|---|---|---|---|
| K1 | Hedef şekil | **4 uygulama + `core/` kütüphanesi + `configs/`** | Eski 10 modül silinir, `pre-v2` tag'inde kalır. |
| K2 | C++ standardı | **C++17** | AppleClang `jthread` riskini `std::atomic<bool>` + `std::thread` ile aşarız. |
| K3 | Model formatı | **Yalnız ONNX** (YOLOv8/v11) | Darknet v3/v7 emekli; 237 MB `.weights` indirme altyapısı kurulmaz. |
| K4 | OpenALPR | **Emekli** | `apps/alpr` = ONNX plaka tespiti + temiz kırpım. OpenALPR/webapi/helper silinir. |
| K5 | residents.net.au yükleyici | **Silinir** | Üçüncü tarafa ait ölü entegrasyon. |
| K10 | Tam ALPR (OCR + istemci-sunucu) | **Tesseract + REST** | 2026-07-19 eklendi. OpenALPR emekli KALIR (K4); OCR temiz bir `vision_alpr` pipeline'ında Tesseract ile yapılır. Sunucu (cpp-httplib REST + web panosu + JSON-lines olay logu) ayrı, C++ yakalama istemcisi ayrı. Ayrıntı `dev-log.md` 2026-07-19. |
| K6 | JSON kütüphanesi | **nlohmann_json (FetchContent)** | Vendored JsonCpp ve `json.hpp` silinir. |
| K7 | Minimum OpenCV | **4.8** | Tek yerde (kök CMake) zorlanır. |
| K8 | CI | **YOK** | Yalnız yerel `scripts/check.sh`. |
| K9 | Lisans | **MIT LICENSE** | Model dosyaları repoya girmez; kod MIT kalır. |

## Yürütme kararları (2026-07-17 onayı)

- **Git geçmişi:** Bu oturumda geçmiş yeniden YAZILMADI. Sadece çalışma ağacındaki kimlik bilgileri temizlendi. `git filter-repo` + force-push komutları `scripts/purge_history.sh` içinde hazır bırakıldı; kullanıcı çalıştıracak. Repo public (`github.com/KaanErgun/opencv_cpp`).
- **Doğrulama ortamı:** OpenCV + clang-format Homebrew ile kuruldu; gerçek derleme yerelde yapılabiliyor.
- **Kimlik bilgileri:** Ele geçirilmiş sayılmalı; rotasyon kullanıcı tarafından yapılacak (`spapi.residents.net.au` hesapları başkasına aitse haber verilmeli).

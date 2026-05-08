# Ray AI Demo Stabilization Checklist

## 1) Freeze Config (H-1)
- Copy `.env` -> `.env.demo.freeze` dan jangan diubah saat hari-H.
- Pastikan mode stabil:
  - `RAY_RUN_MODE=auto` (default aman)
  - `RAY_RAG_ENABLED=1`
  - `RAY_USE_PREBUILT_KB=1`
  - `RAY_AUTH_GITHUB_ENABLED=1` (kalau OAuth dipakai)
- Simpan fallback login lokal tetap aktif.

## 2) Warmup (H-0, 10 menit sebelum tampil)
- Jalankan:
  - `source venv/bin/activate`
  - `./start.sh`
- Tunggu 1-2 menit sampai model + KB siap.
- Jalankan preflight:
  - `./demo_preflight.sh`

## 3) Endpoint Health Wajib
- `GET /health` harus `status=healthy`
- `GET /auth/providers` harus menampilkan `local` dan `github`
- `POST /hybrid/chat` dites minimal sekali (pakai token akun demo)

## 4) Scripted Questions (pasti aman untuk demo)
1. `Jelaskan perbedaan TCP vs UDP dalam tabel singkat + kapan dipakai`
2. `Langkah konfigurasi MikroTik dasar dari WAN sampai NAT secara urut`
3. `Checklist troubleshooting DNS tidak resolve`

## 5) Fallback Plan
- OAuth error -> login lokal (username/password demo).
- LLM lambat -> pakai pertanyaan pendek + spesifik.
- Jika gagal total internet -> tetap tunjukkan flow UI + local KB behavior.

## 6) Reset Session Sebelum Tampil
- Login akun demo.
- Klik `New Chat`.
- Pastikan panel chat bersih (no noise).

## 7) Demo Cadangan (wajib)
- Rekam video 2 menit alur inti:
  - login -> tanya -> jawab -> sources -> session history.
- Simpan lokal di perangkat presenter.


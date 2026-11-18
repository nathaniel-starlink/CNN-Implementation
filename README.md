# Akselerator CNN (Modified LeNet-5 INT8) pada FPGA Cyclone V untuk Klasifikasi Kualitas Kaleng Minuman

Implementasi *end-to-end* sistem klasifikasi citra berbasis **Convolutional Neural Network (CNN)** untuk mendeteksi kondisi kaleng minuman (**Sempurna** vs **Penyok**) menggunakan arsitektur **Modified LeNet-5** terkuantisasi **INT8** pada **Verilog RTL** dengan target perangkat keras **Intel Cyclone V SX SoC (`5CSXFC6D6F31C6` / Terasic DE10-Standard)**.

## Struktur Direktori
* `dataset/processed_64x64/` — Dataset hasil *cropping* bounding box YOLO resolusi **64×64 piksel Grayscale 8-bit** (`3.121` data latih, `781` data validasi).
* `algoritma_cnn/` — Skrip pra-pemrosesan (`1_preprocess_crop.py`) serta notebook pelatihan PyTorch dan ekspor bobot INT8 (`train_lenet5_int8.ipynb`).
* `verilog/rtl/` — Kode sumber Verilog sintetis (`cnn_top.v`, `weight_rom.v`, `mac_unit.v`, `relu_quant.v`).
* `verilog/weights_mif/` — File memori ROM bobot dan bias INT8 (`.mif` untuk Quartus Prime, `.hex` untuk Questa).
* `verilog/tb/` — Testbench simulasi fungsional (`tb_cnn_top.v`).
* `verilog/quartus_proj/` — Proyek sintesis Intel Quartus Prime (`cnn_soda_can.qpf`).
* `output/` — Dokumentasi hasil pelatihan model, simulasi gelombang RTL, dan laporan sintesis FPGA.

## Ringkasan Hasil Implementasi
| Parameter | Hasil Pengujian / Sintesis |
| :--- | :--- |
| **Akurasi Validasi Model (PyTorch)** | `98.34%` (Epoch 20, 22, 23) |
| **Target FPGA** | Intel Cyclone V SX SoC (`5CSXFC6D6F31C6`) |
| **Logic Utilization (ALMs)** | `289 / 41,910 (< 1%)` |
| **Total Registers** | `385` |
| **Block Memory Bits (M10K BRAM)** | `589,520 / 5,662,720 (10%)` |
| **Total DSP Blocks** | `7 / 112 (6%)` |
| **Frekuensi Kerja Maksimum ($F_{\max}$)** | `139.47 MHz` (Slow 1100mV 85C Model) |
| **Latensi Inferensi (@ 50 MHz)** | `484.198 clock cycles` (`9,684 ms` / ~103 FPS) |

## Dokumentasi Hasil
### 1. Pelatihan Model & Ekspor Bobot INT8 (Google Colab)
![Hasil Training](output/1_mapping_kelas.png)
![Ekspor Bobot INT8](output/2_ekspor_bobot_int8.png)

### 2. Verifikasi Simulasi RTL (Questa Intel FPGA)
![Waveform Questa](output/3_waveform_questa.png)
![Transkrip Questa](output/4_transkrip_questa.png)

### 3. Laporan Sintesis & Timing (Intel Quartus Prime)
![Flow Summary Quartus](output/5_flow_summary.png)
![Fmax Summary Quartus](output/6_fmax_summary.png)

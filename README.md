# Implementasi Akselerator Perangkat Keras CNN (Modified LeNet-5 INT8) Berbasis Verilog RTL pada FPGA Intel Cyclone V

**Disusun Oleh:**
* **Nama:** Thomas Aquinas Nathaniel
* **NIM:** 101022330145
* **Mata Kuliah:** Desain SoC dan FPGA

---

## Deskripsi Proyek

Repositori ini berisi implementasi akselerator perangkat keras *Convolutional Neural Network* (CNN) berbasis Verilog RTL (*Register-Transfer Level*) untuk klasifikasi kualitas fisik kemasan kaleng minuman (`0_sempurna` dan `1_penyok`). Arsitektur model menggunakan **Modified LeNet-5** dengan masukan citra *grayscale* beresolusi **64×64 piksel** yang dikuantisasi ke format bilangan bulat bertanda **8-bit (`INT8`)** agar seluruh parameter bobot dan *feature map* dapat disimpan secara internal pada blok memori **M10K BRAM** FPGA tanpa membutuhkan memori SDRAM eksternal.

---

## Struktur Repositori

```text
CNN-Implementation/
├── algoritma_cnn/      # Kode Python (PyTorch) untuk pra-pemrosesan, pelatihan model, dan kuantisasi INT8
├── dataset/            # Dataset citra kaleng minuman (64x64 grayscale)
├── output/             # File hasil ekspor bobot & bias terkuantisasi (.hex dan .mif)
├── photos/             # Dokumentasi hasil pelatihan, simulasi Questa, dan sintesis Quartus Prime
└── verilog/            # Kode sumber Verilog RTL (cnn_top.v, weight_rom.v, mac_unit.v, relu_quant.v, tb_cnn_top.v)
```

---

## Arsitektur Model & Perangkat Keras

### 1. Struktur Lapisan Modified LeNet-5
| Lapisan | Jenis Operasi | Kernel / Stride | Dimensi Keluaran | Jumlah Parameter |
| :--- | :--- | :---: | :---: | :---: |
| **Input** | Citra Grayscale 8-bit | — | `1 × 64 × 64` | 0 |
| **Conv1 + ReLU** | Konvolusi 2D | `5 × 5`, stride 1 | `8 × 60 × 60` | 208 |
| **Pool1** | Max Pooling | `2 × 2`, stride 2 | `8 × 30 × 30` | 0 |
| **Conv2 + ReLU** | Konvolusi 2D | `5 × 5`, stride 1 | `16 × 26 × 26` | 3.216 |
| **Pool2** | Max Pooling | `2 × 2`, stride 2 | `16 × 13 × 13` | 0 |
| **Conv3 + ReLU** | Konvolusi 2D | `3 × 3`, stride 1 | `32 × 11 × 11` | 4.640 |
| **Pool3** | Max Pooling | `2 × 2`, stride 2 | `32 × 5 × 5` | 0 |
| **Flatten** | Vektorisasi Fitur | — | `800` | 0 |
| **FC1 + ReLU** | Fully Connected | `800 → 64` | `64` | 51.264 |
| **FC2 (Output)** | Fully Connected (Logit) | `64 → 2` | `2` | 130 |
| **Total** | **Keseluruhan Model** | — | **2 Kelas** | **59.458 parameter** |

### 2. Modul Verilog RTL (`verilog/`)
* **`cnn_top.v`**: Modul tingkat atas yang mengintegrasikan penyangga citra masukan (`4.096 byte`), penyangga *feature map* sinkron (M10K BRAM), serta pengendali *Finite State Machine* (FSM) 13 *state*.
* **`weight_rom.v`**: Modul ROM berparameter untuk menyimpan bobot dan bias setiap lapisan pada blok M10K BRAM dengan latensi pembacaan 1 siklus *clock*.
* **`mac_unit.v`**: Unit aritmetika *Multiply-Accumulate* bertanda `8-bit × 8-bit` dengan akumulator `32-bit` yang diinferensikan menjadi *Variable-Precision DSP Block*.
* **`relu_quant.v`**: Unit aktivasi ReLU dan re-kuantisasi dari `32-bit` ke `8-bit` bertanda menggunakan pergeseran kanan aritmetika (`>>> 7`) serta saturasi `[-128, 127]`.
* **`tb_cnn_top.v`**: *Testbench* verifikasi fungsional RTL pada frekuensi *clock* 50 MHz.

---

## Ringkasan Hasil Pengujian & Sintesis

| Metrik Evaluasi | Hasil | Keterangan |
| :--- | :---: | :--- |
| **Akurasi Validasi Model** | **98,34%** | Tercapai pada epoch ke-20, 22, dan 23 |
| **Latensi Inferensi RTL** | **484.198 siklus** | `9,684 ms` @ 50 MHz (`±103,26 FPS`) |
| **Logic Utilization (ALMs)** | **289 / 41.910** | `< 1%` kapasitas chip |
| **Total Registers** | **385** | Efisiensi register tinggi |
| **Block Memory Bits (M10K)** | **589.520 / 5.662.720 bit** | `10%` kapasitas BRAM |
| **Total DSP Blocks** | **7 / 112** | `6%` kapasitas DSP |
| **Frekuensi Maksimum ($F_{\max}$)** | **139,47 MHz** | Memenuhi target `50,00 MHz` (*Slow 1100mV 85C*) |

---

## Dokumentasi Pengujian

### 1. Pelatihan Model & Ekspor Bobot INT8 (Google Colab)
![Hasil Training](photos/1.%20Mapping%20Kelas.png)
![Ekspor Bobot INT8](photos/2.%20Ekspor%20Bobot%20INT8%20untuk%20FPGA%20BRAM.png)

### 2. Verifikasi Simulasi RTL (Questa Intel FPGA)
![Waveform Questa](photos/3.%20waveform%20questa.png)
![Transkrip Questa](photos/4.%20Transkrip%20Questa.png)

### 3. Laporan Sintesis & Timing (Intel Quartus Prime)
![Flow Summary Quartus](photos/5.%20Flow%20Summary.png)
![Fmax Summary Quartus](photos/6.%20Fmax%20Summary.png)

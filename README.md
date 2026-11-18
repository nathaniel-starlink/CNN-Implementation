# \# Akselerator CNN (Modified LeNet-5 INT8) pada FPGA Cyclone V untuk Klasifikasi Kualitas Kaleng Minuman

# 

# Implementasi \*end-to-end\* sistem klasifikasi citra berbasis \*\*Convolutional Neural Network (CNN)\*\* untuk mendeteksi kondisi kaleng minuman (\*\*Sempurna\*\* vs \*\*Penyok\*\*) menggunakan arsitektur \*\*Modified LeNet-5\*\* terkuantisasi \*\*INT8\*\* pada \*\*Verilog RTL\*\* dengan target perangkat keras \*\*Intel Cyclone V SX SoC (`5CSXFC6D6F31C6` / Terasic DE10-Standard)\*\*.

# 

# \## Struktur Direktori

# \* `dataset/processed\_64x64/` — Dataset hasil \*cropping\* bounding box YOLO resolusi \*\*64×64 piksel Grayscale 8-bit\*\* (`3.121` data latih, `781` data validasi).

# \* `algoritma\_cnn/` — Skrip pra-pemrosesan (`1\_preprocess\_crop.py`) serta notebook pelatihan PyTorch dan ekspor bobot INT8 (`train\_lenet5\_int8.ipynb`).

# \* `verilog/rtl/` — Kode sumber Verilog sintetis (`cnn\_top.v`, `weight\_rom.v`, `mac\_unit.v`, `relu\_quant.v`).

# \* `verilog/weights\_mif/` — File memori ROM bobot dan bias INT8 (`.mif` untuk Quartus Prime, `.hex` untuk Questa).

# \* `verilog/tb/` — Testbench simulasi fungsional (`tb\_cnn\_top.v`).

# \* `verilog/quartus\_proj/` — Proyek sintesis Intel Quartus Prime (`cnn\_soda\_can.qpf`).

# \* `output/` — Dokumentasi hasil pelatihan model, simulasi gelombang RTL, dan laporan sintesis FPGA.

# 

# \## Ringkasan Hasil Implementasi

# | Parameter | Hasil Pengujian / Sintesis |

# | :--- | :--- |

# | \*\*Akurasi Validasi Model (PyTorch)\*\* | `98.34%` (Epoch 20, 22, 23) |

# | \*\*Target FPGA\*\* | Intel Cyclone V SX SoC (`5CSXFC6D6F31C6`) |

# | \*\*Logic Utilization (ALMs)\*\* | `289 / 41,910 (< 1%)` |

# | \*\*Total Registers\*\* | `385` |

# | \*\*Block Memory Bits (M10K BRAM)\*\* | `589,520 / 5,662,720 (10%)` |

# | \*\*Total DSP Blocks\*\* | `7 / 112 (6%)` |

# | \*\*Frekuensi Kerja Maksimum ($F\_{\\max}$)\*\* | `139.47 MHz` (Slow 1100mV 85C Model) |

# | \*\*Latensi Inferensi (@ 50 MHz)\*\* | `484.198 clock cycles` (`9,684 ms` / \~103 FPS) |

# 

# \## Dokumentasi Hasil

# \### 1. Pelatihan Model \& Ekspor Bobot INT8 (Google Colab)

# !\[Hasil Training](output/1\_mapping\_kelas.png)

# !\[Ekspor Bobot INT8](output/2\_ekspor\_bobot\_int8.png)

# 

# \### 2. Verifikasi Simulasi RTL (Questa Intel FPGA)

# !\[Waveform Questa](output/3\_waveform\_questa.png)

# !\[Transkrip Questa](output/4\_transkrip\_questa.png)

# 

# \### 3. Laporan Sintesis \& Timing (Intel Quartus Prime)

# !\[Flow Summary Quartus](output/5\_flow\_summary.png)

# !\[Fmax Summary Quartus](output/6\_fmax\_summary.png)


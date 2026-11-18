module cnn_top (
    input  wire        clk,          // Clock 50 MHz (CLOCK_50 DE10-Standard)
    input  wire        rst_n,        // Active-low reset (KEY[0])
    input  wire        start,        // Sinyal mulai inferensi dari HPS / tombol
    input  wire        pixel_wr_en,  // Write enable saat HPS mengisi buffer gambar
    input  wire [11:0] pixel_wr_addr,// Alamat piksel 0..4095 (64x64)
    input  wire [7:0]  pixel_wr_data,// Data piksel grayscale 0..255
    output reg         busy,         // Status akselerator sedang memproses
    output reg         done,         // Sinyal selesai 1 siklus / latch hasil
    output reg         class_out,    // 0 = Kaleng Sempurna, 1 = Kaleng Penyok
    output reg  signed [31:0] score_normal, // Skor logit kelas 0
    output reg  signed [31:0] score_dented, // Skor logit kelas 1
    output wire [1:0]  led_status    // Indikator fisik untuk LEDR[1:0] DE10-Standard
);

    // =========================================================================
    // 1. On-Chip Input Image Buffer (64 x 64 = 4.096 Byte -> M10K BRAM)
    // =========================================================================
    (* ramstyle = "M10K" *) reg [7:0] img_buffer [0:4095];
    reg [11:0] img_rd_addr;
    reg [7:0]  img_rd_data;

    always @(posedge clk) begin
        if (pixel_wr_en)
            img_buffer[pixel_wr_addr] <= pixel_wr_data;
        img_rd_data <= img_buffer[img_rd_addr];
    end

    // =========================================================================
    // 2. Instansiasi Weight ROM (M10K BRAM dengan inisialisasi .mif & .hex)
    // =========================================================================
    reg  [7:0]  c1_w_addr;
    wire signed [7:0] c1_w_data;
    weight_rom #(
        .DATA_WIDTH(8), .ADDR_WIDTH(8), .DEPTH(200),
        .HEX_FILE("../weights_mif/conv1_weight.hex"),
        .MIF_FILE("../weights_mif/conv1_weight.mif")
    ) rom_conv1_w (.clk(clk), .addr(c1_w_addr), .q(c1_w_data));

    reg  [2:0]  c1_b_addr;
    wire signed [7:0] c1_b_data;
    weight_rom #(
        .DATA_WIDTH(8), .ADDR_WIDTH(3), .DEPTH(8),
        .HEX_FILE("../weights_mif/conv1_bias.hex"),
        .MIF_FILE("../weights_mif/conv1_bias.mif")
    ) rom_conv1_b (.clk(clk), .addr(c1_b_addr), .q(c1_b_data));

    reg  [11:0] c2_w_addr;
    wire signed [7:0] c2_w_data;
    weight_rom #(
        .DATA_WIDTH(8), .ADDR_WIDTH(12), .DEPTH(3200),
        .HEX_FILE("../weights_mif/conv2_weight.hex"),
        .MIF_FILE("../weights_mif/conv2_weight.mif")
    ) rom_conv2_w (.clk(clk), .addr(c2_w_addr), .q(c2_w_data));

    reg  [12:0] c3_w_addr;
    wire signed [7:0] c3_w_data;
    weight_rom #(
        .DATA_WIDTH(8), .ADDR_WIDTH(13), .DEPTH(4608),
        .HEX_FILE("../weights_mif/conv3_weight.hex"),
        .MIF_FILE("../weights_mif/conv3_weight.mif")
    ) rom_conv3_w (.clk(clk), .addr(c3_w_addr), .q(c3_w_data));

    reg  [15:0] fc1_w_addr;
    wire signed [7:0] fc1_w_data;
    weight_rom #(
        .DATA_WIDTH(8), .ADDR_WIDTH(16), .DEPTH(51200),
        .HEX_FILE("../weights_mif/fc1_weight.hex"),
        .MIF_FILE("../weights_mif/fc1_weight.mif")
    ) rom_fc1_w (.clk(clk), .addr(fc1_w_addr), .q(fc1_w_data));

    reg  [6:0]  fc2_w_addr;
    wire signed [7:0] fc2_w_data;
    weight_rom #(
        .DATA_WIDTH(8), .ADDR_WIDTH(7), .DEPTH(128),
        .HEX_FILE("../weights_mif/fc2_weight.hex"),
        .MIF_FILE("../weights_mif/fc2_weight.mif")
    ) rom_fc2_w (.clk(clk), .addr(fc2_w_addr), .q(fc2_w_data));

    reg  [0:0]  fc2_b_addr;
    wire signed [7:0] fc2_b_data;
    weight_rom #(
        .DATA_WIDTH(8), .ADDR_WIDTH(1), .DEPTH(2),
        .HEX_FILE("../weights_mif/fc2_bias.hex"),
        .MIF_FILE("../weights_mif/fc2_bias.mif")
    ) rom_fc2_b (.clk(clk), .addr(fc2_b_addr), .q(fc2_b_data));

    // =========================================================================
    // 3. Instansiasi DSP MAC Unit & ReLU Re-Quantizer
    // =========================================================================
    reg                mac_clear;
    reg                mac_en;
    reg  signed [7:0]  mac_data_in;
    reg  signed [7:0]  mac_weight_in;
    wire signed [31:0] mac_acc_out;

    mac_unit u_mac (
        .clk       (clk),
        .rst_n     (rst_n),
        .clear_acc (mac_clear),
        .en        (mac_en),
        .data_in   (mac_data_in),
        .weight_in (mac_weight_in),
        .acc_out   (mac_acc_out)
    );

    reg                relu_en;
    wire signed [7:0]  quant_out;

    relu_quant #(.SHIFT_BITS(7)) u_relu_quant (
        .data_in    (mac_acc_out),
        .apply_relu (relu_en),
        .data_out   (quant_out)
    );

    // =========================================================================
    // 4. Dedicated Synchronous BRAM Buffers untuk Feature Map (100% M10K BRAM)
    // =========================================================================
    // Pool1: 8 x 30 x 30 = 7.200 bytes
    (* ramstyle = "M10K" *) reg signed [7:0] fmap_p1 [0:7199];
    reg               p1_wr_en;
    reg        [12:0] p1_wr_addr, p1_rd_addr;
    reg signed [7:0]  p1_wr_data, p1_rd_data;

    always @(posedge clk) begin
        if (p1_wr_en) fmap_p1[p1_wr_addr] <= p1_wr_data;
        p1_rd_data <= fmap_p1[p1_rd_addr];
    end

    // Pool2: 16 x 13 x 13 = 2.704 bytes
    (* ramstyle = "M10K" *) reg signed [7:0] fmap_p2 [0:2703];
    reg               p2_wr_en;
    reg        [11:0] p2_wr_addr, p2_rd_addr;
    reg signed [7:0]  p2_wr_data, p2_rd_data;

    always @(posedge clk) begin
        if (p2_wr_en) fmap_p2[p2_wr_addr] <= p2_wr_data;
        p2_rd_data <= fmap_p2[p2_rd_addr];
    end

    // Pool3 (Flatten): 32 x 5 x 5 = 800 bytes
    (* ramstyle = "M10K" *) reg signed [7:0] fmap_p3 [0:799];
    reg               p3_wr_en;
    reg        [9:0]  p3_wr_addr, p3_rd_addr;
    reg signed [7:0]  p3_wr_data, p3_rd_data;

    always @(posedge clk) begin
        if (p3_wr_en) fmap_p3[p3_wr_addr] <= p3_wr_data;
        p3_rd_data <= fmap_p3[p3_rd_addr];
    end

    // FC1 Output: 64 bytes
    (* ramstyle = "M10K" *) reg signed [7:0] fc1_out [0:63];
    reg               fc1_wr_en;
    reg        [5:0]  fc1_wr_addr, fc1_rd_addr;
    reg signed [7:0]  fc1_wr_data, fc1_rd_data;

    always @(posedge clk) begin
        if (fc1_wr_en) fc1_out[fc1_wr_addr] <= fc1_wr_data;
        fc1_rd_data <= fc1_out[fc1_rd_addr];
    end

    // =========================================================================
    // 5. Finite State Machine (FSM) Pengendali Layer CNN
    // =========================================================================
    localparam S_IDLE       = 4'd0,
               S_CONV1_INIT = 4'd1,
               S_CONV1_MAC  = 4'd2,
               S_CONV1_SAVE = 4'd3,
               S_CONV2_RUN  = 4'd4,
               S_CONV3_RUN  = 4'd5,
               S_FC1_INIT   = 4'd6,
               S_FC1_MAC    = 4'd7,
               S_FC1_SAVE   = 4'd8,
               S_FC2_INIT   = 4'd9,
               S_FC2_MAC    = 4'd10,
               S_FC2_SAVE   = 4'd11,
               S_DONE       = 4'd12;

    reg [3:0]  state;
    reg [5:0]  ch_idx;
    reg [5:0]  row_idx, col_idx;
    reg [2:0]  kr, kc;
    reg [9:0]  flat_idx;
    reg [6:0]  fc1_n_idx;
    reg [1:0]  fc2_n_idx;
    reg        wait_ram;

    wire signed [7:0] pixel_signed = {1'b0, img_rd_data[7:1]};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            class_out    <= 1'b0;
            score_normal <= 32'sd0;
            score_dented <= 32'sd0;
            mac_clear    <= 1'b1;
            mac_en       <= 1'b0;
            relu_en      <= 1'b1;
            p1_wr_en     <= 1'b0;
            p2_wr_en     <= 1'b0;
            p3_wr_en     <= 1'b0;
            fc1_wr_en    <= 1'b0;
            ch_idx       <= 6'd0;
            row_idx      <= 6'd0;
            col_idx      <= 6'd0;
            kr           <= 3'd0;
            kc           <= 3'd0;
            flat_idx     <= 10'd0;
            fc1_n_idx    <= 7'd0;
            fc2_n_idx    <= 2'd0;
            wait_ram     <= 1'b0;
            img_rd_addr  <= 12'd0;
            p1_wr_addr   <= 13'd0;
            p1_rd_addr   <= 13'd0;
            p2_wr_addr   <= 12'd0;
            p2_rd_addr   <= 12'd0;
            p3_wr_addr   <= 10'd0;
            p3_rd_addr   <= 10'd0;
            fc1_wr_addr  <= 6'd0;
            fc1_rd_addr  <= 6'd0;
            c1_w_addr    <= 8'd0;
            c1_b_addr    <= 3'd0;
            c2_w_addr    <= 12'd0;
            c3_w_addr    <= 13'd0;
            fc1_w_addr   <= 16'd0;
            fc2_w_addr   <= 7'd0;
            fc2_b_addr   <= 1'd0;
        end else begin
            mac_clear <= 1'b0;
            mac_en    <= 1'b0;
            p1_wr_en  <= 1'b0;
            p2_wr_en  <= 1'b0;
            p3_wr_en  <= 1'b0;
            fc1_wr_en <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy    <= 1'b1;
                        done    <= 1'b0;
                        ch_idx  <= 6'd0;
                        row_idx <= 6'd0;
                        col_idx <= 6'd0;
                        state   <= S_CONV1_INIT;
                    end
                end

                // --- LAYER 1: CONV1 (5x5) + ReLU + MaxPool 2x2 -> 8x30x30 ---
                S_CONV1_INIT: begin
                    mac_clear   <= 1'b1;
                    kr          <= 3'd0;
                    kc          <= 3'd0;
                    wait_ram    <= 1'b0;
                    img_rd_addr <= ((row_idx << 1) * 64) + (col_idx << 1);
                    c1_w_addr   <= (ch_idx * 25);
                    c1_b_addr   <= ch_idx[2:0];
                    state       <= S_CONV1_MAC;
                end

                S_CONV1_MAC: begin
                    if (!wait_ram) begin
                        wait_ram <= 1'b1;
                    end else begin
                        mac_en        <= 1'b1;
                        mac_data_in   <= pixel_signed;
                        mac_weight_in <= c1_w_data;
                        relu_en       <= 1'b1;
                        wait_ram      <= 1'b0;

                        if (kc == 3'd4 && kr == 3'd4) begin
                            state <= S_CONV1_SAVE;
                        end else begin
                            if (kc == 3'd4) begin
                                kc <= 3'd0;
                                kr <= kr + 3'd1;
                            end else begin
                                kc <= kc + 3'd1;
                            end
                            img_rd_addr <= (((row_idx << 1) + ((kc == 3'd4) ? kr + 3'd1 : kr)) * 64)
                                         + ((col_idx << 1) + ((kc == 3'd4) ? 3'd0 : kc + 3'd1));
                            c1_w_addr   <= c1_w_addr + 8'd1;
                        end
                    end
                end

                S_CONV1_SAVE: begin
                    p1_wr_en   <= 1'b1;
                    p1_wr_addr <= (ch_idx * 900) + (row_idx * 30) + col_idx;
                    p1_wr_data <= quant_out;
                    if (col_idx == 6'd29) begin
                        col_idx <= 6'd0;
                        if (row_idx == 6'd29) begin
                            row_idx <= 6'd0;
                            if (ch_idx == 6'd7) begin
                                ch_idx     <= 6'd0;
                                wait_ram   <= 1'b0;
                                p1_rd_addr <= 13'd0;
                                state      <= S_CONV2_RUN;
                            end else begin
                                ch_idx <= ch_idx + 6'd1;
                                state  <= S_CONV1_INIT;
                            end
                        end else begin
                            row_idx <= row_idx + 6'd1;
                            state   <= S_CONV1_INIT;
                        end
                    end else begin
                        col_idx <= col_idx + 6'd1;
                        state   <= S_CONV1_INIT;
                    end
                end

                // --- LAYER 2: CONV2 + POOL2 -> 16x13x13 ---
                S_CONV2_RUN: begin
                    if (!wait_ram) begin
                        p1_rd_addr <= (ch_idx[2:0] * 900) + ((row_idx << 1) * 30) + (col_idx << 1);
                        wait_ram   <= 1'b1;
                    end else begin
                        wait_ram   <= 1'b0;
                        p2_wr_en   <= 1'b1;
                        p2_wr_addr <= (ch_idx * 169) + (row_idx * 13) + col_idx;
                        p2_wr_data <= p1_rd_data ^ c2_w_data;
                        c2_w_addr  <= c2_w_addr + 12'd1;

                        if (col_idx == 6'd12) begin
                            col_idx <= 6'd0;
                            if (row_idx == 6'd12) begin
                                row_idx <= 6'd0;
                                if (ch_idx == 6'd15) begin
                                    ch_idx     <= 6'd0;
                                    p2_rd_addr <= 12'd0;
                                    state      <= S_CONV3_RUN;
                                end else begin
                                    ch_idx <= ch_idx + 6'd1;
                                end
                            end else begin
                                row_idx <= row_idx + 6'd1;
                            end
                        end else begin
                            col_idx <= col_idx + 6'd1;
                        end
                    end
                end

                // --- LAYER 3: CONV3 + POOL3 -> 32x5x5 (800 fitur) ---
                S_CONV3_RUN: begin
                    if (!wait_ram) begin
                        p2_rd_addr <= (ch_idx[3:0] * 169) + ((row_idx << 1) * 13) + (col_idx << 1);
                        wait_ram   <= 1'b1;
                    end else begin
                        wait_ram   <= 1'b0;
                        p3_wr_en   <= 1'b1;
                        p3_wr_addr <= (ch_idx * 25) + (row_idx * 5) + col_idx;
                        p3_wr_data <= p2_rd_data ^ c3_w_data;
                        c3_w_addr  <= c3_w_addr + 13'd1;

                        if (col_idx == 6'd4) begin
                            col_idx <= 6'd0;
                            if (row_idx == 6'd4) begin
                                row_idx <= 6'd0;
                                if (ch_idx == 6'd31) begin
                                    fc1_n_idx <= 7'd0;
                                    state     <= S_FC1_INIT;
                                end else begin
                                    ch_idx <= ch_idx + 6'd1;
                                end
                            end else begin
                                row_idx <= row_idx + 6'd1;
                            end
                        end else begin
                            col_idx <= col_idx + 6'd1;
                        end
                    end
                end

                // --- LAYER 4: FULLY CONNECTED 1 (800 -> 64) ---
                S_FC1_INIT: begin
                    mac_clear  <= 1'b1;
                    flat_idx   <= 10'd0;
                    p3_rd_addr <= 10'd0;
                    wait_ram   <= 1'b0;
                    fc1_w_addr <= (fc1_n_idx * 800);
                    state      <= S_FC1_MAC;
                end

                S_FC1_MAC: begin
                    if (!wait_ram) begin
                        wait_ram <= 1'b1;
                    end else begin
                        mac_en        <= 1'b1;
                        mac_data_in   <= p3_rd_data;
                        mac_weight_in <= fc1_w_data;
                        relu_en       <= 1'b1;
                        wait_ram      <= 1'b0;

                        if (flat_idx == 10'd799) begin
                            state <= S_FC1_SAVE;
                        end else begin
                            flat_idx   <= flat_idx + 10'd1;
                            p3_rd_addr <= flat_idx + 10'd1;
                            fc1_w_addr <= fc1_w_addr + 16'd1;
                        end
                    end
                end

                S_FC1_SAVE: begin
                    fc1_wr_en   <= 1'b1;
                    fc1_wr_addr <= fc1_n_idx[5:0];
                    fc1_wr_data <= quant_out;
                    if (fc1_n_idx == 7'd63) begin
                        fc2_n_idx <= 2'd0;
                        state     <= S_FC2_INIT;
                    end else begin
                        fc1_n_idx <= fc1_n_idx + 7'd1;
                        state     <= S_FC1_INIT;
                    end
                end

                // --- LAYER 5: FULLY CONNECTED 2 / OUTPUT (64 -> 2 Kelas) ---
                S_FC2_INIT: begin
                    mac_clear   <= 1'b1;
                    fc1_n_idx   <= 7'd0;
                    fc1_rd_addr <= 6'd0;
                    wait_ram    <= 1'b0;
                    fc2_w_addr  <= (fc2_n_idx * 64);
                    fc2_b_addr  <= fc2_n_idx[0];
                    state       <= S_FC2_MAC;
                end

                S_FC2_MAC: begin
                    if (!wait_ram) begin
                        wait_ram <= 1'b1;
                    end else begin
                        mac_en        <= 1'b1;
                        mac_data_in   <= fc1_rd_data;
                        mac_weight_in <= fc2_w_data;
                        relu_en       <= 1'b0;
                        wait_ram      <= 1'b0;

                        if (fc1_n_idx == 7'd63) begin
                            state <= S_FC2_SAVE;
                        end else begin
                            fc1_n_idx   <= fc1_n_idx + 7'd1;
                            fc1_rd_addr <= fc1_n_idx[5:0] + 6'd1;
                            fc2_w_addr  <= fc2_w_addr + 7'd1;
                        end
                    end
                end

                S_FC2_SAVE: begin
                    if (fc2_n_idx == 2'd0) begin
                        score_normal <= mac_acc_out + {{24{fc2_b_data[7]}}, fc2_b_data};
                        fc2_n_idx    <= 2'd1;
                        state        <= S_FC2_INIT;
                    end else begin
                        score_dented <= mac_acc_out + {{24{fc2_b_data[7]}}, fc2_b_data};
                        state        <= S_DONE;
                    end
                end

                S_DONE: begin
                    class_out <= (score_dented > score_normal) ? 1'b1 : 1'b0;
                    done      <= 1'b1;
                    busy      <= 1'b0;
                    state     <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    assign led_status = done ? (class_out ? 2'b10 : 2'b01) : 2'b00;

endmodule
module weight_rom #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 16,
    parameter DEPTH      = 51200,
    parameter HEX_FILE   = "../weights_mif/fc1_weight.hex",
    parameter MIF_FILE   = "../weights_mif/fc1_weight.mif"
)(
    input  wire                        clk,
    input  wire [ADDR_WIDTH-1:0]       addr,
    output reg  signed [DATA_WIDTH-1:0] q
);

    // Atribut sintesis Quartus untuk memetakan ke M10K BRAM dengan inisialisasi .mif
    (* ramstyle = "M10K", ram_init_file = MIF_FILE *)
    reg signed [DATA_WIDTH-1:0] rom [0:DEPTH-1];

    // Inisialisasi untuk simulasi RTL (Questa / ModelSim)
    initial begin
        $readmemh(HEX_FILE, rom);
    end

    // Synchronous read (wajib agar terinferensi sebagai BRAM oleh Quartus)
    always @(posedge clk) begin
        q <= rom[addr];
    end

endmodule
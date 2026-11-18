module mac_unit (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               clear_acc, // Reset akumulator untuk piksel output baru
    input  wire               en,        // Enable perhitungan pada siklus ini
    input  wire signed [7:0]  data_in,   // Aktivasi / piksel input (signed 8-bit)
    input  wire signed [7:0]  weight_in, // Bobot kernel / FC (signed INT8)
    output reg  signed [31:0] acc_out    // Hasil akumulasi 32-bit (mencegah overflow)
);

    wire signed [15:0] prod;
    assign prod = data_in * weight_in;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= 32'sd0;
        end else if (clear_acc) begin
            acc_out <= 32'sd0;
        end else if (en) begin
            acc_out <= acc_out + {{16{prod[15]}}, prod};
        end
    end

endmodule
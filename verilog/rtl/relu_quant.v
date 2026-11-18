module relu_quant #(
    parameter SHIFT_BITS = 8
)(
    input  wire signed [31:0] data_in,
    input  wire               apply_relu, // 1 = terapkan ReLU, 0 = linear (untuk FC2 akhir)
    output reg  signed [7:0]  data_out
);

    wire signed [31:0] shifted;
    assign shifted = data_in >>> SHIFT_BITS;

    always @(*) begin
        if (apply_relu && (shifted < 32'sd0)) begin
            data_out = 8'sd0;
        end else if (shifted > 32'sd127) begin
            data_out = 8'sd127;
        end else if (shifted < -32'sd128) begin
            data_out = -8'sd128;
        end else begin
            data_out = shifted[7:0];
        end
    end

endmodule
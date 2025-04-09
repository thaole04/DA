`timescale 1ns / 1ps

module PE_conv_1x1 #(
    parameter IN_CHANNEL = 20
)(
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      input_ready,
    input  wire [8*IN_CHANNEL-1:0]   input_data,   // unsigned [7:0]
    input  wire [8*IN_CHANNEL-1:0]   kernel_data,  // signed [7:0]
    input  wire [15:0]               coeff,        // Q0.16 unsigned
    input  wire [31:0]               bias,         // Q8.16 signed
    output reg  [7:0]                output_data,  // unsigned 8-bit
    output reg                       output_valid
);

    // Output from macc
    wire [15+$clog2(IN_CHANNEL):0] macc_out;
    wire                           macc_valid;

    // Instantiate MACC module (parallel mult + adder tree)
    macc_8bit_single #(
        .NUM_INPUTS(IN_CHANNEL)
    ) u_macc_8bit_single (
        .o_data   (macc_out),
        .o_valid  (macc_valid),
        .i_data_a (input_data),
        .i_data_b (kernel_data),
        .i_valid  (input_ready),
        .clk      (clk),
        .rst_n    (rst_n)
    );

    // Internal pipeline registers
    reg signed [47:0] scaled_result;
    //reg signed [24:0] scaled_result;
    reg signed [31:0] biased_result;
    reg [7:0]         relu_result;
    reg signed [31:0] biased_rounded;   
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_data  <= 8'd0;
            output_valid <= 1'b0;
        end else begin
            if (macc_valid) begin
                // Multiply by coeff (Q0.16)
                scaled_result = $signed(macc_out) * coeff;

                // Add bias (Q8.16)
                biased_result = scaled_result[31:0] + bias;

                // ReLU + Clamping
                if (biased_result[23:0] < 0)
                    relu_result = 8'd5;
                else
                    biased_rounded = biased_result + 32'h00008000;
                    relu_result = (biased_result[23:16] > 8'd255) ? 8'd255 : biased_result[23:16];

                output_data  <= relu_result;
                output_valid <= 1'b1;
            end else begin
                output_valid <= 1'b0;
            end
        end
    end
endmodule

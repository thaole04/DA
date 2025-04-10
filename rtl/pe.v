`timescale 1ns / 1ps

module PE_conv_1x1 #(
    parameter IN_CHANNEL = 20
)(
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire                      input_ready,
    input  wire [8*IN_CHANNEL-1:0]   input_data,   // unsigned [7:0]
    input  wire [8*IN_CHANNEL-1:0]   kernel_data,  // signed [7:0]
    input  wire [15:0]               coeff,        // Q1.16 unsigned
    input  wire [31:0]               bias,         // Q8.16 signed
    output [7:0]                     output_data,  // unsigned 8-bit
    output                           output_valid
);

    // Output from macc
    localparam MACC_OUTPUT_DATA_WIDTH = 16 + $clog2(IN_CHANNEL);

    wire [MACC_OUTPUT_DATA_WIDTH-1:0] macc_data_out;
    wire                           macc_valid_o;

    // Instantiate MACC module (parallel mult + adder tree)
    macc_8bit_single #(
        .NUM_INPUTS(IN_CHANNEL)
    ) u_macc_8bit_single (
        .o_data   (macc_data_out),
        .o_valid  (macc_valid_o),
        .i_data_a (input_data),
        .i_data_b (kernel_data),
        .i_valid  (input_ready),
        .clk      (clk),
        .rst_n    (rst_n)
    );
    
    // MACC out reg
    reg signed [MACC_OUTPUT_DATA_WIDTH-1:0] macc_data_out_reg;
    reg                                     macc_valid_o_reg;
    
    always @ (posedge clk) begin
        if (macc_valid_o) begin
            macc_data_out_reg <= macc_data_out;
        end
    end

    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            macc_valid_o_reg <= 1'b0;
        end
        else begin
            macc_valid_o_reg <= macc_valid_o;
        end
    end
    
    
    // MACC co-eff
    // Internal pipeline registers
    reg signed [17+MACC_OUTPUT_DATA_WIDTH-1:0] coeff_prod;
    //reg signed [24:0] coeff_prod;
    reg                                        coeff_valid;   
    wire signed [16:0] coeff_ext = {1'b0, coeff};
    
    always @(posedge clk or negedge rst_n) begin
        if (macc_valid_o_reg) begin
          // Multiply by coeff_ext (Q1.16)
          coeff_prod = $signed(macc_data_out_reg) * coeff_ext;
        end
    end
    
    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            coeff_valid <= 1'b0;
        end
        else begin
            coeff_valid <= macc_valid_o_reg;
        end
    end
    
    // Bias
    reg signed [17+MACC_OUTPUT_DATA_WIDTH-1:0] bias_sum;
    //reg signed [31:0] bias_sum;
    reg                                        bias_valid;
    
    always @ (posedge clk) begin
        if (coeff_valid) begin
            // Add bias (Q8.16)
            bias_sum <= coeff_prod + bias;
        end
    end

    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            bias_valid <= 1'b0;
        end
        else begin
            bias_valid <= coeff_valid;
        end
    end
    
    // Output
    assign output_valid = bias_valid;
    assign output_data  = bias_sum; //!
    //always @(posedge clk or negedge rst_n) begin
    //    if (!rst_n) begin
    //        output_data  <= 8'd0;
    //        output_valid <= 1'b0;
    //    end else begin



    //            // ReLU + Clamping
    //            if (bias_sum[23:0] < 0)
    //                relu_result = 8'd5;
    //            else
    //                biased_rounded = bias_sum + 32'h00008000;
    //                relu_result = (bias_sum[23:16] > 8'd255) ? 8'd255 : bias_sum[23:16];

    //            output_data  <= relu_result;
    //            output_valid <= 1'b1;
    //        end else begin
    //            output_valid <= 1'b0;
    //        end
    //    end
    //end
    reg [7:0]         relu_result;
    reg signed [31:0] biased_rounded;
    
endmodule



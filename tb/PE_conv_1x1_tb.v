`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/09/2025 11:40:21 PM
// Design Name: 
// Module Name: PE_conv_1x1_tb
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module PE_conv_1x1_tb(

    );
    // Parameters
    parameter IN_CHANNEL = 4; 

    // Signals
    reg                         clk;
    reg                         rst_n;
    reg                         input_ready;
    reg [8*IN_CHANNEL-1:0]      input_data;
    reg [8*IN_CHANNEL-1:0]      kernel_data;
    reg [15:0]                  coeff;
    reg [31:0]                  bias;
    wire [7:0]                  output_data;
    wire                        output_valid;

    // Instantiate DUT
    PE_conv_1x1 #(
        .IN_CHANNEL(IN_CHANNEL)
    ) dut (
        .clk         (clk),
        .rst_n       (rst_n),
        .input_ready (input_ready),
        .input_data  (input_data),
        .kernel_data (kernel_data),
        .coeff       (coeff),
        .bias        (bias),
        .output_data (output_data),
        .output_valid(output_valid)
    );

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk; // 100MHz clock

    // Reset
    initial begin
        rst_n = 0;
        input_ready = 0;
        input_data = 0;
        kernel_data = 0;
        coeff = 0;
        bias = 0;

        #20;
        rst_n = 1;

        #10;

        // Set test values
        @(posedge clk);
        input_ready = 1;
        input_data = {
            8'd3, 8'd2, 8'd1, 8'd4
        }; // IN_CHANNEL = 4

        kernel_data = {
            $signed(8'sd1), $signed(-8'sd2), $signed(8'sd3), $signed(-8'sd1)
        };

        coeff = 16'h0100; // Q0.16 = 0.00390625
        bias = 32'h00010000; // Q8.16 = 1.0



    
        wait (output_valid);
        $display("Output: %d", output_data);

  
        #50;
        $finish;
    end
endmodule

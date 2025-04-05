`timescale 1ns / 1ps

module pe_incha_single_tb();

    // Parameters
    parameter IN_WIDTH    = 3;
    parameter IN_HEIGHT   = 3;
    parameter IN_CHANNEL  = 2;
    parameter OUT_CHANNEL = 4;
    parameter KERNEL_0    = 3;
    parameter KERNEL_1    = 3;
    parameter KERNEL_PTS  = KERNEL_0 * KERNEL_1;
    parameter OUTPUT_MODE = "relu";
    
    parameter OUTPUT_DATA_WIDTH = OUTPUT_MODE == "relu" ? 8 : 16;

    // Clock generation
    reg clk = 0;
    always #5 clk = ~clk;  // 100 MHz clock

    // Signals
    reg rst_n;
    reg [8*IN_CHANNEL*KERNEL_PTS-1:0] i_data;
    reg i_valid;
    reg [15:0] weight_wr_data;
    reg [31:0] weight_wr_addr;
    reg weight_wr_en;
    
    wire [OUTPUT_DATA_WIDTH*OUT_CHANNEL-1:0] o_data;
    wire o_valid;
    wire pe_ready;
    wire pe_ack;
    
    // Test data registers
    reg [7:0] input_data[0:IN_CHANNEL*KERNEL_PTS-1];
    
    // Định dạng địa chỉ
    parameter ADDR_TYPE_KERNEL = 8'h00;
    parameter ADDR_TYPE_BIAS   = 8'h01;

    // DUT instantiation
    pe_incha_single #(
        .IN_WIDTH    (IN_WIDTH),
        .IN_HEIGHT   (IN_HEIGHT),
        .IN_CHANNEL  (IN_CHANNEL),
        .OUT_CHANNEL (OUT_CHANNEL),
        .OUTPUT_MODE (OUTPUT_MODE),
        .KERNEL_0    (KERNEL_0),
        .KERNEL_1    (KERNEL_1),
        .DILATION_0  (1),
        .DILATION_1  (1),
        .PADDING_0   (1),
        .PADDING_1   (1),
        .STRIDE_0    (1),
        .STRIDE_1    (1)
    ) dut (
        .o_data          (o_data),
        .o_valid         (o_valid),
        .pe_ready        (pe_ready),
        .pe_ack          (pe_ack),
        .i_data          (i_data),
        .i_valid         (i_valid),
        .weight_wr_data  (weight_wr_data),
        .weight_wr_addr  (weight_wr_addr),
        .weight_wr_en    (weight_wr_en),
        .clk             (clk),
        .rst_n           (rst_n)
    );
    
    // Monitor variables
    integer i, j;
    
    // Task for writing kernel weights
    task write_kernel;
        input [7:0] channel;    // Output channel
        input [7:0] position;   // Position in kernel
        input [7:0] value;      // Value to write
    begin
        @(posedge clk);
        weight_wr_en = 1;
        // Format: [31:24]=Type (00=kernel), [23:16]=OutChannel, [15:8]=Position, [7:0]=Reserved
        weight_wr_addr = {ADDR_TYPE_KERNEL, channel, position, 8'h00};
        weight_wr_data = {8'h00, value};    // Chỉ dùng 8 bit thấp cho kernel
        @(posedge clk);
        weight_wr_en = 0;
        #5; // Đợi một chút để đảm bảo ghi xong
    end
    endtask
    
    // Task for writing bias values
    task write_bias;
        input [7:0] channel;         // Output channel
        input signed [15:0] value;   // Bias value (16-bit)
    begin
        @(posedge clk);
        weight_wr_en = 1;
        // Format: [31:24]=Type (01=bias), [23:16]=OutChannel, [15:0]=Reserved
        weight_wr_addr = {ADDR_TYPE_BIAS, channel, 16'h0000};
        weight_wr_data = value;      // Dùng đủ 16 bit cho bias
        @(posedge clk);
        weight_wr_en = 0;
        #5; // Đợi một chút để đảm bảo ghi xong
    end
    endtask
    
    // Task for debugging - capture internal signals
    task debug_values;
    begin
        $display("----- DEBUG VALUES -----");
        $display("kernel_cnt: %0d", dut.kernel_cnt);
        $display("bias_cnt: %0d", dut.bias_cnt);
        $display("macc_data_out: %0d", $signed(dut.macc_data_out));
        $display("macc_data_out_reg: %0d", $signed(dut.macc_data_out_reg));
        $display("coeff_prod: %0d", $signed(dut.coeff_prod));
        $display("bias: %0d", $signed(dut.bias));
        $display("bias_adjusted: %0d", $signed(dut.bias_adjusted));
        $display("bias_sum: %0d", $signed(dut.bias_sum));
        $display("obuffer_data: %0d", $signed(dut.obuffer_data));
        $display("-------------------------");
    end
    endtask
    
    // Task for running one test case
    task run_test;
        input [8*32-1:0] test_name;  // Mảng 32 ký tự (mỗi ký tự 8-bit)
    begin
        $display("\n=== Starting Test: %s ===", test_name);
        
        // Wait for PE to be ready
        wait(pe_ready);
        
        // Prepare the input data
        pack_input_data();
        
        // Send valid data
        @(posedge clk);
        i_valid = 1;
        
        // Wait for acknowledgment
        wait(pe_ack);
        @(posedge clk);
        i_valid = 0;
        
        // Wait for computation to complete
        #50;
        debug_values();
        
        // Wait for result
        wait(o_valid);
        $display("DEBUG: bias_sum = %0d (hex: %0h)", $signed(dut.bias_sum), dut.bias_sum);
        $display("DEBUG: bias_sum[23] = %0d", dut.bias_sum[23]);
        $display("DEBUG: bias_sum[22:16] = %0b", dut.bias_sum[22:16]);
        $display("DEBUG: bias_sum[23:16] = %0d", $signed(dut.bias_sum[23:16]));
        $display("DEBUG: rounding_bit = %0d", dut.gen_relu.rounding_bit);
        // Display results
        @(posedge clk);
        $display("Output valid, data = %h", o_data);
        
        for (i = 0; i < OUT_CHANNEL; i = i + 1) begin
            $display("Channel %0d: %0d", i, $signed(o_data[(i+1)*OUTPUT_DATA_WIDTH-1 -: OUTPUT_DATA_WIDTH]));
        end
        
        $display("=== Test Complete: %s ===\n", test_name);
    end
    endtask
    
    // Helper to pack input data from individual registers to the i_data vector
    task pack_input_data;
    begin
        for (i = 0; i < IN_CHANNEL*KERNEL_PTS; i = i + 1) begin
            i_data[i*8 +: 8] = input_data[i];
        end
    end
    endtask
    
    // Test stimulus
    initial begin
        // Initialize inputs
        rst_n = 0;
        i_data = 0;
        i_valid = 0;
        weight_wr_data = 0;
        weight_wr_addr = 0;
        weight_wr_en = 0;
        
        // Initialize test data
        for (i = 0; i < IN_CHANNEL*KERNEL_PTS; i = i + 1) begin
            input_data[i] = 0;
        end
        
        // Reset sequence
        #20;
        rst_n = 1;
        #20;
        
        $display("========================================");
        $display("=== PE_INCHA_SINGLE TESTBENCH START ===");
        $display("========================================");
        
        $display("\n--- Loading weights and biases ---");
        
        // Write kernel weights for 4 output channels
        // Channel 0: All weights = 1
        for (i = 0; i < KERNEL_PTS * IN_CHANNEL; i = i + 1) begin
            write_kernel(8'd0, i[7:0], 8'd1);
        end
        
        // Channel 1: All weights = 2
        for (i = 0; i < KERNEL_PTS * IN_CHANNEL; i = i + 1) begin
            write_kernel(8'd1, i[7:0], 8'd2);
        end
        
        // Channel 2: All weights = 3
        for (i = 0; i < KERNEL_PTS * IN_CHANNEL; i = i + 1) begin
            write_kernel(8'd2, i[7:0], 8'd3);
        end
        
        // Channel 3: All weights = 4
        for (i = 0; i < KERNEL_PTS * IN_CHANNEL; i = i + 1) begin
            write_kernel(8'd3, i[7:0], 8'd4);
        end
        
        // Write bias values
        write_bias(8'd0, 16'd10);    // Channel 0: bias = 10
        write_bias(8'd1, 16'd20);    // Channel 1: bias = 20
        write_bias(8'd2, -16'd30);   // Channel 2: bias = -30 (để test ReLU)
        write_bias(8'd3, 16'd40);    // Channel 3: bias = 40
        
        $display("All weights and biases written!\n");
        
        // Verify a few weight values (optional)
        $display("Verifying a few weights...");
        $display("Kernel[0][0] = %0d", dut.u_kernel.ram[0][0 +: 8]);
        $display("Kernel[1][0] = %0d", dut.u_kernel.ram[1][0 +: 8]);
        $display("Kernel[2][0] = %0d", dut.u_kernel.ram[2][0 +: 8]);
        $display("Kernel[3][0] = %0d", dut.u_kernel.ram[3][0 +: 8]);
        
        $display("Verifying bias values...");
        $display("Bias[0] = %0d", $signed(dut.u_bias.ram[0]));
        $display("Bias[1] = %0d", $signed(dut.u_bias.ram[1]));
        $display("Bias[2] = %0d", $signed(dut.u_bias.ram[2]));
        $display("Bias[3] = %0d", $signed(dut.u_bias.ram[3]));
        
        // TEST CASE 1: All inputs = 1
        for (i = 0; i < IN_CHANNEL*KERNEL_PTS; i = i + 1) begin
            input_data[i] = 8'd1;
        end
        run_test("All Inputs = 1");
        
        // TEST CASE 2: Channel 0 inputs = 1, Channel 1 inputs = 2
        for (i = 0; i < KERNEL_PTS; i = i + 1) begin
            input_data[i] = 8'd1; // Channel 0
        end
        for (i = KERNEL_PTS; i < 2*KERNEL_PTS; i = i + 1) begin
            input_data[i] = 8'd2; // Channel 1
        end
        run_test("Channel 0 = 1, Channel 1 = 2");
        
        // TEST CASE 3: Increasing values
        for (i = 0; i < IN_CHANNEL*KERNEL_PTS; i = i + 1) begin
            input_data[i] = i + 1;
        end
        run_test("Increasing Values");
        
        // TEST CASE 4: Với hệ số khác
        dut.macc_coeff = 16'd2; // Đặt hệ số = 2
        $display("\n--- Changed coefficient to 2 ---");
        
        for (i = 0; i < IN_CHANNEL*KERNEL_PTS; i = i + 1) begin
            input_data[i] = 8'd1;
        end
        run_test("All Inputs = 1 with coeff = 2");
        
        // Complete testing
        #100;
        
        $display("========================================");
        $display("=== PE_INCHA_SINGLE TESTBENCH END   ===");
        $display("========================================");
        
        $finish;
    end
    
    // Monitor output
    initial begin
        $monitor("Time=%0t, rst_n=%b, i_valid=%b, pe_ready=%b, pe_ack=%b, o_valid=%b", 
                 $time, rst_n, i_valid, pe_ready, pe_ack, o_valid);
    end
    
    // Add waveform dumping if using a simulator that supports it
    initial begin
        $dumpfile("pe_incha_single_tb.vcd");
        $dumpvars(0, pe_incha_single_tb);
    end

endmodule
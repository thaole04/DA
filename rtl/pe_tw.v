`timescale 1ns / 1ps

module pe_test_weight();

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

    // DUT signals
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
        #2; // Đợi một chút để đảm bảo ghi xong
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
        #2; // Đợi một chút để đảm bảo ghi xong
    end
    endtask
    
    // Task for verifying kernel weight - sửa lại
    task check_kernel_weight;
        input [7:0] channel;    // Output channel
        input [7:0] position;   // Position in kernel
        input [7:0] expected;   // Expected value
        
        reg [7:0] actual;
    begin
        // Truy cập đúng vào từng word trong RAM
        actual = dut.u_kernel.ram[channel][position*8 +: 8];
        
        if (actual === expected) begin
            $display("[OK] Kernel[%0d][%0d] = %0d (expected %0d)", 
                    channel, position, actual, expected);
        end else begin
            $display("[ERROR] Kernel[%0d][%0d] = %0d (expected %0d)", 
                    channel, position, actual, expected);
        end
    end
    endtask
    
    // Task for verifying bias
    task check_bias;
        input [7:0] channel;           // Output channel 
        input signed [15:0] expected;  // Expected value
        
        reg signed [15:0] actual;
    begin
        actual = dut.u_bias.ram[channel];
        
        if (actual === expected) begin
            $display("[OK] Bias[%0d] = %0d (expected %0d)", channel, $signed(actual), $signed(expected));
        end
        else begin
            $display("[ERROR] Bias[%0d] = %0d (expected %0d)", channel, $signed(actual), $signed(expected));
        end
    end
    endtask

    // Bước ghi và kiểm tra
    integer i;
    
    initial begin
        // Initialize signals
        rst_n = 0;
        i_data = 0;
        i_valid = 0;
        weight_wr_data = 0;
        weight_wr_addr = 0;
        weight_wr_en = 0;
        
        // Reset
        #20 rst_n = 1;
        #20;
        
        $display("--- TEST 1: WRITING KERNEL WEIGHTS ---");
        
        // Ghi các giá trị khác nhau vào từng vị trí kernel khác nhau
        // Channel 0: Ghi giá trị bằng vị trí
        for (i = 0; i < KERNEL_PTS * IN_CHANNEL; i = i + 1) begin
            write_kernel(8'd0, i[7:0], i[7:0] + 10);
        end
        
        // Channel 1: Ghi giá trị bằng 2 * vị trí
        for (i = 0; i < KERNEL_PTS * IN_CHANNEL; i = i + 1) begin
            write_kernel(8'd1, i[7:0], i[7:0] * 2);
        end
        
        // Channel 2: Ghi giá trị bằng 100 - vị trí
        for (i = 0; i < KERNEL_PTS * IN_CHANNEL; i = i + 1) begin
            write_kernel(8'd2, i[7:0], 100 - i[7:0]);
        end
        
        // Channel 3: Ghi giá trị bằng 5
        for (i = 0; i < KERNEL_PTS * IN_CHANNEL; i = i + 1) begin
            write_kernel(8'd3, i[7:0], 5);
        end
        
        #10;
        $display("--- VERIFYING KERNEL WEIGHTS ---");
        
        // Kiểm tra các giá trị đã ghi
        // Kiểm tra Channel 0
        for (i = 0; i < KERNEL_PTS * IN_CHANNEL; i = i + 1) begin
            check_kernel_weight(8'd0, i[7:0], i[7:0] + 10);
        end
        
        // Kiểm tra Channel 1
        for (i = 0; i < KERNEL_PTS * IN_CHANNEL; i = i + 1) begin
            check_kernel_weight(8'd1, i[7:0], i[7:0] * 2);
        end
        
        // Kiểm tra Channel 2
        for (i = 0; i < KERNEL_PTS * IN_CHANNEL; i = i + 1) begin
            check_kernel_weight(8'd2, i[7:0], 100 - i[7:0]);
        end
        
        // Kiểm tra Channel 3
        for (i = 0; i < KERNEL_PTS * IN_CHANNEL; i = i + 1) begin
            check_kernel_weight(8'd3, i[7:0], 5);
        end
        
        $display("--- TEST 2: WRITING BIAS VALUES ---");
        
        // Ghi các giá trị bias khác nhau
        write_bias(8'd0, 16'd100);     // Giá trị bias dương nhỏ
        write_bias(8'd1, 16'd1000);    // Giá trị bias dương trung bình
        write_bias(8'd2, -16'd50);     // Giá trị bias âm
        write_bias(8'd3, 16'd32767);   // Giá trị bias dương lớn nhất
        
        #10;
        $display("--- VERIFYING BIAS VALUES ---");
        
        // Kiểm tra các giá trị bias đã ghi
        check_bias(8'd0, 16'd100);
        check_bias(8'd1, 16'd1000);
        check_bias(8'd2, -16'd50);
        check_bias(8'd3, 16'd32767);
        
        // Test xung đột địa chỉ
        $display("--- TEST 3: ADDRESS CONFLICT TEST ---");
        
        // Ghi kernel và bias vào cùng một output channel
        write_kernel(8'd0, 8'd0, 8'd42);   // Kernel channel 0, vị trí 0
        check_kernel_weight(8'd0, 8'd0, 8'd42);
        
        write_bias(8'd0, 16'd200);         // Bias channel 0
        check_bias(8'd0, 16'd200);
        
        // Kiểm tra xem việc ghi bias không làm thay đổi kernel
        check_kernel_weight(8'd0, 8'd0, 8'd42);
        
        $display("--- TEST 4: PARTIAL ADDRESS WRITE TEST ---");
        
        // Ghi chỉ đặt một số bit của địa chỉ
        weight_wr_en = 1;
        weight_wr_addr = {ADDR_TYPE_KERNEL, 8'd2, 8'd5, 8'h00};
        weight_wr_data = 16'h00A5;
        @(posedge clk);
        weight_wr_en = 0;
        #2;
        
        check_kernel_weight(8'd2, 8'd5, 8'hA5);
        
        $display("--- TEST 5: MULTIPLE CONSECUTIVE WRITES ---");
        
        // Ghi nhiều lần liên tiếp không có khoảng trống
        @(posedge clk);
        
        // Ghi liên tiếp 5 giá trị vào channel 3
        weight_wr_en = 1;
        for (i = 0; i < 5; i = i + 1) begin
            weight_wr_addr = {ADDR_TYPE_KERNEL, 8'd3, i[7:0], 8'h00};
            weight_wr_data = 16'h0080 + i;
            @(posedge clk);
        end
        weight_wr_en = 0;
        #2;
        
        // Kiểm tra các giá trị đã ghi
        for (i = 0; i < 5; i = i + 1) begin
            check_kernel_weight(8'd3, i[7:0], 8'h80 + i);
        end
        
        $display("--- WEIGHT MEMORY TEST COMPLETE ---");
        
        #100;
        $finish;
    end
    
    // Monitor output
    initial begin
        $monitor("Time=%0t, rst_n=%b, weight_wr_en=%b, weight_wr_addr=%h, weight_wr_data=%h", 
                 $time, rst_n, weight_wr_en, weight_wr_addr, weight_wr_data);
    end
    
    // Dump waveforms
    initial begin
        $dumpfile("pe_test_weight.vcd");
        $dumpvars(0, pe_test_weight);
    end
    
endmodule
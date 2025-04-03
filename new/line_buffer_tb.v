`timescale 1ns / 1ps

module line_buffer_tb();

    // Tham số cho testbench - sử dụng giá trị nhỏ để dễ quan sát
    parameter DATA_WIDTH = 8;        // Độ rộng dữ liệu 8 bit
    parameter IN_CHANNEL = 1;        // Chỉ 1 kênh để đơn giản
    parameter IN_WIDTH   = 4;        // Ma trận nhỏ 4x4
    parameter IN_HEIGHT  = 4;
    parameter KERNEL_0   = 3;        // Kernel 3x3
    parameter KERNEL_1   = 3;
    parameter DILATION_0 = 1;        // Dilation 1 (thông thường)
    parameter DILATION_1 = 1;
    parameter PADDING_0  = 1;        // Padding 1
    parameter PADDING_1  = 1;
    parameter STRIDE_0   = 1;        // Stride 1
    parameter STRIDE_1   = 1;

    // Tín hiệu cần thiết
    reg                            clk;
    reg                            rst_n;
    reg  [DATA_WIDTH*IN_CHANNEL-1:0] i_data;
    reg                            i_valid;
    reg                            fifo_almost_full;
    reg                            pe_ready;
    reg                            pe_ack;
    
    wire [DATA_WIDTH*IN_CHANNEL*KERNEL_0*KERNEL_1-1:0] o_data;
    wire                                            o_valid;
    wire                                            fifo_rd_en;
    
    // Instantiate DUT
    line_buffer #(
        .DATA_WIDTH   (DATA_WIDTH),
        .IN_CHANNEL   (IN_CHANNEL),
        .IN_WIDTH     (IN_WIDTH),
        .IN_HEIGHT    (IN_HEIGHT),
        .KERNEL_0     (KERNEL_0),
        .KERNEL_1     (KERNEL_1),
        .DILATION_0   (DILATION_0),
        .DILATION_1   (DILATION_1),
        .PADDING_0    (PADDING_0),
        .PADDING_1    (PADDING_1),
        .STRIDE_0     (STRIDE_0),
        .STRIDE_1     (STRIDE_1)
    ) dut (
        .o_data           (o_data),
        .o_valid          (o_valid),
        .fifo_rd_en       (fifo_rd_en),
        .i_data           (i_data),
        .i_valid          (i_valid),
        .fifo_almost_full (fifo_almost_full),
        .pe_ready         (pe_ready),
        .pe_ack           (pe_ack),
        .clk              (clk),
        .rst_n            (rst_n)
    );
    
    // Biến cho quá trình test
    integer i, j;
    reg [7:0] test_matrix [0:IN_HEIGHT-1][0:IN_WIDTH-1];
    integer pixel_count = 0;
    
    // Tạo xung đồng hồ
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // Chu kỳ 10ns
    end
    
    // Hiển thị kết quả
    task display_output;
        input [DATA_WIDTH*IN_CHANNEL*KERNEL_0*KERNEL_1-1:0] data;
        integer k, l;
        reg [DATA_WIDTH-1:0] pixel;
        begin
            $display("Kernel window:");
            for (k = 0; k < KERNEL_0; k = k + 1) begin
                for (l = 0; l < KERNEL_1; l = l + 1) begin
                    pixel = data[(k*KERNEL_1+l+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                    $write("%d\t", pixel);
                end
                $write("\n");
            end
            $write("\n");
        end
    endtask
    
    // Quá trình test
    initial begin
        // Khởi tạo
        rst_n = 0;
        i_valid = 0;
        i_data = 0;
        fifo_almost_full = 0;
        pe_ready = 1; // PE luôn sẵn sàng
        pe_ack = 0;
        
        // Khởi tạo ma trận test với giá trị tăng dần
        for (i = 0; i < IN_HEIGHT; i = i + 1) begin
            for (j = 0; j < IN_WIDTH; j = j + 1) begin
                test_matrix[i][j] = i*IN_WIDTH + j + 1; // Giá trị 1-16 cho ma trận 4x4
            end
        end
        
        // Hiển thị ma trận đầu vào
        $display("Ma trận đầu vào (4x4):");
        for (i = 0; i < IN_HEIGHT; i = i + 1) begin
            for (j = 0; j < IN_WIDTH; j = j + 1) begin
                $write("%d\t", test_matrix[i][j]);
            end
            $write("\n");
        end
        $display("");
        
        // Reset hệ thống
        #20;
        rst_n = 1;
        #10;
        
        // Đưa dữ liệu vào theo thứ tự
        for (i = 0; i < IN_HEIGHT; i = i + 1) begin
            for (j = 0; j < IN_WIDTH; j = j + 1) begin
                @(posedge clk);
                i_valid = 1;
                i_data = test_matrix[i][j];
                pixel_count = pixel_count + 1;
                
                // Hiển thị hành động
                $display("Đưa vào pixel: %d (%0d,%0d)", test_matrix[i][j], i, j);
                
                // Nếu có đầu ra hợp lệ, hiển thị
                @(posedge clk);
                if (o_valid) begin
                    display_output(o_data);
                    pe_ack = 1;
                    @(posedge clk);
                    pe_ack = 0;
                end
            end
        end
        
        // Đảm bảo tất cả các pixel đã được xử lý
        repeat (10) begin
            @(posedge clk);
            i_valid = 0;
            // Kiểm tra đầu ra
            if (o_valid) begin
                display_output(o_data);
                pe_ack = 1;
                @(posedge clk);
                pe_ack = 0;
            end
        end
        
        $display("Hoàn thành testbench");
        $finish;
    end
    
    // Giám sát các tín hiệu điều khiển
    initial begin
        $monitor("Thời gian=%0t, rst_n=%b, i_valid=%b, o_valid=%b, fifo_rd_en=%b, is_padding=%b", 
                 $time, rst_n, i_valid, o_valid, fifo_rd_en, dut.is_padding);
    end

endmodule
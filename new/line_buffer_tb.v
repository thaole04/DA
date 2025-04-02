`timescale 1ns / 1ps

module line_buffer_tb;

    // Tham số
    parameter IMG_WIDTH = 6;
    parameter IMG_HEIGHT = 6;
    parameter CHANNELS = 3;
    parameter KERNEL_SIZE = 3;
    parameter PADDING = 1;
    parameter DATA_WIDTH = 8;
    
    // Số window kỳ vọng
    parameter EXPECTED_WINDOWS = (IMG_WIDTH + 2*PADDING - KERNEL_SIZE + 1) * 
                                 (IMG_HEIGHT + 2*PADDING - KERNEL_SIZE + 1);
    
    // Tín hiệu
    reg clk;
    reg rst_n;
    reg [DATA_WIDTH*CHANNELS-1:0] fifo_data;
    reg fifo_empty;
    reg window_ready;
    
    wire fifo_read_en;
    wire window_valid;
    wire [DATA_WIDTH*KERNEL_SIZE*KERNEL_SIZE*CHANNELS-1:0] window_data;
    
    // Biến đếm
    integer window_count;
    integer i, j, c;
    
    // Khởi tạo module
    line_buffer #(
        .IMG_WIDTH(IMG_WIDTH),
        .IMG_HEIGHT(IMG_HEIGHT),
        .CHANNELS(CHANNELS),
        .KERNEL_SIZE(KERNEL_SIZE),
        .PADDING(PADDING),
        .DATA_WIDTH(DATA_WIDTH)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .fifo_read_en(fifo_read_en),
        .fifo_data(fifo_data),
        .fifo_empty(fifo_empty),
        .window_valid(window_valid),
        .window_data(window_data),
        .window_ready(window_ready)
    );
    
    // Tạo xung clock 10ns (100MHz)
    always #5 clk = ~clk;
    
    // Mảng lưu dữ liệu đầu vào
    reg [DATA_WIDTH-1:0] input_data_r [0:IMG_HEIGHT-1][0:IMG_WIDTH-1];
    reg [DATA_WIDTH-1:0] input_data_g [0:IMG_HEIGHT-1][0:IMG_WIDTH-1];
    reg [DATA_WIDTH-1:0] input_data_b [0:IMG_HEIGHT-1][0:IMG_WIDTH-1];
    
    initial begin
        // Khởi tạo tín hiệu
        clk = 0;
        rst_n = 0;
        fifo_empty = 1;
        window_ready = 0;
        window_count = 0;
        fifo_data = 0;
        
        // Khởi tạo dữ liệu ảnh đầu vào
        for (i = 0; i < IMG_HEIGHT; i = i + 1) begin
            for (j = 0; j < IMG_WIDTH; j = j + 1) begin
                input_data_r[i][j] = 1;
                input_data_g[i][j] = 2;
                input_data_b[i][j] = 3;
            end
        end
        
        // Reset module
        #20 rst_n = 1;
        
        // Bắt đầu quá trình kiểm tra
        #10 fifo_empty = 0;
        
        // Đưa dữ liệu vào module từ FIFO
        for (i = 0; i < IMG_HEIGHT; i = i + 1) begin
            for (j = 0; j < IMG_WIDTH; j = j + 1) begin
                @(posedge clk);
                while (!fifo_read_en) @(posedge clk); // Đợi cho đến khi module sẵn sàng đọc
                
                fifo_data = {input_data_b[i][j], input_data_g[i][j], input_data_r[i][j]};
                fifo_empty = 0; // Đánh dấu FIFO không trống
            end
        end
        
        // Đánh dấu FIFO trống sau khi load xong
        @(posedge clk) fifo_empty = 1;
        $display("FIFO đã trống sau khi load xong dữ liệu.");
        // Thêm trong testbench
        $display("Buffer R sau khi load:");
        for (i = 0; i < 8; i = i + 1) begin
            for (j = 0; j < 8; j = j + 1) begin
                $write("%3d ", uut.buffer_r[i][j]);
            end
            $write("\n");
        end
        $display("Buffer G sau khi load:");
        for (i = 0; i < 8; i = i + 1) begin
            for (j = 0; j < 8; j = j + 1) begin
                $write("%3d ", uut.buffer_g[i][j]);
            end
            $write("\n");
        end
        $display("Buffer B sau khi load:");
        for (i = 0; i < 8; i = i + 1) begin
            for (j = 0; j < 8; j = j + 1) begin
                $write("%3d ", uut.buffer_b[i][j]);
            end
            $write("\n");
        end

        // Thay vì đợi cố định, chờ đến khi module chắc chắn chuyển sang PROCESS
        $display("Đợi module chuyển sang trạng thái PROCESS...");
        while (uut.state != 2) @(posedge clk);
        $display("Module đã chuyển sang trạng thái PROCESS tại thời điểm %0t", $time);

        // Thêm một chút thời gian để đảm bảo trạng thái ổn định
        #10;
        
        // Bắt đầu nhận window
        window_ready = 1;
        $display("Đã đặt window_ready = 1, bắt đầu nhận cửa sổ...");
        
        // Đếm số window và kiểm tra giá trị
        while (window_count < EXPECTED_WINDOWS) begin
            @(posedge clk);
            
            if (window_valid) begin
                window_count = window_count + 1;
                
                // In thông tin window hiện tại
                $display("Window #%0d:", window_count);
                
                // Kiểm tra và in ra từng kênh màu
                for (c = 0; c < CHANNELS; c = c + 1) begin
                    $display("  Channel %0d:", c);
                    for (i = 0; i < KERNEL_SIZE; i = i + 1) begin
                        $write("    ");
                        for (j = 0; j < KERNEL_SIZE; j = j + 1) begin
                            $write("%3d ", window_data[(c*KERNEL_SIZE*KERNEL_SIZE + i*KERNEL_SIZE + j)*DATA_WIDTH +: DATA_WIDTH]);
                        end
                        $write("\n");
                    end
                end
                
                // Tạm dừng ready để kiểm tra xử lý window_ready
                if (window_count % 5 == 0) begin
                    window_ready = 0;
                    #20 window_ready = 1;
                end
            end
        end
        
        // Kiểm tra tổng số window
        if (window_count == EXPECTED_WINDOWS) begin
            $display("TEST PASSED: Detected %0d windows (expected %0d)", window_count, EXPECTED_WINDOWS);
        end else begin
            $display("TEST FAILED: Detected %0d windows (expected %0d)", window_count, EXPECTED_WINDOWS);
        end
        
        // Kết thúc mô phỏng
        #1000 $finish;
    end
    
    // Monitor để theo dõi tín hiệu
    initial begin
        $monitor("Time: %0t, State: %0d, Row: %0d, Col: %0d, Window Valid: %0b, Window Count: %0d",
                 $time, uut.state, uut.window_row, uut.window_col, window_valid, window_count);
    end

endmodule
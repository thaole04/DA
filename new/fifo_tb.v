`timescale 1ns / 1ps

module fifo_single_read_tb();

    // Tham số
    parameter DATA_WIDTH = 8;
    parameter DEPTH = 8;
    parameter CLK_PERIOD = 10; // 10ns

    // Tín hiệu
    reg [DATA_WIDTH-1:0] wr_data;
    reg wr_en;
    reg rd_en;
    reg rst_n;
    reg clk;
    
    wire [DATA_WIDTH-1:0] rd_data;
    wire empty;
    wire full;
    wire almost_full;
    
    // Khởi tạo DUT (Device Under Test)
    fifo_single_read #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH),
        .ALMOST_FULL_THRES(2)  // Báo gần đầy khi còn 2 vị trí trống
    ) dut (
        .rd_data(rd_data),
        .empty(empty),
        .full(full),
        .almost_full(almost_full),
        .wr_data(wr_data),
        .wr_en(wr_en),
        .rd_en(rd_en),
        .rst_n(rst_n),
        .clk(clk)
    );
    
    // Tạo xung nhịp đồng hồ
    always begin
        #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Hiển thị thông tin trong quá trình mô phỏng
    initial begin
        $monitor("Time=%0t, wr_en=%b, rd_en=%b, wr_data=%h, rd_data=%h, empty=%b, full=%b, almost_full=%b", 
                $time, wr_en, rd_en, wr_data, rd_data, empty, full, almost_full);
    end
    
    // Kịch bản kiểm tra
    initial begin
        // Khởi tạo các tín hiệu
        clk = 0;
        rst_n = 0;
        wr_en = 0;
        rd_en = 0;
        wr_data = 8'h00;
        
        // Reset hệ thống
        #20 rst_n = 1;
        #10;
        
        // Kiểm tra 1: Trạng thái ban đầu
        $display("1. Kiểm tra trạng thái sau reset");
        if (empty) $display("PASS: FIFO rỗng sau reset");
        else $display("FAIL: FIFO không rỗng sau reset");
        
        // Kiểm tra 2: Ghi dữ liệu vào FIFO đến khi đầy
        $display("2. Ghi dữ liệu vào FIFO đến khi đầy");
        wr_en = 1;
        for (integer i = 1; i <= DEPTH; i = i + 1) begin
            wr_data = i;
            #CLK_PERIOD;
        end
        wr_en = 0;
        
        if (full) $display("PASS: FIFO đầy sau khi ghi %0d phần tử", DEPTH);
        else $display("FAIL: FIFO không đầy sau khi ghi %0d phần tử", DEPTH);
        
        // Kiểm tra 3: Đọc dữ liệu từ FIFO
        $display("3. Đọc dữ liệu từ FIFO");
        rd_en = 1;
        for (integer i = 1; i <= DEPTH; i = i + 1) begin
            #CLK_PERIOD;
            $display("Đọc dữ liệu thứ %0d: %h", i, rd_data);
        end
        rd_en = 0;
        
        if (empty) $display("PASS: FIFO rỗng sau khi đọc hết");
        else $display("FAIL: FIFO không rỗng sau khi đọc hết");
        
        // Kiểm tra 4: Ghi và đọc đan xen
        $display("4. Ghi và đọc đan xen");
        for (integer i = 1; i <= 4; i = i + 1) begin
            // Ghi
            wr_en = 1;
            wr_data = 8'hA0 + i;
            #CLK_PERIOD;
            wr_en = 0;
            
            // Đọc sau khi ghi
            rd_en = 1;
            #CLK_PERIOD;
            rd_en = 0;
            $display("Đọc dữ liệu: %h", rd_data);
        end
        
        // Kiểm tra 5: Kiểm tra almost_full
        $display("5. Kiểm tra tín hiệu almost_full");
        wr_en = 1;
        for (integer i = 1; i <= DEPTH-2; i = i + 1) begin
            wr_data = 8'hB0 + i;
            #CLK_PERIOD;
        end
        
        if (almost_full) $display("PASS: FIFO báo almost_full khi còn 2 vị trí trống");
        else $display("FAIL: FIFO không báo almost_full khi còn 2 vị trí trống");
        
        // Ghi thêm đến khi đầy
        for (integer i = 1; i <= 2; i = i + 1) begin
            wr_data = 8'hC0 + i;
            #CLK_PERIOD;
        end
        wr_en = 0;
        
        // Kết thúc mô phỏng
        $display("Kết thúc mô phỏng");
        #100;
        $finish;
    end
    
endmodule
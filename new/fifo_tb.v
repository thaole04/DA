`timescale 1ns/1ps

module fifo_tb();
    // Các tham số cho testbench
    parameter DATA_WIDTH = 8;
    parameter FIFO_DEPTH = 16;  // Dùng độ sâu nhỏ hơn để dễ kiểm tra
    parameter CLK_PERIOD = 10;  // 10ns = 100MHz
    
    // Các tín hiệu để kết nối với module FIFO
    reg clk;
    reg rst_n;
    reg write_en;
    reg read_en;
    reg [DATA_WIDTH-1:0] data_in;
    wire [DATA_WIDTH-1:0] data_out;
    wire empty;
    wire full;
    
    // Biến đếm và kiểm tra
    integer i;
    integer error_count = 0;
    reg [DATA_WIDTH-1:0] expected_data;
    
    // Khởi tạo đối tượng FIFO
    fifo #(
        .DATA_WIDTH(DATA_WIDTH),
        .FIFO_DEPTH(FIFO_DEPTH)
    ) fifo_inst (
        .clk(clk),
        .rst_n(rst_n),
        .write_en(write_en),
        .read_en(read_en),
        .data_in(data_in),
        .data_out(data_out),
        .empty(empty),
        .full(full)
    );
    
    // Tạo xung clock
    always begin
        #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Kiểm tra các chức năng của FIFO
    initial begin
        // Khởi tạo
        clk = 0;
        rst_n = 0;
        write_en = 0;
        read_en = 0;
        data_in = 0;
        
        // Hiển thị thông tin bắt đầu
        $display("=== BẮT ĐẦU KIỂM TRA FIFO ===");
        $display("Thời gian: %0t ns", $time);
        $display("Độ rộng dữ liệu: %0d bits", DATA_WIDTH);
        $display("Độ sâu FIFO: %0d phần tử", FIFO_DEPTH);
        
        // Reset hệ thống
        #(CLK_PERIOD*2);
        rst_n = 1;
        #(CLK_PERIOD);
        
        // Kiểm tra cờ empty sau khi reset
        if (!empty) begin
            $display("LỖI: FIFO không rỗng sau khi reset");
            error_count = error_count + 1;
        end else begin
            $display("OK: FIFO rỗng sau khi reset");
        end
        
        // Kiểm tra cờ full sau khi reset
        if (full) begin
            $display("LỖI: FIFO đầy sau khi reset");
            error_count = error_count + 1;
        end else begin
            $display("OK: FIFO không đầy sau khi reset");
        end
        
        // Kiểm tra ghi dữ liệu vào FIFO
        $display("\n--- KIỂM TRA GHI DỮ LIỆU ---");
        for (i = 0; i < FIFO_DEPTH; i = i + 1) begin
            data_in = i + 10;  // Giá trị bắt đầu từ 10
            write_en = 1;
            @(posedge clk);
            #1; // Đợi một chút để tín hiệu cập nhật
            
            $display("Ghi dữ liệu: %0d, Full: %0b, Empty: %0b", data_in, full, empty);
            
            if (i == FIFO_DEPTH-1 && !full) begin
                $display("LỖI: FIFO phải đầy sau khi ghi %0d phần tử", FIFO_DEPTH);
                error_count = error_count + 1;
            end
        end
        
        write_en = 0;
        #(CLK_PERIOD);
        
        // Kiểm tra đọc dữ liệu từ FIFO
        $display("\n--- KIỂM TRA ĐỌC DỮ LIỆU ---");
        for (i = 0; i < FIFO_DEPTH; i = i + 1) begin
            expected_data = i + 10;  // Giá trị bắt đầu từ 10
            read_en = 1;
            @(posedge clk);
            #1; // Đợi một chút để tín hiệu cập nhật
            
            $display("Đọc dữ liệu: %0d, Kỳ vọng: %0d, Full: %0b, Empty: %0b", 
                      data_out, expected_data, full, empty);
                      
            if (data_out !== expected_data) begin
                $display("LỖI: Dữ liệu đọc ra không khớp với kỳ vọng");
                error_count = error_count + 1;
            end
            
            if (i == FIFO_DEPTH-1 && !empty) begin
                $display("LỖI: FIFO phải rỗng sau khi đọc %0d phần tử", FIFO_DEPTH);
                error_count = error_count + 1;
            end
        end
        
        read_en = 0;
        #(CLK_PERIOD);
        
        // Kiểm tra trường hợp đặc biệt: Đọc khi FIFO rỗng
        $display("\n--- KIỂM TRA ĐỌC KHI FIFO RỖNG ---");
        read_en = 1;
        @(posedge clk);
        #1;
        $display("Đọc khi FIFO rỗng: data_out = %0d", data_out);
        read_en = 0;
        #(CLK_PERIOD);
        
        // Kiểm tra trường hợp đặc biệt: Ghi và đọc đồng thời
        $display("\n--- KIỂM TRA GHI VÀ ĐỌC ĐỒNG THỜI ---");
        for (i = 0; i < 5; i = i + 1) begin
            data_in = 100 + i;
            write_en = 1;
            read_en = 1;
            @(posedge clk);
            #1;
            $display("Ghi: %0d, Đọc: %0d, Full: %0b, Empty: %0b", 
                      data_in, data_out, full, empty);
        end
        
        write_en = 0;
        read_en = 0;
        #(CLK_PERIOD);
        
        // Hiển thị kết quả cuối cùng
        $display("\n=== KẾT QUẢ KIỂM TRA ===");
        if (error_count == 0) begin
            $display("THÀNH CÔNG: Tất cả các kiểm tra đã vượt qua");
        end else begin
            $display("THẤT BẠI: Có %0d lỗi xảy ra trong quá trình kiểm tra", error_count);
        end
        
        #(CLK_PERIOD*5);
        $finish;
    end
    
    // Hiển thị thông tin khi FIFO đầy
    always @(posedge full) begin
        $display("FIFO đã đầy tại thời điểm %0t ns", $time);
    end
    
    // Hiển thị thông tin khi FIFO rỗng
    always @(posedge empty) begin
        $display("FIFO đã rỗng tại thời điểm %0t ns", $time);
    end
    
    // Tạo file VCD để quan sát sóng
    initial begin
        $dumpfile("fifo_tb.vcd");
        $dumpvars(0, fifo_tb);
    end
    
endmodule
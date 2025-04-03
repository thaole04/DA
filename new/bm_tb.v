`timescale 1ns / 1ps

module block_ram_multi_word_tb();

    // Tham số
    parameter DATA_WIDTH = 8;
    parameter DEPTH = 4;        // Kích thước nhỏ hơn để dễ quan sát
    parameter NUM_WORDS = 9;    // 4 từ mỗi địa chỉ
    parameter CLK_PERIOD = 10;  // 10ns

    // Tín hiệu
    reg [DATA_WIDTH-1:0] wr_data;
    reg [$clog2(DEPTH)-1:0] wr_addr;
    reg [$clog2(DEPTH)-1:0] rd_addr;
    reg [NUM_WORDS-1:0] wr_en;
    reg rd_en;
    reg clk;
    
    wire [DATA_WIDTH*NUM_WORDS-1:0] rd_data;
    
    // Khởi tạo DUT (Device Under Test)
    block_ram_multi_word #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH),
        .NUM_WORDS(NUM_WORDS),
        .RAM_STYLE("auto"),
        .OUTPUT_REGISTER("false")
    ) dut (
        .rd_data(rd_data),
        .wr_data(wr_data),
        .wr_addr(wr_addr),
        .rd_addr(rd_addr),
        .wr_en(wr_en),
        .rd_en(rd_en),
        .clk(clk)
    );
    
    // Tạo xung nhịp đồng hồ
    always begin
        #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Hàm hiển thị dữ liệu theo từng word
    task display_data(input [$clog2(DEPTH)-1:0] addr, input [DATA_WIDTH*NUM_WORDS-1:0] data);
        reg [DATA_WIDTH-1:0] words [0:NUM_WORDS-1]; // Mảng tạm để lưu từng word
        begin
            // Trích xuất từng word từ dữ liệu (cách thủ công)
            words[0] = data[DATA_WIDTH-1:0];
            words[1] = data[2*DATA_WIDTH-1:DATA_WIDTH];
            words[2] = data[3*DATA_WIDTH-1:2*DATA_WIDTH];
            words[3] = data[4*DATA_WIDTH-1:3*DATA_WIDTH];
            
            $display("Địa chỉ RAM[%0d]:", addr);
            for (integer j = 0; j < NUM_WORDS; j = j + 1) begin
                $display("  Word %0d: %b (Hex: %h)", j, words[j], words[j]);
            end
            $display("  Giá trị đầy đủ: %b", data);
            $display("  Kiểm tra: %s", (data === {DATA_WIDTH*NUM_WORDS{1'b1}}) ? "PASS" : "FAIL");
            $display("");
        end
    endtask
    
    // Kịch bản kiểm tra
    initial begin
        // Khởi tạo các tín hiệu
        clk = 0;
        wr_en = 0;
        rd_en = 1;
        wr_addr = 0;
        rd_addr = 0;
        wr_data = 0;
        
        // Đợi vài chu kỳ clock để RAM khởi tạo hoàn tất
        #30;
        
        // In header
        $display("======== Kiểm tra Giá trị Khởi tạo RAM Multi-Word ========");
        $display("Cấu hình: DATA_WIDTH=%0d, DEPTH=%0d, NUM_WORDS=%0d", 
                DATA_WIDTH, DEPTH, NUM_WORDS);
        $display("Mỗi địa chỉ RAM chứa %0d bit dữ liệu", DATA_WIDTH*NUM_WORDS);
        $display("");
        
        // Đọc tuần tự các giá trị trong RAM
        for (integer i = 0; i < DEPTH; i = i + 1) begin
            rd_addr = i;
            #CLK_PERIOD; // Đợi 1 chu kỳ clock cho giá trị đọc cập nhật
            
            // Hiển thị giá trị đọc được
            display_data(i, rd_data);
        end
        
        // Kết thúc mô phỏng
        $display("================ Kết thúc kiểm tra ================");
        #10 $finish;
    end
    
endmodule
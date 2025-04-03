`timescale 1ns / 1ps

module block_ram_single_port_tb();

    // Tham số
    parameter DATA_WIDTH = 8;
    parameter DEPTH = 8;
    parameter CLK_PERIOD = 10; // 10ns

    // Tín hiệu
    reg [DATA_WIDTH-1:0] wr_data;
    reg [$clog2(DEPTH)-1:0] wr_addr;
    reg [$clog2(DEPTH)-1:0] rd_addr;
    reg wr_en;
    reg rd_en;
    reg clk;
    
    wire [DATA_WIDTH-1:0] rd_data;
    
    // Khởi tạo DUT (Device Under Test)
    block_ram_single_port #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH),
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
    
    // Kịch bản kiểm tra
    initial begin
        // Khởi tạo các tín hiệu
        clk = 0;
        wr_en = 0;
        rd_en = 1; // Luôn bật đọc
        wr_addr = 0;
        rd_addr = 0;
        wr_data = 0;
        
        // Đợi vài chu kỳ clock để RAM khởi tạo hoàn tất
        #20;
        
        // In header
        $display("========== Kiểm tra Giá trị Khởi tạo RAM ==========");
        $display("Địa chỉ\t\tGiá trị\t\tKiểm tra");
        
        // Đọc tuần tự các giá trị trong RAM
        for (integer i = 0; i < DEPTH; i = i + 1) begin
            rd_addr = i;
            #CLK_PERIOD; // Đợi 1 chu kỳ clock cho giá trị đọc cập nhật
            
            // In giá trị đọc và kiểm tra xem có bằng giá trị toàn 1 không
            $display("%d\t\t%b\t%s", 
                    i, rd_data, 
                    (rd_data === {DATA_WIDTH{1'b1}}) ? "PASS" : "FAIL");
        end
        
        // Kết thúc mô phỏng
        $display("================ Kết thúc kiểm tra ================");
        #10 $finish;
    end
    
endmodule
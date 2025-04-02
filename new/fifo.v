module fifo #(
    parameter DATA_WIDTH = 8,          // Độ rộng bit của dữ liệu
    parameter FIFO_DEPTH = 256         // Độ sâu của FIFO
)(
    input wire clk,                     // Tín hiệu clock
    input wire rst_n,                   // Tín hiệu reset (active low)
    input wire write_en,                // Cho phép ghi
    input wire read_en,                 // Cho phép đọc
    input wire [DATA_WIDTH-1:0] data_in, // Dữ liệu đầu vào
    output reg [DATA_WIDTH-1:0] data_out, // Dữ liệu đầu ra
    output reg empty,                   // Báo FIFO rỗng
    output reg full                     // Báo FIFO đầy
);

    // Mảng lưu trữ dữ liệu
    reg [DATA_WIDTH-1:0] memory [0:FIFO_DEPTH-1];
    
    // Con trỏ đọc và ghi
    reg [$clog2(FIFO_DEPTH):0] write_ptr;
    reg [$clog2(FIFO_DEPTH):0] read_ptr;
    
    // Số lượng phần tử trong FIFO
    wire [$clog2(FIFO_DEPTH):0] count = write_ptr - read_ptr;
    
    // Gán tín hiệu đầu ra
    always @(*) begin
        empty = (count == 0);
        full = (count == FIFO_DEPTH);
    end
    
    // Xử lý đọc và ghi
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_ptr <= 0;
            read_ptr <= 0;
            data_out <= 0;
        end else begin
            // Ghi dữ liệu
            if (write_en && !full) begin
                memory[write_ptr[$clog2(FIFO_DEPTH)-1:0]] <= data_in;
                write_ptr <= write_ptr + 1;
            end
            
            // Đọc dữ liệu
            if (read_en && !empty) begin
                data_out <= memory[read_ptr[$clog2(FIFO_DEPTH)-1:0]];
                read_ptr <= read_ptr + 1;
            end
        end
    end
    
endmodule
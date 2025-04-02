module line_buffer #(
    parameter IMG_WIDTH = 6,           // Chiều rộng ảnh
    parameter IMG_HEIGHT = 6,          // Chiều cao ảnh
    parameter CHANNELS = 3,            // Số kênh màu
    parameter KERNEL_SIZE = 3,         // Kích thước kernel
    parameter PADDING = 1,             // Padding (0: không padding, 1: padding 1 pixel,...)
    parameter DATA_WIDTH = 8           // Độ rộng bit dữ liệu
)(
    input wire clk,
    input wire rst_n,
    
    // Giao tiếp với FIFO đầu vào
    output reg fifo_read_en,
    input wire [DATA_WIDTH*CHANNELS-1:0] fifo_data,
    input wire fifo_empty,
    
    // Đầu ra cửa sổ
    output reg window_valid,
    output reg [DATA_WIDTH*KERNEL_SIZE*KERNEL_SIZE*CHANNELS-1:0] window_data,
    input wire window_ready
);
    
    // Kích thước ảnh sau khi padding
    localparam PADDED_WIDTH = IMG_WIDTH + 2*PADDING;
    localparam PADDED_HEIGHT = IMG_HEIGHT + 2*PADDING;
    
    // Các buffer cho từng kênh màu
    reg [DATA_WIDTH-1:0] buffer_r [0:PADDED_HEIGHT-1][0:PADDED_WIDTH-1];
    reg [DATA_WIDTH-1:0] buffer_g [0:PADDED_HEIGHT-1][0:PADDED_WIDTH-1];
    reg [DATA_WIDTH-1:0] buffer_b [0:PADDED_HEIGHT-1][0:PADDED_WIDTH-1];
    
    // Biến đếm và điều khiển
    reg [7:0] row_cnt, col_cnt;
    reg [7:0] window_row, window_col;
    reg loading_done;
    reg processed_last_window;  // Biến mới để theo dõi cửa sổ cuối cùng
    
    // Các trạng thái
    localparam IDLE = 2'd0;
    localparam LOAD = 2'd1;
    localparam PROCESS = 2'd2;
    localparam DONE = 2'd3;
    
    reg [1:0] state;
    
    // Tín hiệu đệm
    wire [DATA_WIDTH-1:0] data_b = fifo_data[3*DATA_WIDTH-1:2*DATA_WIDTH]; 
    wire [DATA_WIDTH-1:0] data_g = fifo_data[2*DATA_WIDTH-1:DATA_WIDTH];   
    wire [DATA_WIDTH-1:0] data_r = fifo_data[DATA_WIDTH-1:0];              
    
    integer i, j, k, m, n, ch;
    
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            state <= IDLE;
            fifo_read_en <= 0;
            window_valid <= 0;
            loading_done <= 0;
            row_cnt <= 0;
            col_cnt <= 0;
            window_row <= 0;
            window_col <= 0;
            processed_last_window <= 0;
            
            // Khởi tạo buffer với giá trị padding (0)
            for (i = 0; i < PADDED_HEIGHT; i = i + 1) begin
                for (j = 0; j < PADDED_WIDTH; j = j + 1) begin
                    buffer_r[i][j] <= 0;
                    buffer_g[i][j] <= 0;
                    buffer_b[i][j] <= 0;
                end
            end
        end else begin
            case (state)
                IDLE: begin
                    if (~fifo_empty) begin
                        state <= LOAD;
                        fifo_read_en <= 1;
                    end
                end
                
                LOAD: begin
                    if (~fifo_empty) begin
                        // Đặt fifo_read_en trước, đọc dữ liệu ở chu kỳ sau
                        fifo_read_en <= 1;

                        // Chỉ ghi vào buffer khi fifo_read_en đã là 1
                        if (fifo_read_en) begin
                            buffer_r[PADDING + row_cnt][PADDING + col_cnt] <= data_r;
                            buffer_g[PADDING + row_cnt][PADDING + col_cnt] <= data_g;
                            buffer_b[PADDING + row_cnt][PADDING + col_cnt] <= data_b;
                            
                            // Cập nhật con trỏ chỉ khi đã đọc dữ liệu
                            if (col_cnt == IMG_WIDTH - 1) begin
                                col_cnt <= 0;
                                if (row_cnt == IMG_HEIGHT - 1) begin
                                    row_cnt <= 0;
                                    loading_done <= 1;
                                    fifo_read_en <= 0;
                                    state <= PROCESS;
                                end else begin
                                    row_cnt <= row_cnt + 1;
                                end
                            end else begin
                                col_cnt <= col_cnt + 1;
                            end
                        end
                    end else begin
                        fifo_read_en <= 0;
                    end
                end
                
                PROCESS: begin
                    if (window_ready) begin
                        // Tạo cửa sổ trượt từ buffer
                        window_valid <= 1;
                        
                        // Đóng gói dữ liệu cửa sổ
                        for (k = 0; k < CHANNELS; k = k + 1) begin
                            for (m = 0; m < KERNEL_SIZE; m = m + 1) begin
                                for (n = 0; n < KERNEL_SIZE; n = n + 1) begin
                                    if (k == 0) begin
                                        window_data[(k*KERNEL_SIZE*KERNEL_SIZE + m*KERNEL_SIZE + n)*DATA_WIDTH +: DATA_WIDTH] 
                                            <= buffer_r[window_row + m][window_col + n];
                                    end else if (k == 1) begin
                                        window_data[(k*KERNEL_SIZE*KERNEL_SIZE + m*KERNEL_SIZE + n)*DATA_WIDTH +: DATA_WIDTH] 
                                            <= buffer_g[window_row + m][window_col + n];
                                    end else begin
                                        window_data[(k*KERNEL_SIZE*KERNEL_SIZE + m*KERNEL_SIZE + n)*DATA_WIDTH +: DATA_WIDTH] 
                                            <= buffer_b[window_row + m][window_col + n];
                                    end
                                end
                            end
                        end
                        
                        // Kiểm tra cửa sổ cuối cùng
                        if ((window_row == IMG_HEIGHT + 2*PADDING - KERNEL_SIZE) && 
                            (window_col == IMG_WIDTH + 2*PADDING - KERNEL_SIZE)) begin
                            // Đánh dấu đã xử lý cửa sổ cuối cùng
                            processed_last_window <= 1;
                        end
                        else begin
                            // Di chuyển cửa sổ tiếp theo
                            if (window_col == IMG_WIDTH + 2*PADDING - KERNEL_SIZE) begin
                                window_col <= 0;
                                window_row <= window_row + 1;
                            end else begin
                                window_col <= window_col + 1;
                            end
                        end
                    end else begin
                        window_valid <= 0;
                    end
                    
                    // Kiểm tra nếu đã xử lý xong cửa sổ cuối
                    if (processed_last_window && window_valid) begin
                        window_valid <= 0;
                        processed_last_window <= 0;
                        window_row <= 0;
                        window_col <= 0;
                        state <= DONE;
                    end
                end
                
                DONE: begin
                    // Chờ tín hiệu reset hoặc lệnh mới
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule
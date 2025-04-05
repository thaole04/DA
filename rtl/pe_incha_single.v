`timescale 1ns / 1ps

module pe_incha_single #(
    // Layer parameters
    parameter IN_WIDTH    = 3,
    parameter IN_HEIGHT   = 3,
    parameter IN_CHANNEL  = 2,
    parameter OUT_CHANNEL = 4,
    parameter OUTPUT_MODE = "relu",
    // parameter OUTPUT_MODE = "dequant",

    // Conv parameters
    parameter KERNEL_0   = 3,
    parameter KERNEL_1   = 3,
    parameter DILATION_0 = 1,
    parameter DILATION_1 = 1,
    parameter PADDING_0  = 1,
    parameter PADDING_1  = 1,
    parameter STRIDE_0   = 1,
    parameter STRIDE_1   = 1,
    
    // Địa chỉ định dạng parameters
    parameter ADDR_TYPE_WIDTH   = 8,
    parameter ADDR_CHANNEL_WIDTH = 8,
    parameter ADDR_POSITION_WIDTH = 8,
    parameter ADDR_RESERVED_WIDTH = 8,
    
    // Địa chỉ Type IDs
    parameter ADDR_TYPE_KERNEL = 8'h00,
    parameter ADDR_TYPE_BIAS   = 8'h01,
    
    // Vị trí field trong địa chỉ
    parameter ADDR_TYPE_MSB     = 31,
    parameter ADDR_TYPE_LSB     = 24,
    parameter ADDR_CHANNEL_MSB  = 23,
    parameter ADDR_CHANNEL_LSB  = 16,
    parameter ADDR_POSITION_MSB = 15,
    parameter ADDR_POSITION_LSB = 8,
    parameter ADDR_RESERVED_MSB = 7,
    parameter ADDR_RESERVED_LSB = 0
)(
    o_data,
    o_valid,
    pe_ready,
    pe_ack,
    i_data,
    i_valid,
    weight_wr_data,
    weight_wr_addr,
    weight_wr_en,
    clk,
    rst_n
);

    localparam KERNEL_PTS        = KERNEL_0 * KERNEL_1;
    localparam OUTPUT_DATA_WIDTH = OUTPUT_MODE == "relu" ? 8 : 16;
    localparam OUT_CHANNEL_BITS  = $clog2(OUT_CHANNEL);
    localparam MACC_OUTPUT_DATA_WIDTH = 16 + $clog2(KERNEL_PTS * IN_CHANNEL);

    output [OUTPUT_DATA_WIDTH*OUT_CHANNEL-1:0] o_data;
    output                                     o_valid;
    output                                     pe_ready;
    output                                     pe_ack;
    input [8*IN_CHANNEL*KERNEL_PTS-1:0]        i_data;
    input                                      i_valid;
    input [15:0]                               weight_wr_data;
    input [31:0]                               weight_wr_addr;
    input                                      weight_wr_en;
    input                                      clk;
    input                                      rst_n;

    // Extract address fields
    wire [ADDR_TYPE_WIDTH-1:0]     addr_type;
    wire [ADDR_CHANNEL_WIDTH-1:0]  addr_channel;
    wire [ADDR_POSITION_WIDTH-1:0] addr_position;
    
    assign addr_type     = weight_wr_addr[ADDR_TYPE_MSB:ADDR_TYPE_LSB];
    assign addr_channel  = weight_wr_addr[ADDR_CHANNEL_MSB:ADDR_CHANNEL_LSB];
    assign addr_position = weight_wr_addr[ADDR_POSITION_MSB:ADDR_POSITION_LSB];
    
    // Controller
    wire cnt_en;
    wire cnt_limit;

    pe_controller u_control (
        .cnt_en    (cnt_en),
        .pe_ready  (pe_ready),
        .pe_ack    (pe_ack),
        .cnt_limit (cnt_limit),
        .i_valid   (i_valid),
        .clk       (clk),
        .rst_n     (rst_n)
    );

    // Input registers
    reg [8*IN_CHANNEL*KERNEL_PTS-1:0] i_data_reg;

    always @ (posedge clk) begin
        if (pe_ack) begin
            i_data_reg <= i_data;
        end
    end

    // Kernel ram
    wire [8*IN_CHANNEL*KERNEL_PTS-1:0] kernel;
    reg  [OUT_CHANNEL_BITS-1:0]        kernel_cnt;
    wire                               is_kernel_write;
    
    assign is_kernel_write = weight_wr_en & (addr_type == ADDR_TYPE_KERNEL);
    assign cnt_limit = kernel_cnt == OUT_CHANNEL - 1;

    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            kernel_cnt <= 0;
        end
        else if (cnt_en) begin
            kernel_cnt <= cnt_limit ? 0 : kernel_cnt + 1;
        end
    end

    reg [KERNEL_PTS*IN_CHANNEL-1:0] kernel_wr_en;
    integer j;

    // Tạo one-hot encoding cho kernel_wr_en - bật đúng bit tương ứng với addr_position
    always @(*) begin
        // Khởi tạo tất cả bit bằng 0
        for (j = 0; j < KERNEL_PTS*IN_CHANNEL; j = j + 1) begin
            kernel_wr_en[j] = 1'b0;
        end
        
        // Nếu là ghi kernel và addr_position hợp lệ, bật bit tương ứng
        if (is_kernel_write && addr_position < KERNEL_PTS*IN_CHANNEL) begin
            kernel_wr_en[addr_position] = 1'b1;
        end
    end

    block_ram_multi_word #(
        .DATA_WIDTH      (8),
        .DEPTH           (OUT_CHANNEL),
        .NUM_WORDS       (KERNEL_PTS * IN_CHANNEL),
        .RAM_STYLE       ("auto"),
        .OUTPUT_REGISTER ("true")
    ) u_kernel (
        .rd_data         (kernel),
        .wr_data         (weight_wr_data[7:0]),
        .rd_addr         (kernel_cnt),
        .wr_addr         (addr_channel[OUT_CHANNEL_BITS-1:0]),
        .wr_en           (kernel_wr_en),  // Sử dụng tín hiệu one-hot được tạo
        .rd_en           (1'b1),
        .clk             (clk)
    );

    // Bias ram
    wire signed [15:0]           bias;
    reg  [OUT_CHANNEL_BITS-1:0]  bias_cnt;
    wire                         bias_cnt_en;
    wire                         bias_cnt_limit;
    wire                         is_bias_write;
    
    assign bias_cnt_limit = bias_cnt == OUT_CHANNEL - 1;
    assign is_bias_write = weight_wr_en & (addr_type == ADDR_TYPE_BIAS);

    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            bias_cnt <= 0;
        end
        else if (bias_cnt_en) begin
            bias_cnt <= bias_cnt_limit ? 0 : bias_cnt + 1;
        end
    end

    block_ram_single_port #(
        .DATA_WIDTH      (16),
        .DEPTH           (OUT_CHANNEL),
        .RAM_STYLE       ("auto"),
        .OUTPUT_REGISTER ("true")
    ) u_bias (
        .rd_data         (bias),
        .wr_data         (weight_wr_data),
        .wr_addr         (addr_channel[OUT_CHANNEL_BITS-1:0]),
        .rd_addr         (bias_cnt),
        .wr_en           (is_bias_write),
        .rd_en           (1'b1),
        .clk             (clk)
    );

    // MACC co-efficient reg
    reg signed [15:0] macc_coeff = 1;

    // MACC
    wire [MACC_OUTPUT_DATA_WIDTH-1:0] macc_data_out;
    wire                              macc_valid_o;
    reg                               macc_valid_i;

    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            macc_valid_i <= 1'b0;
        end
        else begin
            macc_valid_i <= cnt_en;
        end
    end

    // BRAM output pipeline register
    reg [8*IN_CHANNEL*KERNEL_PTS-1:0] i_data_reg_pipeline;
    reg                               macc_valid_i_pipeline;

    always @ (posedge clk) begin
        i_data_reg_pipeline <= i_data_reg;
    end

    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            macc_valid_i_pipeline <= 1'b0;
        end
        else begin
            macc_valid_i_pipeline <= macc_valid_i;
        end
    end

    // MACC module
    macc_8bit_single #(
        .NUM_INPUTS (KERNEL_PTS * IN_CHANNEL)
    ) u_macc_single (
        .o_data     (macc_data_out),
        .o_valid    (macc_valid_o),
        .i_data_a   (kernel),
        .i_data_b   (i_data_reg_pipeline),
        .i_valid    (macc_valid_i_pipeline),
        .clk        (clk),
        .rst_n      (rst_n)
    );

    // MACC out reg
    reg signed [MACC_OUTPUT_DATA_WIDTH-1:0] macc_data_out_reg;
    reg                                     macc_valid_o_reg;

    always @ (posedge clk) begin
        if (macc_valid_o) begin
            macc_data_out_reg <= macc_data_out;
        end
    end

    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            macc_valid_o_reg <= 1'b0;
        end
        else begin
            macc_valid_o_reg <= macc_valid_o;
        end
    end

    // MACC co-efficient
    localparam COEFF_RESULT_WIDTH = MACC_OUTPUT_DATA_WIDTH + 16;
    
    reg signed [COEFF_RESULT_WIDTH-1:0] coeff_prod;
    reg                                 coeff_valid;

    always @ (posedge clk) begin
        if (macc_valid_o_reg) begin
            coeff_prod <= macc_coeff * macc_data_out_reg;
        end
    end

    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            coeff_valid <= 1'b0;
        end
        else begin
            coeff_valid <= macc_valid_o_reg;
        end
    end

    // Bias
    localparam BIAS_ADJUSTED_WIDTH = 24;
    
    wire signed [BIAS_ADJUSTED_WIDTH-1:0]  bias_adjusted;
    reg  signed [COEFF_RESULT_WIDTH-1:0]   bias_sum;
    reg                                    bias_valid;
    
    assign bias_adjusted = {bias, {8{1'b0}}};
    assign bias_cnt_en = macc_valid_o;

    always @ (posedge clk) begin
        if (coeff_valid) begin
            bias_sum <= coeff_prod + bias_adjusted;
        end
    end

    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            bias_valid <= 1'b0;
        end
        else begin
            bias_valid <= coeff_valid;
        end
    end

    // Output
    wire signed [OUTPUT_DATA_WIDTH-1:0] obuffer_data;
    wire                                obuffer_valid;

    generate
        if (OUTPUT_MODE == "relu") begin : gen_relu
            // Constants for ReLU activation
            localparam RELU_MAX_VALUE = 127;
            localparam RELU_MIN_VALUE = 0;
            
            // Các bit cần kiểm tra cho overflow
            wire overflow_positive = bias_sum[23] || bias_sum[22:16] == {7{1'b1}};
            // Bit làm tròn
            wire rounding_bit = bias_sum[15] & |bias_sum[14:12];
            
            // ReLU với saturation và rounding
            assign obuffer_data = bias_sum < 0 ? RELU_MIN_VALUE : 
                                 (overflow_positive ? RELU_MAX_VALUE : 
                                 (bias_sum[23:16] + rounding_bit));
            assign obuffer_valid = bias_valid;
        end
    endgenerate

    // Output buffer
    pe_incha_obuffer #(
        .DATA_WIDTH  (OUTPUT_DATA_WIDTH),
        .NUM_INPUTS  (1),
        .OUT_CHANNEL (OUT_CHANNEL)
    ) u_obuffer (
        .o_data      (o_data),
        .o_valid     (o_valid),
        .i_data      (obuffer_data),
        .i_valid     (obuffer_valid),
        .clk         (clk),
        .rst_n       (rst_n)
    );

endmodule
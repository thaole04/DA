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
    parameter STRIDE_1   = 1
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
    reg  [$clog2(OUT_CHANNEL)-1:0]     kernel_cnt;

    assign cnt_limit = kernel_cnt == OUT_CHANNEL - 1;

    always @ (posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            kernel_cnt <= 0;
        end
        else if (cnt_en) begin
            kernel_cnt <= cnt_limit ? 0 : kernel_cnt + 1;
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
        .wr_data         (1'b1),
        .rd_addr         (kernel_cnt),
        .wr_addr         (1'b1),
        .wr_en           (kernel_ram_wr_en),
        .rd_en           (1'b1),
        .clk             (clk)
    );

    // Bias ram
    wire signed [15:0] bias;

    reg  [$clog2(OUT_CHANNEL)-1:0] bias_cnt;
    wire                           bias_cnt_en;
    wire                           bias_cnt_limit = bias_cnt == OUT_CHANNEL - 1;

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
        .wr_addr         (bias_wr_addr),
        .rd_addr         (bias_cnt),
        .wr_en           (bias_wr_en),
        .rd_en           (1'b1),
        .clk             (clk)
    );

    // MACC co-efficient reg
    reg signed [15:0] macc_coeff = 16'h0080;

    // MACC
    localparam MACC_OUTPUT_DATA_WIDTH = 16 + $clog2(KERNEL_PTS * IN_CHANNEL);

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

    ////////////////////////////////////////////////////////////////////////////////////////////////

    // BRAM output pipeline register because Vivado won't stop screaming about it
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

    ////////////////////////////////////////////////////////////////////////////////////////////////

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
    reg signed [MACC_OUTPUT_DATA_WIDTH+16-1:0] coeff_prod;
    reg                                        coeff_valid;

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
    wire signed [23:0]                          bias_adjusted = {bias, {8{1'b0}}};
    reg  signed [MACC_OUTPUT_DATA_WIDTH+16-1:0] bias_sum;
    reg                                         bias_valid;

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
        if (OUTPUT_MODE == "relu") begin : gen0
            assign obuffer_data = bias_sum < 0 ? 0 : ((bias_sum[23] || bias_sum[22:16] == {7{1'b1}}) ? 127 : (bias_sum[23:16] + (bias_sum[15] & |bias_sum[14:12])));
            assign obuffer_valid  = bias_valid;
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

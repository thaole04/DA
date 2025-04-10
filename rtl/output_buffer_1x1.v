`timescale 1ns / 1ps

module output_buffer_1x1 #(
    parameter DATA_WIDTH      = 8,
    parameter OUT_CHANNELS    = 3,
    parameter IN_WIDTH        = 5,
    parameter IN_HEIGHT       = 5,
    parameter DEPTH           = IN_WIDTH * IN_HEIGHT * OUT_CHANNELS,
    parameter RAM_STYLE       = "auto",
    parameter OUTPUT_REGISTER = "true"
)(
    output [DATA_WIDTH*OUT_CHANNELS-1:0]                rd_data,
    input  [$clog2(IN_WIDTH*IN_HEIGHT)-1:0]             rd_addr,
    input                                               rd_en,
    input  [DATA_WIDTH-1:0]                             wr_data,
    input  [$clog2(DEPTH)-1:0]                          wr_addr,
    input                                               wr_en,
    input                                               clk
);
    (* ram_style = RAM_STYLE *) reg [DATA_WIDTH-1:0] ram [0:DEPTH-1];

    integer i;

    always @ (posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end

    reg [DATA_WIDTH*OUT_CHANNELS-1:0] rd_data_reg;

    always @ (posedge clk) begin
        if (rd_en) begin
            for (i = 0; i < OUT_CHANNELS; i = i + 1) begin
                rd_data_reg[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= ram[rd_addr*OUT_CHANNELS+i];
            end
        end
    end
    assign rd_data = rd_data_reg;

endmodule
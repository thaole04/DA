`timescale 1ns / 1ps

module bram_weight_1x1 #(
    parameter DATA_WIDTH      = 8,
    parameter IN_CHANNELS     = 3,
    parameter OUT_CHANNELS    = 5,
    parameter DEPTH           = IN_CHANNELS * OUT_CHANNELS, // in_channels*out_channels = 3*5 = 15
    parameter RAM_STYLE       = "auto",
    parameter OUTPUT_REGISTER = "false"
)(
    output [DATA_WIDTH*IN_CHANNELS-1:0]    rd_data,
    input  [$clog2(OUT_CHANNELS)-1:0]      rd_addr, // Address for reading data, 0 for in_channels 1st, 1 for in_channels 2nd, etc.
    input                                  rd_en,
    input  [DATA_WIDTH-1:0]                wr_data,
    input  [$clog2(DEPTH)-1:0]             wr_addr,
    input                                  wr_en,
    input                                  clk
);
    // |In_channels*DATA_WIDTH|In_channels*DATA_WIDTH|In_channels*DATA_WIDTH|In_channels*DATA_WIDTH|In_channels*DATA_WIDTH|
    (* ram_style = RAM_STYLE *) reg [DATA_WIDTH-1:0] ram [0:IN_CHANNELS*OUT_CHANNELS-1];
    
    integer i;

    // Write port, write data to the RAM for the specified address
    always @ (posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end

    // Read port
    generate
        if (OUTPUT_REGISTER == "true") begin
            reg [DATA_WIDTH*IN_CHANNELS-1:0] rd_data_reg;
            always @ (posedge clk) begin
                if (rd_en) begin
                    for (i = 0; i < IN_CHANNELS; i = i + 1) begin
                        rd_data_reg[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] <= ram[rd_addr*IN_CHANNELS + i];
                    end
                end
            end
            assign rd_data = rd_data_reg;
        end else begin
            reg [DATA_WIDTH*IN_CHANNELS-1:0] rd_data_comb;
            always @ (*) begin
                rd_data_comb = 0;
                if (rd_en) begin
                    for (i = 0; i < IN_CHANNELS; i = i + 1) begin
                        rd_data_comb[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = ram[rd_addr*IN_CHANNELS + i];
                    end
                end
            end
            assign rd_data = rd_data_comb;
        end
    endgenerate

endmodule
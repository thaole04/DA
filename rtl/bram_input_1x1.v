`timescale 1ns / 1ps

module bram_input_1x1 #(
    parameter DATA_WIDTH      = 8,
    parameter IN_CHANNELS     = 3,
    parameter IN_WIDTH        = 5,
    parameter IN_HEIGHT       = 5,
    parameter DEPTH           = IN_WIDTH * IN_HEIGHT * IN_CHANNELS, // width*height*in_channels = 5*5*3 = 75
    parameter RAM_STYLE       = "auto",
    parameter OUTPUT_REGISTER = "false"
)(
    output [DATA_WIDTH*IN_CHANNELS-1:0]          rd_data,
    input  [$clog2(IN_WIDTH*IN_HEIGHT)-1:0]      rd_addr, // Position in feature map (x,y coordinate flattened)
    input                                        rd_en,
    input  [DATA_WIDTH-1:0]                      wr_data,
    input  [$clog2(DEPTH)-1:0]                   wr_addr,
    input                                        wr_en,
    input                                        clk
);
    // Format: |channel0 pixel0|channel1 pixel0|channel2 pixel0|...|channel0 pixel1|channel1 pixel1|...
    (* ram_style = RAM_STYLE *) reg [DATA_WIDTH-1:0] ram [0:IN_CHANNELS*IN_WIDTH*IN_HEIGHT-1];
    
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
                        // For each position in feature map, read all channels
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
                        // For each position in feature map, read all channels
                        rd_data_comb[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = ram[rd_addr*IN_CHANNELS + i];
                    end
                end
            end
            assign rd_data = rd_data_comb;
        end
    endgenerate

endmodule
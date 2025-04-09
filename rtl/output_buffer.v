`timescale 1ns / 1ps

module output_buffer #(
    parameter DATA_WIDTH      = 8,
    parameter OUT_CHANNELS    = 3,
    parameter IN_WIDTH        = 5,
    parameter IN_HEIGHT       = 5,
    parameter DEPTH           = IN_WIDTH * IN_HEIGHT * OUT_CHANNELS, // width*height*OUT_CHANNELS = 5*5*3 = 75
    parameter RAM_STYLE       = "auto",
    parameter OUTPUT_REGISTER = "false"
)(
    output [DATA_WIDTH-1:0]                                 rd_data,
    input  [$clog2(DEPTH)-1:0]                              rd_addr,
    input                                                   rd_en,
    input  [DATA_WIDTH-1:0]                                 wr_data,
    input  [$clog2(DEPTH)-1:0]                              wr_addr,
    input                                                   wr_en,
    input                                                   clk
);
    (* ram_style = RAM_STYLE *) reg [DATA_WIDTH-1:0] ram [0:IN_WIDTH*IN_HEIGHT*OUT_CHANNELS-1];
    // Write port, write data to the RAM for the specified address
    always @ (posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end

    // Read port
    generate
        if (OUTPUT_REGISTER == "true") begin
            reg [DATA_WIDTH-1:0] rd_data_reg;
            always @ (posedge clk) begin
                if (rd_en) begin
                    rd_data_reg <= ram[rd_addr];
                end
            end
            assign rd_data = rd_data_reg;
        end else begin
            reg [DATA_WIDTH-1:0] rd_data_comb;
            always @ (*) begin
                rd_data_comb = 0;
                if (rd_en) begin
                    rd_data_comb = ram[rd_addr];
                end
            end
            assign rd_data = rd_data_comb;
        end
    endgenerate

endmodule
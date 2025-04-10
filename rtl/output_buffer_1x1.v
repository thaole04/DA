`timescale 1ns / 1ps

module output_buffer_1x1 #(
    parameter DATA_WIDTH      = 8,
    parameter OUT_CHANNELS    = 3,
    parameter IN_WIDTH        = 5,
    parameter IN_HEIGHT       = 5,
    parameter DEPTH           = IN_WIDTH * IN_HEIGHT * OUT_CHANNELS, // 5*5*3 = 75
    parameter RAM_STYLE       = "auto"
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

    always @ (posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= wr_data;
        end
    end


    reg [DATA_WIDTH*OUT_CHANNELS-1:0] rd_data_reg;
    wire [DATA_WIDTH*OUT_CHANNELS-1:0] rd_data_next;

    genvar ch;
    for (ch = 0; ch < OUT_CHANNELS; ch = ch + 1) begin : CHAN_READ_GEN

        wire [$clog2(DEPTH)-1:0] current_ram_addr = rd_addr * OUT_CHANNELS + ch;

        localparam integer base_bit_index = ch * DATA_WIDTH;
        localparam integer high_bit_index = base_bit_index + DATA_WIDTH - 1;

        wire [DATA_WIDTH-1:0] ram_read_value;
        assign ram_read_value = (current_ram_addr >= 0 && current_ram_addr < DEPTH) ?
                                ram[current_ram_addr] : {DATA_WIDTH{1'b0}};

        assign rd_data_next[high_bit_index : base_bit_index] = rd_en ? ram_read_value : rd_data_reg[high_bit_index : base_bit_index];

    end // CHAN_READ_GEN

    // Register the next value on clock edge
    always @(posedge clk) begin
        rd_data_reg <= rd_data_next;
    end

    // Assign the registered output
    assign rd_data = rd_data_reg;

endmodule
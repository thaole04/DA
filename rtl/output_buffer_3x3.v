`timescale 1ns / 1ps

module output_buffer_3x3 #(
    parameter DATA_WIDTH      = 8,
    parameter OUT_CHANNELS    = 3,
    parameter IN_WIDTH        = 5,
    parameter IN_HEIGHT       = 5,
    parameter PAD_WIDTH       = IN_WIDTH + 2,
    parameter PAD_HEIGHT      = IN_HEIGHT + 2,
    parameter DEPTH           = PAD_WIDTH * PAD_HEIGHT * OUT_CHANNELS,
    parameter RAM_STYLE       = "auto"
)(
    output [9*DATA_WIDTH*OUT_CHANNELS-1:0]              rd_data,
    input  [$clog2(IN_WIDTH*IN_HEIGHT)-1:0]             rd_addr,
    input                                               rd_en,
    input  [DATA_WIDTH-1:0]                             wr_data,
    input  [$clog2(DEPTH)-1:0]                          wr_addr,
    input                                               is_padding,
    input                                               wr_en,
    input                                               clk
);

    (* ram_style = RAM_STYLE *) reg [DATA_WIDTH-1:0] ram [0:DEPTH-1];

    always @ (posedge clk) begin
        if (wr_en) begin
            ram[wr_addr] <= is_padding ? {DATA_WIDTH{1'b0}} : wr_data;
        end
    end


    wire [$clog2(IN_HEIGHT)-1:0] center_r_orig = rd_addr / IN_WIDTH;
    wire [$clog2(IN_WIDTH)-1:0]  center_c_orig = rd_addr % IN_WIDTH;
    wire [$clog2(PAD_HEIGHT)-1:0] center_r_pad = center_r_orig + 1;
    wire [$clog2(PAD_WIDTH)-1:0]  center_c_pad = center_c_orig + 1;

    reg [9*DATA_WIDTH*OUT_CHANNELS-1:0] rd_data_reg;
    wire [9*DATA_WIDTH*OUT_CHANNELS-1:0] rd_data_next;

    genvar r_offset_idx, c_offset_idx, ch;
    for (r_offset_idx = 0; r_offset_idx < 3; r_offset_idx = r_offset_idx + 1) begin : ROW_GEN
        for (c_offset_idx = 0; c_offset_idx < 3; c_offset_idx = c_offset_idx + 1) begin : COL_GEN
            // Calculate current pixel's coordinates in padded space
            wire [$clog2(PAD_HEIGHT)-1:0] current_r_pad = center_r_pad + r_offset_idx - 1;
            wire [$clog2(PAD_WIDTH)-1:0]  current_c_pad = center_c_pad + c_offset_idx - 1;

            // Calculate flattened pixel index within the 3x3 window (0 to 8)
            localparam integer pixel_3x3_idx = r_offset_idx * 3 + c_offset_idx;

            // Calculate flattened pixel index in the *padded* space
            wire [$clog2(PAD_WIDTH*PAD_HEIGHT)-1:0] flat_pixel_idx_pad = current_r_pad * PAD_WIDTH + current_c_pad;


            for (ch = 0; ch < OUT_CHANNELS; ch = ch + 1) begin : CHAN_GEN
                wire [$clog2(DEPTH)-1:0] current_ram_addr = flat_pixel_idx_pad * OUT_CHANNELS + ch;

                localparam integer base_bit_index = (pixel_3x3_idx * OUT_CHANNELS + ch) * DATA_WIDTH;
                localparam integer high_bit_index = base_bit_index + DATA_WIDTH - 1;


                wire [DATA_WIDTH-1:0] ram_read_value;
                assign ram_read_value = (current_ram_addr >= 0 && current_ram_addr < DEPTH) ?
                                        ram[current_ram_addr] : {DATA_WIDTH{1'b0}};

                // Assign combinatorially to the intermediate wire based on rd_en
                assign rd_data_next[high_bit_index : base_bit_index] = rd_en ? ram_read_value : rd_data_reg[high_bit_index : base_bit_index];

            end // CHAN_GEN
        end // COL_GEN
    end // ROW_GEN

    always @(posedge clk) begin
        rd_data_reg <= rd_data_next;
    end

    // Assign the registered output
    assign rd_data = rd_data_reg;

endmodule
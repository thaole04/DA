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
    // --- ---
    input                                               clk
);

    (* ram_style = RAM_STYLE *) reg [DATA_WIDTH-1:0] ram [0:DEPTH-1];

    always @ (posedge clk) begin
        if (wr_en) begin
            if (is_padding) begin
                ram[wr_addr] <= {DATA_WIDTH{1'b0}}; // Write zero for padding
            end else begin
                ram[wr_addr] <= wr_data;
            end
        end
    end


    integer ro, co, current_r_pad, current_c_pad, pixel_3x3_idx, flat_pixel_idx_pad, c, current_ram_addr, base_bit_index;
    wire [$clog2(IN_HEIGHT)-1:0] center_r_orig = rd_addr / IN_WIDTH;
    wire [$clog2(IN_WIDTH)-1:0]  center_c_orig = rd_addr % IN_WIDTH;
    wire [$clog2(PAD_HEIGHT)-1:0] center_r_pad = center_r_orig + 1; // Add padding offset
    wire [$clog2(PAD_WIDTH)-1:0]  center_c_pad = center_c_orig + 1; // Add padding offset

    reg [9*DATA_WIDTH*OUT_CHANNELS-1:0] rd_data_reg;

    
    always @ (posedge clk) begin
        if (rd_en) begin
            // Loop through 3x3 window offsets (-1, 0, 1)
            for (ro = -1; ro <= 1; ro = ro + 1) begin
                for (co = -1; co <= 1; co = co + 1) begin
                    current_r_pad = center_r_pad + ro;
                    current_c_pad = center_c_pad + co;

                    // Calculate flattened pixel index within the 3x3 window (0 to 8, row-major)
                    pixel_3x3_idx = (ro + 1) * 3 + (co + 1);

                    // Calculate flattened pixel index in the *padded* space
                    flat_pixel_idx_pad = current_r_pad * PAD_WIDTH + current_c_pad;

                    for (c = 0; c < OUT_CHANNELS; c = c + 1) begin
                        current_ram_addr = flat_pixel_idx_pad * OUT_CHANNELS + c;

                        // Output format: [pix8_chN..ch0, pix7_chN..ch0, ..., pix0_chN..ch0]
                        base_bit_index = (pixel_3x3_idx * OUT_CHANNELS + c) * DATA_WIDTH;

                        rd_data_reg[base_bit_index + DATA_WIDTH - 1 : base_bit_index] <= ram[current_ram_addr];
                    end
                end
            end
        end
    end

    // Assign the registered output
    assign rd_data = rd_data_reg;

endmodule
`timescale 1ns / 1ps

module testbench_output_buffer_3x3_simple();

    // --- Parameters matching the DUT ---
    localparam DATA_WIDTH   = 8;
    localparam OUT_CHANNELS = 3;
    localparam IN_WIDTH     = 5;
    localparam IN_HEIGHT    = 5;

    // --- Calculated Parameters ---
    localparam PAD_WIDTH    = IN_WIDTH + 2;     // 7
    localparam PAD_HEIGHT   = IN_HEIGHT + 2;    // 7
    localparam DEPTH        = PAD_WIDTH * PAD_HEIGHT * OUT_CHANNELS; // 147
    localparam RD_ADDR_WIDTH= $clog2(IN_WIDTH*IN_HEIGHT); // 5
    localparam WR_ADDR_WIDTH= $clog2(DEPTH);           // 8
    localparam RD_DATA_WIDTH= 9 * DATA_WIDTH * OUT_CHANNELS; // 216

    // --- Testbench Signals ---
    reg                             clk;
    reg                             wr_en;
    reg                             is_padding; // DUT needs this, but we might simplify writing
    reg [DATA_WIDTH-1:0]            wr_data;
    reg [WR_ADDR_WIDTH-1:0]         wr_addr;
    reg                             rd_en;
    reg [RD_ADDR_WIDTH-1:0]         rd_addr;
    wire [RD_DATA_WIDTH-1:0]        rd_data;

    // --- Instantiate the DUT ---
    output_buffer_3x3 #(
        .DATA_WIDTH      (DATA_WIDTH),
        .OUT_CHANNELS    (OUT_CHANNELS),
        .IN_WIDTH        (IN_WIDTH),
        .IN_HEIGHT       (IN_HEIGHT)
    ) uut (
        .rd_data         (rd_data),
        .rd_addr         (rd_addr),
        .rd_en           (rd_en),
        .wr_data         (wr_data),
        .wr_addr         (wr_addr),
        .is_padding      (is_padding),
        .wr_en           (wr_en),
        .clk             (clk)
    );

    // --- Clock Generation ---
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz clock
    end

    // --- Stimulus ---
    initial begin
        integer r_pad, c_pad, ch; // Loop variables for writing RAM

        $display("Starting Simple Testbench for output_buffer_3x3");

        // 1. Initialize signals
        wr_en      = 0;
        is_padding = 0;
        wr_data    = 0;
        wr_addr    = 0;
        rd_en      = 0;
        rd_addr    = 0;
        repeat(2) @(posedge clk); // Wait a couple of cycles

        // 2. Write a predictable pattern to RAM
        // We only need to write the area relevant for reading rd_addr=6
        // which corresponds to original (1,1) -> padded (2,2) center
        // This read accesses padded rows 1..3 and padded cols 1..3
        $display("--- Writing Test Pattern to RAM ---");
        wr_en = 1;
        is_padding = 0; // Assume we are writing valid data for this test
        for (r_pad = 0; r_pad < PAD_HEIGHT; r_pad = r_pad + 1) begin
             for (c_pad = 0; c_pad < PAD_WIDTH; c_pad = c_pad + 1) begin
                 for (ch = 0; ch < OUT_CHANNELS; ch = ch + 1) begin
                     wr_addr = (r_pad * PAD_WIDTH + c_pad) * OUT_CHANNELS + ch;
                     // Pattern: Row major, Padded Coords, Channel specific offset
                     // Example: Data = (padded_row * 100) + (padded_col * 10) + channel
                     wr_data = ((r_pad * 100) + (c_pad * 10) + ch) % 256; // Modulo to fit 8 bits
                     @(posedge clk);
                     // Optional display for debug:
                     // $display("WRITE: Addr=%3d, Padded(%d,%d), Ch=%d, Data=0x%h",
                     //          wr_addr, r_pad, c_pad, ch, wr_data);
                 end
             end
        end
        wr_en = 0;
        @(posedge clk); // Wait one cycle after last write

        // 3. Perform a single read operation
        // Read the 3x3 window centered at original pixel (1,1) -> rd_addr = 6
        $display("--- Performing Read at rd_addr = 6 ---");
        rd_addr = RD_ADDR_WIDTH'(6); // Center pixel (original 1,1 -> padded 2,2)
        rd_en = 1;
        @(posedge clk); // Assert rd_en for one cycle
        rd_en = 0;

        // 4. Wait for the registered output (1 clock cycle delay)
        @(posedge clk);

        // 5. Display the result
        $display("T=%0t: READ DONE. rd_addr=6", $time);
        $display("  rd_data (total %0d bits) = %h", RD_DATA_WIDTH, rd_data);

        // Output format is: [pix8_chN..ch0, pix7_chN..ch0, ..., pix0_chN..ch0]
        // pix0 = padded(1,1); pix1=padded(1,2); pix2=padded(1,3)
        // pix3 = padded(2,1); pix4=padded(2,2); pix5=padded(2,3)
        // pix6 = padded(3,1); pix7=padded(3,2); pix8=padded(3,3)

        $display("  Breakdown (Hex - Ch0 Ch1 Ch2):");
        // Top-Left Pixel (pix0), Padded Coords (1,1) -> Expected Data: 110, 111, 112
        $display("    Pixel 0 (Padded 1,1): %h %h %h",
                 rd_data[0*DATA_WIDTH +: DATA_WIDTH],
                 rd_data[1*DATA_WIDTH +: DATA_WIDTH],
                 rd_data[2*DATA_WIDTH +: DATA_WIDTH]); // Ch0=0x6E, Ch1=0x6F, Ch2=0x70

        // Center Pixel (pix4), Padded Coords (2,2) -> Expected Data: 220, 221, 222
        $display("    Pixel 4 (Padded 2,2): %h %h %h",
                 rd_data[(4*OUT_CHANNELS+0)*DATA_WIDTH +: DATA_WIDTH],
                 rd_data[(4*OUT_CHANNELS+1)*DATA_WIDTH +: DATA_WIDTH],
                 rd_data[(4*OUT_CHANNELS+2)*DATA_WIDTH +: DATA_WIDTH]); // Ch0=0xDC, Ch1=0xDD, Ch2=0xDE

        // Bottom-Right Pixel (pix8), Padded Coords (3,3) -> Expected Data: 330, 331, 332 -> mod 256 -> 74, 75, 76
        $display("    Pixel 8 (Padded 3,3): %h %h %h",
                 rd_data[(8*OUT_CHANNELS+0)*DATA_WIDTH +: DATA_WIDTH],
                 rd_data[(8*OUT_CHANNELS+1)*DATA_WIDTH +: DATA_WIDTH],
                 rd_data[(8*OUT_CHANNELS+2)*DATA_WIDTH +: DATA_WIDTH]); // Ch0=0x4A, Ch1=0x4B, Ch2=0x4C

        $display("--- Testbench Finished ---");
        $finish;
    end

endmodule
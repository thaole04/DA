`timescale 1ns / 1ps

module testbench_output_buffer_1x1_simple();

    // --- Parameters matching the DUT ---
    localparam DATA_WIDTH   = 8;
    localparam OUT_CHANNELS = 3;
    localparam IN_WIDTH     = 5;
    localparam IN_HEIGHT    = 5;

    // --- Calculated Parameters ---
    localparam DEPTH        = IN_WIDTH * IN_HEIGHT * OUT_CHANNELS; // 75
    localparam RD_ADDR_WIDTH= $clog2(IN_WIDTH*IN_HEIGHT); // 5
    localparam WR_ADDR_WIDTH= $clog2(DEPTH);           // 8
    localparam RD_DATA_WIDTH= DATA_WIDTH * OUT_CHANNELS;      // 8*3 = 24

    // --- Testbench Signals ---
    reg                             clk;
    reg                             wr_en;
    reg [DATA_WIDTH-1:0]            wr_data;
    reg [WR_ADDR_WIDTH-1:0]         wr_addr;
    reg                             rd_en;
    reg [RD_ADDR_WIDTH-1:0]         rd_addr;
    wire [RD_DATA_WIDTH-1:0]        rd_data;

    // --- Instantiate the DUT ---
    output_buffer_1x1 #(
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
        integer pixel_idx, ch; // Loop variables for writing RAM

        $display("Starting Simple Testbench for output_buffer_1x1");

        // 1. Initialize signals & Wait
        wr_en      = 0;
        wr_data    = 0;
        wr_addr    = 0;
        rd_en      = 0;
        rd_addr    = 0;
        repeat(2) @(posedge clk);

        // 2. Write
        $display("--- Writing Test Pattern to RAM ---");
        wr_en = 1;
        for (pixel_idx = 0; pixel_idx < IN_WIDTH*IN_HEIGHT; pixel_idx = pixel_idx + 1) begin
            if (pixel_idx < 10) begin
                for (ch = 0; ch < OUT_CHANNELS; ch = ch + 1) begin
                    wr_addr = pixel_idx * OUT_CHANNELS + ch;
                    // Pattern: Pixel index * 10 + channel index
                    wr_data = (pixel_idx * 10 + ch) % 256;
                    @(posedge clk);
                    $display("WRITE: Addr=%3d, Pixel=%d, Ch=%d, Data=0x%h",
                             wr_addr, pixel_idx, ch, wr_data);
                end
            end
        end
        wr_en = 0;
        @(posedge clk); // Wait one cycle after last write

        // 3. Perform a single read operation for pixel 6
        $display("--- Performing Read at rd_addr = 6 ---");
        rd_addr = RD_ADDR_WIDTH'(6); // Pixel index 6
        rd_en = 1;
        @(posedge clk); // Assert rd_en for one cycle
        rd_en = 0;

        // 4. Wait for the registered output (1 clock cycle delay)
        @(posedge clk);

        // 5. Display the result
        $display("T=%0t: READ DONE. rd_addr=6", $time);
        $display("  rd_data (total %0d bits) = %h", RD_DATA_WIDTH, rd_data);

        // Breakdown for easier checking (Expected data: 60, 61, 62 dec -> 3C, 3D, 3E hex)
        $display("  Breakdown (Hex - Ch0 Ch1 Ch2): %h %h %h",
                 rd_data[0*DATA_WIDTH +: DATA_WIDTH],    // Channel 0 data for pixel 6
                 rd_data[1*DATA_WIDTH +: DATA_WIDTH],    // Channel 1 data for pixel 6
                 rd_data[2*DATA_WIDTH +: DATA_WIDTH]);   // Channel 2 data for pixel 6

        $display("--- Testbench Finished ---");
        $finish;
    end

endmodule
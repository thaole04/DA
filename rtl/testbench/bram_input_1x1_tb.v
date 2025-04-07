`timescale 1ns / 1ps

module bram_input_1x1_tb();
    // Parameters
    parameter DATA_WIDTH = 8;
    parameter IN_CHANNELS = 3;
    parameter IN_WIDTH = 4;
    parameter IN_HEIGHT = 4;
    parameter DEPTH = IN_WIDTH * IN_HEIGHT * IN_CHANNELS;
    parameter CLK_PERIOD = 10; // 100MHz

    // Signals
    reg clk;
    reg [DATA_WIDTH-1:0] wr_data;
    reg [$clog2(DEPTH)-1:0] wr_addr;
    reg wr_en;
    reg [$clog2(IN_WIDTH*IN_HEIGHT)-1:0] rd_addr;
    reg rd_en;
    wire [DATA_WIDTH*IN_CHANNELS-1:0] rd_data;

    // Instantiate the Unit Under Test (UUT)
    bram_input_1x1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .IN_CHANNELS(IN_CHANNELS),
        .IN_WIDTH(IN_WIDTH),
        .IN_HEIGHT(IN_HEIGHT),
        .DEPTH(DEPTH),
        .OUTPUT_REGISTER("false")  // Test without output register first
    ) uut (
        .rd_data(rd_data),
        .rd_addr(rd_addr),
        .rd_en(rd_en),
        .wr_data(wr_data),
        .wr_addr(wr_addr),
        .wr_en(wr_en),
        .clk(clk)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Dump waveforms
    initial begin
        $dumpfile("bram_input_1x1_tb.vcd");
        $dumpvars(0, bram_input_1x1_tb);
    end

    // Test procedure
    integer i, j, k;
    integer pixel_idx;
    integer error_count = 0;
    
    initial begin
        // Initialize
        wr_data = 0;
        wr_addr = 0;
        wr_en = 0;
        rd_addr = 0;
        rd_en = 0;
        
        // Reset
        #(CLK_PERIOD*2);
        
        // Write test data: Fill the feature map with patterns
        // Format: value = (channel * 100) + (y * 10) + x
        // This way we can easily identify values by position
        wr_en = 1;
        pixel_idx = 0;
        
        for (j = 0; j < IN_HEIGHT; j = j + 1) begin
            for (k = 0; k < IN_WIDTH; k = k + 1) begin
                for (i = 0; i < IN_CHANNELS; i = i + 1) begin
                    // Calculate address: pixel_idx * IN_CHANNELS + i
                    wr_addr = pixel_idx * IN_CHANNELS + i;
                    // Set data value based on channel and position
                    wr_data = (i * 100) + (j * 10) + k;
                    #CLK_PERIOD;
                end
                pixel_idx = pixel_idx + 1;
            end
        end
        wr_en = 0;
        
        // Add some delay
        #(CLK_PERIOD*2);
        
        // Read test: Read each pixel position and verify all channels
        rd_en = 1;
        
        // Test multiple pixel positions
        for (pixel_idx = 0; pixel_idx < IN_WIDTH * IN_HEIGHT; pixel_idx = pixel_idx + 1) begin
            // Set read address to current pixel
            rd_addr = pixel_idx;
            // Calculate expected values for this pixel
            j = pixel_idx / IN_WIDTH;  // row (y)
            k = pixel_idx % IN_WIDTH;  // column (x)
            
            // Wait for read to complete
            #CLK_PERIOD;
            
            // Display read results
            $display("Reading pixel position %d (x=%d, y=%d):", pixel_idx, k, j);
            
            // Check each channel value
            for (i = 0; i < IN_CHANNELS; i = i + 1) begin
                // Extract the value for this channel
                // Format is [(i+1)*DATA_WIDTH-1 -: DATA_WIDTH]
                if (rd_data[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] !== ((i * 100) + (j * 10) + k)) begin
                    $display("ERROR: Channel %d - Expected: %d, Got: %d", 
                             i, 
                             ((i * 100) + (j * 10) + k), 
                             rd_data[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH]);
                    error_count = error_count + 1;
                end else begin
                    $display("Channel %d: Expected: %d, Got: %d ✓", 
                             i, 
                             ((i * 100) + (j * 10) + k), 
                             rd_data[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH]);
                end
            end
        end
        
        // Test with rd_en = 0
        rd_en = 0;
        rd_addr = 0; 
        #CLK_PERIOD;
        $display("\nRead with rd_en=0:");
        $display("Expected: 0, Got: %h", rd_data);
        if (rd_data !== 0) begin
            $display("ERROR: Data should be 0 when rd_en=0");
            error_count = error_count + 1;
        end else begin
            $display("Data is 0 when rd_en=0 ✓");
        end
        
        // Test completed
        if (error_count == 0)
            $display("\nTEST PASSED!");
        else
            $display("\nTEST FAILED with %d errors", error_count);
            
        #(CLK_PERIOD*2);
        $finish;
    end

    // Optional - Test with registered output
    // You can add another instance with OUTPUT_REGISTER="true"
    // and verify the delay behavior

endmodule
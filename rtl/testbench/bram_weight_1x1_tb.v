`timescale 1ns / 1ps

module bram_weight_1x1_tb();
    // Parameters
    parameter DATA_WIDTH = 8;
    parameter IN_CHANNELS = 3;
    parameter OUT_CHANNELS = 4;
    parameter DEPTH = IN_CHANNELS * OUT_CHANNELS;
    parameter CLK_PERIOD = 10; // 100MHz

    // Signals
    reg clk;
    reg [DATA_WIDTH-1:0] wr_data;
    reg [$clog2(DEPTH)-1:0] wr_addr;
    reg wr_en;
    reg [$clog2(OUT_CHANNELS)-1:0] rd_addr;
    reg rd_en;
    wire [DATA_WIDTH*IN_CHANNELS-1:0] rd_data;

    // Instantiate the Unit Under Test (UUT)
    bram_weight_1x1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
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
        $dumpfile("bram_weight_1x1_tb.vcd");
        $dumpvars(0, bram_weight_1x1_tb);
    end

    // Test procedure
    integer i;
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
        
        // Write test data: Each kernel will have increasing values
        // OUT_CHANNEL 0: [10, 11, 12]
        // OUT_CHANNEL 1: [20, 21, 22]
        // OUT_CHANNEL 2: [30, 31, 32]
        // OUT_CHANNEL 3: [40, 41, 42]
        wr_en = 1;
        for (i = 0; i < OUT_CHANNELS; i = i + 1) begin
            // Write IN_CHANNELS values for each output channel
            wr_addr = i * IN_CHANNELS;
            wr_data = (i+1) * 10 + 0; // First value
            #CLK_PERIOD;
            
            wr_addr = i * IN_CHANNELS + 1;
            wr_data = (i+1) * 10 + 1; // Second value
            #CLK_PERIOD;
            
            wr_addr = i * IN_CHANNELS + 2;
            wr_data = (i+1) * 10 + 2; // Third value
            #CLK_PERIOD;
        end
        wr_en = 0;
        
        // Add some delay
        #(CLK_PERIOD*2);
        
        // Read test: Read each output channel kernel and verify
        rd_en = 1;
        
        // Read OUT_CHANNEL 0
        rd_addr = 0;
        #CLK_PERIOD;
        $display("Read OUT_CHANNEL 0:");
        $display("Expected: [12, 11, 10], Got: [%d, %d, %d]", 
                 rd_data[DATA_WIDTH*1-1:DATA_WIDTH*0], 
                 rd_data[DATA_WIDTH*2-1:DATA_WIDTH*1], 
                 rd_data[DATA_WIDTH*3-1:DATA_WIDTH*2]);
        if (rd_data !== {8'd12, 8'd11, 8'd10}) begin
            $display("ERROR: Incorrect data read for OUT_CHANNEL 0");
            error_count = error_count + 1;
        end
        
        // Read OUT_CHANNEL 1
        rd_addr = 1;
        #CLK_PERIOD;
        $display("Read OUT_CHANNEL 1:");
        $display("Expected: [22, 21, 20], Got: [%d, %d, %d]", 
                 rd_data[DATA_WIDTH*1-1:DATA_WIDTH*0], 
                 rd_data[DATA_WIDTH*2-1:DATA_WIDTH*1], 
                 rd_data[DATA_WIDTH*3-1:DATA_WIDTH*2]);
        if (rd_data !== {8'd22, 8'd21, 8'd20}) begin
            $display("ERROR: Incorrect data read for OUT_CHANNEL 1");
            error_count = error_count + 1;
        end
        
        // Read OUT_CHANNEL 2
        rd_addr = 2;
        #CLK_PERIOD;
        $display("Read OUT_CHANNEL 2:");
        $display("Expected: [32, 31, 30], Got: [%d, %d, %d]", 
                 rd_data[DATA_WIDTH*1-1:DATA_WIDTH*0], 
                 rd_data[DATA_WIDTH*2-1:DATA_WIDTH*1], 
                 rd_data[DATA_WIDTH*3-1:DATA_WIDTH*2]);
        if (rd_data !== {8'd32, 8'd31, 8'd30}) begin
            $display("ERROR: Incorrect data read for OUT_CHANNEL 2");
            error_count = error_count + 1;
        end
        
        // Read OUT_CHANNEL 3
        rd_addr = 3;
        #CLK_PERIOD;
        $display("Read OUT_CHANNEL 3:");
        $display("Expected: [42, 41, 40], Got: [%d, %d, %d]", 
                 rd_data[DATA_WIDTH*1-1:DATA_WIDTH*0], 
                 rd_data[DATA_WIDTH*2-1:DATA_WIDTH*1], 
                 rd_data[DATA_WIDTH*3-1:DATA_WIDTH*2]);
        if (rd_data !== {8'd42, 8'd41, 8'd40}) begin
            $display("ERROR: Incorrect data read for OUT_CHANNEL 3");
            error_count = error_count + 1;
        end
        
        // Test with rd_en = 0
        rd_en = 0;
        rd_addr = 0; // Try to read OUT_CHANNEL 0
        #CLK_PERIOD;
        $display("Read with rd_en=0:");
        $display("Expected: 0, Got: %h", rd_data);
        if (rd_data !== 0) begin
            $display("ERROR: Data should be 0 when rd_en=0");
            error_count = error_count + 1;
        end
        
        // Test completed
        if (error_count == 0)
            $display("TEST PASSED!");
        else
            $display("TEST FAILED with %d errors", error_count);
            
        #(CLK_PERIOD*2);
        $finish;
    end
endmodule
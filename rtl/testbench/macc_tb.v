`timescale 1ns / 1ps

module macc_tb();
    // Parameters
    parameter NUM_INPUTS = 8;
    parameter CLK_PERIOD = 10; // 10ns - 100MHz
    
    // Derived parameters
    localparam ADDER_LAYERS = $clog2(NUM_INPUTS);
    localparam OUTPUT_DATA_WIDTH = 16 + ADDER_LAYERS;
    
    // Signals
    reg  [8*NUM_INPUTS-1:0]      i_data_a;
    reg  [8*NUM_INPUTS-1:0]      i_data_b;
    reg                          i_valid;
    reg                          clk;
    reg                          rst_n;
    wire [OUTPUT_DATA_WIDTH-1:0] o_data;
    wire                         o_valid;
    
    // Expected result calculation
    reg signed [31:0] expected_result;
    
    // Variables for testcase
    integer i;
    integer test_count = 0;
    integer error_count = 0;
    integer cycles_to_output;
    
    // DUT instantiation
    macc #(
        .NUM_INPUTS(NUM_INPUTS)
    ) dut (
        .o_data(o_data),
        .o_valid(o_valid),
        .i_data_a(i_data_a),
        .i_data_b(i_data_b),
        .i_valid(i_valid),
        .clk(clk),
        .rst_n(rst_n)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Monitor output
    always @(posedge clk) begin
        if (o_valid) begin
            $display("Output valid, o_data = %d, Expected = %d", 
                     $signed(o_data), expected_result);
            
            if ($signed(o_data) !== expected_result) begin
                $display("ERROR: Mismatch detected! Difference: %d", 
                         $signed(o_data) - expected_result);
                error_count = error_count + 1;
            end else begin
                $display("PASS: Output matches expected value");
            end
        end
    end
    
    // Calculate expected result
    task calculate_expected;
        input [8*NUM_INPUTS-1:0] data_a;
        input [8*NUM_INPUTS-1:0] data_b;
        integer i;
        reg signed [7:0] a, b;
        reg signed [31:0] sum;
        begin
            sum = 0;
            for (i = 0; i < NUM_INPUTS; i = i + 1) begin
                a = data_a[(i+1)*8-1 -: 8]; // Extract each 8-bit value
                b = data_b[(i+1)*8-1 -: 8];
                sum = sum + (a * b);        // Accumulate a*b
            end
            expected_result = sum;
            $display("Calculated expected result: %d", expected_result);
        end
    endtask
    
    // Calculate pipeline delay
    function integer calculate_pipeline_delay;
        input integer num_inputs;
        integer adder_layers;
        begin
            // 1 cycle for multiplication + adder tree depth
            adder_layers = $clog2(num_inputs);
            calculate_pipeline_delay = 1 + adder_layers;
        end
    endfunction
    
    // Main test procedure
    initial begin
        // Initialize
        i_data_a = 0;
        i_data_b = 0;
        i_valid = 0;
        rst_n = 1;
        
        // Calculate pipeline delay
        cycles_to_output = calculate_pipeline_delay(NUM_INPUTS);
        $display("Pipeline depth: %d cycles", cycles_to_output);
        
        // Reset sequence
        @(posedge clk);
        rst_n = 0;
        repeat(3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        
        // Test Case 1: All 1's multiplication
        test_count = test_count + 1;
        $display("\n--- Test Case %0d: All 1's multiplication ---", test_count);
        for (i = 0; i < NUM_INPUTS; i = i + 1) begin
            i_data_a[i*8 +: 8] = 8'd1;
            i_data_b[i*8 +: 8] = 8'd1;
        end
        
        calculate_expected(i_data_a, i_data_b);
        
        i_valid = 1;
        @(posedge clk);
        i_valid = 0;
        
        // Wait for output
        repeat(cycles_to_output) @(posedge clk);
        
        // Test Case 2: Alternating positive/negative values
        test_count = test_count + 1;
        $display("\n--- Test Case %0d: Alternating positive/negative values ---", test_count);
        for (i = 0; i < NUM_INPUTS; i = i + 1) begin
            i_data_a[i*8 +: 8] = (i % 2 == 0) ? 8'd5 : -8'd5;
            i_data_b[i*8 +: 8] = 8'd2;
        end
        
        calculate_expected(i_data_a, i_data_b);
        
        i_valid = 1;
        @(posedge clk);
        i_valid = 0;
        
        // Wait for output
        repeat(cycles_to_output) @(posedge clk);
        
        // Test Case 3: Random values
        test_count = test_count + 1;
        $display("\n--- Test Case %0d: Random values ---", test_count);
        for (i = 0; i < NUM_INPUTS; i = i + 1) begin
            i_data_a[i*8 +: 8] = $random % 256; // Random values -128 to 127
            i_data_b[i*8 +: 8] = $random % 256;
            $display("i_data_a[%0d] = %0d, i_data_b[%0d] = %0d", 
                     i, $signed(i_data_a[i*8 +: 8]), 
                     i, $signed(i_data_b[i*8 +: 8]));
        end
        
        calculate_expected(i_data_a, i_data_b);
        
        i_valid = 1;
        @(posedge clk);
        i_valid = 0;
        
        // Wait for output
        repeat(cycles_to_output) @(posedge clk);
        
        // Test Case 4: Max positive values
        test_count = test_count + 1;
        $display("\n--- Test Case %0d: Max positive values ---", test_count);
        for (i = 0; i < NUM_INPUTS; i = i + 1) begin
            i_data_a[i*8 +: 8] = 8'd127;  // Max positive for 8-bit signed
            i_data_b[i*8 +: 8] = 8'd127;
        end
        
        calculate_expected(i_data_a, i_data_b);
        
        i_valid = 1;
        @(posedge clk);
        i_valid = 0;
        
        // Wait for output
        repeat(cycles_to_output) @(posedge clk);
        
        // Test Case 5: Max negative values
        test_count = test_count + 1;
        $display("\n--- Test Case %0d: Max negative values ---", test_count);
        for (i = 0; i < NUM_INPUTS; i = i + 1) begin
            i_data_a[i*8 +: 8] = -8'd128; // Max negative for 8-bit signed
            i_data_b[i*8 +: 8] = 8'd1;
        end
        
        calculate_expected(i_data_a, i_data_b);
        
        i_valid = 1;
        @(posedge clk);
        i_valid = 0;
        
        // Wait for output and a bit more
        repeat(cycles_to_output + 3) @(posedge clk);
        
        // Display test results
        $display("\n=== Test Results ===");
        $display("Total Tests: %0d", test_count);
        if (error_count == 0) begin
            $display("All tests PASSED!");
        end else begin
            $display("FAILED: %0d errors detected", error_count);
        end
        
        $finish;
    end
    
    // Dump waveforms
    initial begin
        $dumpfile("macc_tb.vcd");
        $dumpvars(0, macc_tb);
    end
    
endmodule
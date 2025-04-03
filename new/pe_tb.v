`timescale 1ns / 1ps

module tb_pe_incha_single();

    // Parameters
    parameter IN_WIDTH    = 3;
    parameter IN_HEIGHT   = 3;
    parameter IN_CHANNEL  = 2;
    parameter OUT_CHANNEL = 4;
    parameter OUTPUT_MODE = "relu";
    parameter KERNEL_0    = 3;
    parameter KERNEL_1    = 3;
    parameter DILATION_0  = 1;
    parameter DILATION_1  = 1;
    parameter PADDING_0   = 1;
    parameter PADDING_1   = 1;
    parameter STRIDE_0    = 1;
    parameter STRIDE_1    = 1;
    
    // Local parameters
    localparam KERNEL_PTS = KERNEL_0 * KERNEL_1;
    localparam OUTPUT_DATA_WIDTH = OUTPUT_MODE == "relu" ? 8 : 16;
    localparam LAYER_SCALE_BASE_ADDR = 32'h1000;  // Address for layer scale
    
    // Clock and reset
    reg clk;
    reg rst_n;
    
    // Module I/O
    wire [OUTPUT_DATA_WIDTH*OUT_CHANNEL-1:0] o_data;
    wire o_valid;
    wire pe_ready;
    wire pe_ack;
    reg [8*IN_CHANNEL*KERNEL_PTS-1:0] i_data;
    reg i_valid;
    reg [15:0] weight_wr_data;
    reg [31:0] weight_wr_addr;
    reg weight_wr_en;
    
    // Signals for internal connections
    wire [31:0] kernel_ram_addr;
    wire kernel_ram_wr_en;
    wire [31:0] bias_wr_addr;
    wire bias_wr_en;
    
    // Define logic for memory address mapping
    assign kernel_ram_addr = weight_wr_addr;
    assign kernel_ram_wr_en = weight_wr_en && (weight_wr_addr < LAYER_SCALE_BASE_ADDR);
    assign bias_wr_addr = weight_wr_addr - LAYER_SCALE_BASE_ADDR;
    assign bias_wr_en = weight_wr_en && (weight_wr_addr >= LAYER_SCALE_BASE_ADDR);
    
    // Instantiate the module under test
    pe_incha_single #(
        .IN_WIDTH(IN_WIDTH),
        .IN_HEIGHT(IN_HEIGHT),
        .IN_CHANNEL(IN_CHANNEL),
        .OUT_CHANNEL(OUT_CHANNEL),
        .OUTPUT_MODE(OUTPUT_MODE),
        .KERNEL_0(KERNEL_0),
        .KERNEL_1(KERNEL_1),
        .DILATION_0(DILATION_0),
        .DILATION_1(DILATION_1),
        .PADDING_0(PADDING_0),
        .PADDING_1(PADDING_1),
        .STRIDE_0(STRIDE_0),
        .STRIDE_1(STRIDE_1)
    ) uut (
        .o_data(o_data),
        .o_valid(o_valid),
        .pe_ready(pe_ready),
        .pe_ack(pe_ack),
        .i_data(i_data),
        .i_valid(i_valid),
        .weight_wr_data(weight_wr_data),
        .weight_wr_addr(weight_wr_addr),
        .weight_wr_en(weight_wr_en),
        .clk(clk),
        .rst_n(rst_n)
    );
    
    // Clock generation - 100MHz
    always begin
        #5 clk = ~clk;
    end
    
    // Test sequence
    integer oc, kp;
    initial begin
        // Initialize signals
        clk = 0;
        rst_n = 0;
        i_data = 0;
        i_valid = 0;
        weight_wr_data = 0;
        weight_wr_addr = 0;
        weight_wr_en = 0;
        
        // Reset system
        #20 rst_n = 1;
        #20;
        
        // Write weights to kernel memory
        for (oc = 0; oc < OUT_CHANNEL; oc = oc + 1) begin
            for (kp = 0; kp < KERNEL_PTS * IN_CHANNEL; kp = kp + 1) begin
                @(posedge clk);
                weight_wr_en = 1;
                weight_wr_addr = oc * KERNEL_PTS * IN_CHANNEL + kp;
                weight_wr_data = kp + 1;  // Simple pattern: 1, 2, 3...
            end
        end
        
        @(posedge clk);
        weight_wr_en = 0;
        #20;
        
        // Write biases to bias memory
        for (oc = 0; oc < OUT_CHANNEL; oc = oc + 1) begin
            @(posedge clk);
            weight_wr_en = 1;
            weight_wr_addr = LAYER_SCALE_BASE_ADDR + oc;
            weight_wr_data = oc * 10;  // Simple pattern: 0, 10, 20, 30
        end
        
        @(posedge clk);
        weight_wr_en = 0;
        #20;
        
        // For dequant/sigmoid modes, write layer scale
        if (OUTPUT_MODE == "dequant" || OUTPUT_MODE == "sigmoid") begin
            @(posedge clk);
            weight_wr_en = 1;
            weight_wr_addr = LAYER_SCALE_BASE_ADDR;
            weight_wr_data = 16'h0100;  // 1.0 in fixed point
            @(posedge clk);
            weight_wr_en = 0;
            #20;
        end
        
        // Generate input data - simple pattern with alternating values
        for (kp = 0; kp < 8*IN_CHANNEL*KERNEL_PTS; kp = kp + 1) begin
            i_data[kp] = (kp % 2) ? 8'h01 : 8'h00;
        end
        i_valid = 1;
        
        // Wait for acknowledgment from PE
        fork
            begin
                wait(pe_ack);
                $display("PE acknowledged input at time %t", $time);
            end
            begin
                #1000 $display("Timeout waiting for pe_ack");
                $finish;
            end
        join_any
        disable fork;
        
        @(posedge clk);
        i_valid = 0;
        
        // Wait for output valid
        fork
            begin
                wait(o_valid);
                $display("Output valid at time %t", $time);
            end
            begin
                #2000 $display("Timeout waiting for output");
                $finish;
            end
        join_any
        disable fork;
        
        // Display output data
        $display("Output data: %h", o_data);
        
        // Run for a few more cycles to observe behavior
        #200;
        
        // End simulation
        $finish;
    end
    
    // Monitor key events
    always @(posedge clk) begin
        if (o_valid) 
            $display("Time: %t, Output data: %h", $time, o_data);
        
        if (pe_ready) 
            $display("Time: %t, PE ready", $time);
    end
    
endmodule
`timescale 1ns / 1ps

module conv_1x1 #(
    parameter DATA_WIDTH    = 8,
    parameter IN_CHANNELS   = 16,
    parameter OUT_CHANNELS  = 32,
    parameter IN_WIDTH      = 28,
    parameter IN_HEIGHT     = 28,
    parameter OUTPUT_REG    = "true",  // Add output registers to improve timing
    parameter RAM_STYLE     = "auto"
)(
    // Control signals
    input                                         clk,
    input                                         rst_n,
    input                                         start,
    output                                        done,
    
    // Input feature map interface
    input                                                       in_valid,
    input      [DATA_WIDTH-1:0]                                 in_data,
    input      [$clog2(IN_WIDTH*IN_HEIGHT*IN_CHANNELS)-1:0]     in_addr,
    input                                                       in_wr_en,
    
    // Weight interface
    input                                                       weight_valid,
    input      [DATA_WIDTH-1:0]                                 weight_data,
    input      [$clog2(IN_CHANNELS*OUT_CHANNELS)-1:0]           weight_addr,
    input                                                       weight_wr_en,
    
    // Output feature map interface
    output     [DATA_WIDTH-1:0]                                 out_data,
    output     [$clog2(IN_WIDTH*IN_HEIGHT*OUT_CHANNELS)-1:0]    out_addr,
    output                                                      out_valid,
    input                                                       out_ready
);

    // FSM states
    localparam IDLE         = 3'd0;
    localparam LOAD_WEIGHTS = 3'd1;
    localparam COMPUTE      = 3'd2;
    localparam WRITE_OUTPUT = 3'd3;
    localparam DONE         = 3'd4;

    // State registers
    reg [2:0] state, next_state;
    
    // Counters for addressing
    reg [$clog2(IN_WIDTH*IN_HEIGHT):0] pixel_counter;
    reg [$clog2(OUT_CHANNELS):0] out_channel_counter;
    
    // Internal control signals
    reg  input_rd_en;
    reg  weight_rd_en;
    reg  compute_en;
    wire compute_done;
    
    // Address signals
    reg [$clog2(IN_WIDTH*IN_HEIGHT)-1:0] input_rd_addr;
    reg [$clog2(OUT_CHANNELS)-1:0] weight_rd_addr;
    
    // Data paths
    wire [DATA_WIDTH*IN_CHANNELS-1:0] input_data;
    wire [DATA_WIDTH*IN_CHANNELS-1:0] weight_data_channels;
    wire [16+$clog2(IN_CHANNELS)-1:0] macc_result;
    wire macc_valid;
    
    // Input Buffer
    bram_input_1x1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .IN_CHANNELS(IN_CHANNELS),
        .IN_WIDTH(IN_WIDTH),
        .IN_HEIGHT(IN_HEIGHT),
        .RAM_STYLE(RAM_STYLE),
        .OUTPUT_REGISTER(OUTPUT_REG)
    ) u_input_bram (
        .rd_data(input_data),
        .rd_addr(input_rd_addr),
        .rd_en(input_rd_en),
        .wr_data(in_data),
        .wr_addr(in_addr),
        .wr_en(in_wr_en),
        .clk(clk)
    );
    
    // Weight Buffer
    bram_weight_1x1 #(
        .DATA_WIDTH(DATA_WIDTH),
        .IN_CHANNELS(IN_CHANNELS),
        .OUT_CHANNELS(OUT_CHANNELS),
        .RAM_STYLE(RAM_STYLE),
        .OUTPUT_REGISTER(OUTPUT_REG)
    ) u_weight_bram (
        .rd_data(weight_data_channels),
        .rd_addr(weight_rd_addr),
        .rd_en(weight_rd_en),
        .wr_data(weight_data),
        .wr_addr(weight_addr),
        .wr_en(weight_wr_en),
        .clk(clk)
    );
    
    // Compute Unit (MACC)
    macc #(
        .NUM_INPUTS(IN_CHANNELS)
    ) u_macc (
        .o_data(macc_result),
        .o_valid(macc_valid),
        .i_data_a(input_data),  // Input feature map channels for one pixel
        .i_data_b(weight_data_channels), // Weights for one output channel
        .i_valid(compute_en),
        .clk(clk),
        .rst_n(rst_n)
    );
    
    // Output buffer (can be expanded if needed)
    reg [DATA_WIDTH-1:0] output_buffer [0:IN_WIDTH*IN_HEIGHT*OUT_CHANNELS-1];
    reg [$clog2(IN_WIDTH*IN_HEIGHT*OUT_CHANNELS)-1:0] out_wr_addr;
    
    // FSM Logic
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n)
            state <= IDLE;
        else
            state <= next_state;
    end
    
    // Next state logic
    always @(*) begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start)
                    next_state = COMPUTE;
            end
            
            COMPUTE: begin
                if (pixel_counter == IN_WIDTH*IN_HEIGHT-1 && 
                    out_channel_counter == OUT_CHANNELS-1 && 
                    compute_done)
                    next_state = DONE;
            end
            
            DONE: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // Control logic
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            pixel_counter <= 0;
            out_channel_counter <= 0;
            input_rd_en <= 0;
            weight_rd_en <= 0;
            compute_en <= 0;
            input_rd_addr <= 0;
            weight_rd_addr <= 0;
            out_wr_addr <= 0;
        end
        else begin
            case (state)
                IDLE: begin
                    pixel_counter <= 0;
                    out_channel_counter <= 0;
                    input_rd_en <= 0;
                    weight_rd_en <= 0;
                    compute_en <= 0;
                end
                
                COMPUTE: begin
                    // Set read enables
                    input_rd_en <= 1;
                    weight_rd_en <= 1;
                    
                    // Set addresses
                    input_rd_addr <= pixel_counter;
                    weight_rd_addr <= out_channel_counter;
                    
                    // Enable computation after addresses are valid
                    compute_en <= input_rd_en && weight_rd_en;
                    
                    // Update counters based on completion
                    if (compute_done) begin
                        if (out_channel_counter == OUT_CHANNELS-1) begin
                            out_channel_counter <= 0;
                            if (pixel_counter == IN_WIDTH*IN_HEIGHT-1)
                                pixel_counter <= 0;
                            else
                                pixel_counter <= pixel_counter + 1;
                        end
                        else begin
                            out_channel_counter <= out_channel_counter + 1;
                        end
                    end
                end
                
                default: begin
                    input_rd_en <= 0;
                    weight_rd_en <= 0;
                    compute_en <= 0;
                end
            endcase
        end
    end
    
    // Output writing logic
    always @(posedge clk) begin
        if (macc_valid) begin
            // Calculate output address
            out_wr_addr <= pixel_counter * OUT_CHANNELS + out_channel_counter;
            
            // Store result (with appropriate scaling/quantization)
            output_buffer[out_wr_addr] <= macc_result[16+$clog2(IN_CHANNELS)-1 -: DATA_WIDTH];
        end
    end
    
    // Output interface
    assign out_data = output_buffer[out_addr];
    assign out_valid = (state == DONE);
    assign done = (state == DONE);
    assign compute_done = macc_valid; // Simplification, might need adjustment

endmodule
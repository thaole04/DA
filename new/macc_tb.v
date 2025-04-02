`timescale 1ns / 1ps

module macc_tb();
    // Thông số cấu hình
    parameter NUM_INPUTS = 9; // Sử dụng 9 đầu vào để dễ theo dõi
    parameter CLK_PERIOD = 10; // Clock 10ns = 100MHz
    
    // Tính toán các thông số nội bộ giống như trong module
    localparam ADDER_LAYERS = $clog2(NUM_INPUTS);
    localparam OUTPUT_DATA_WIDTH = 16 + ADDER_LAYERS;
    
    // Tín hiệu kết nối với DUT
    reg clk;
    reg rst_n;
    reg [8*NUM_INPUTS-1:0] i_data_a;
    reg [8*NUM_INPUTS-1:0] i_data_b;
    reg i_valid;
    
    wire [OUTPUT_DATA_WIDTH-1:0] o_data;
    wire o_valid;
    
    // Biến lưu trữ kết quả mong đợi
    reg signed [OUTPUT_DATA_WIDTH-1:0] expected_result;
    integer errors;
    
    // Khởi tạo DUT (Device Under Test)
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
    
    // Tạo clock
    always begin
        #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Tính toán kết quả mong đợi dựa trên đầu vào
    function signed [OUTPUT_DATA_WIDTH-1:0] calculate_expected_result;
        input [8*NUM_INPUTS-1:0] data_a;
        input [8*NUM_INPUTS-1:0] data_b;
        
        integer i;
        reg signed [7:0] a_val;
        reg signed [7:0] b_val;
        reg signed [31:0] sum; // Đủ lớn để tránh tràn số
        
        begin
            sum = 0;
            for (i = 0; i < NUM_INPUTS; i = i + 1) begin
                // Sử dụng phép trích xuất bit truyền thống
                a_val = data_a[(i*8) +: 8];
                b_val = data_b[(i*8) +: 8];
                sum = sum + (a_val * b_val);
            end
            calculate_expected_result = sum[OUTPUT_DATA_WIDTH-1:0];
        end
    endfunction
    
    // Tác vụ để tạo một test case
    task run_test_case;
        input [8*NUM_INPUTS-1:0] data_a;
        input [8*NUM_INPUTS-1:0] data_b;
        input [32*8-1:0] test_name; // Sử dụng mảng thay vì string
        
        begin
            // Thiết lập đầu vào
            i_data_a = data_a;
            i_data_b = data_b;
            i_valid = 1'b1;
            
            // Tính kết quả mong đợi
            expected_result = calculate_expected_result(data_a, data_b);
            
            // Chờ một chu kỳ để dữ liệu được xử lý
            @(posedge clk);
            #1; // Chờ một chút
            i_valid = 1'b0;
            
            // Chờ kết quả hợp lệ
            wait(o_valid);
            #1; // Chờ một chút để kết quả ổn định
            
            // Kiểm tra kết quả
            $display("Test: %s", test_name);
            $display("Expected: %d, Got: %d", expected_result, $signed(o_data));
            
            if ($signed(o_data) !== expected_result) begin
                $display("ERROR: Kết quả không khớp!");
                errors = errors + 1;
            end else begin
                $display("PASS: Kết quả khớp!");
            end
            
            // Chờ thêm một chu kỳ
            @(posedge clk);
            #1;
        end
    endtask
    
    // Test procedure
    initial begin
        // Khởi tạo
        clk = 0;
        rst_n = 0;
        i_valid = 0;
        i_data_a = 0;
        i_data_b = 0;
        errors = 0;
        
        // Reset
        #(CLK_PERIOD*2);
        rst_n = 1;
        #(CLK_PERIOD);
        
        // Test 1: Tất cả số dương
        run_test_case(
            {8'd1, 8'd2, 8'd3, 8'd4, 8'd5, 8'd6, 8'd7, 8'd8, 8'd9},
            {8'd9, 8'd8, 8'd7, 8'd6, 8'd5, 8'd4, 8'd3, 8'd2, 8'd1},
            "Test_1_All_Positive"
        );
        
        // Test 2: Tất cả số âm
        run_test_case(
            {-8'd1, -8'd2, -8'd3, -8'd4, -8'd5, -8'd6, -8'd7, -8'd8, -8'd9},
            {-8'd9, -8'd8, -8'd7, -8'd6, -8'd5, -8'd4, -8'd3, -8'd2, -8'd1},
            "Test_2_All_Negative"
        );
        
        // Test 3: Kết hợp số dương và số âm
        run_test_case(
            {8'd10, -8'd15, 8'd20, -8'd25, 8'd30, -8'd35, 8'd40, -8'd45, 8'd50},
            {-8'd5, 8'd10, -8'd15, 8'd20, -8'd25, 8'd30, -8'd35, 8'd40, -8'd45},
            "Test_3_Mixed_Values"
        );
        
        // Test 4: Số không
        run_test_case(
            {8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0, 8'd0},
            {8'd10, 8'd20, 8'd30, 8'd40, 8'd50, 8'd60, 8'd70, 8'd80, 8'd90},
            "Test_4_Zeros_in_A"
        );
        
        // Test 5: Giá trị cực đại
        run_test_case(
            {8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127},
            {8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127, 8'd127},
            "Test_5_Max_Values"
        );
        
        // Test 6: Giá trị cực tiểu
        run_test_case(
            {-8'd128, -8'd128, -8'd128, -8'd128, -8'd128, -8'd128, -8'd128, -8'd128, -8'd128},
            {-8'd128, -8'd128, -8'd128, -8'd128, -8'd128, -8'd128, -8'd128, -8'd128, -8'd128},
            "Test_6_Min_Values"
        );
        
        // Test 7: Chỉ một giá trị khác 0
        run_test_case(
            {8'd0, 8'd0, 8'd0, 8'd0, 8'd100, 8'd0, 8'd0, 8'd0, 8'd0},
            {8'd0, 8'd0, 8'd0, 8'd0, 8'd100, 8'd0, 8'd0, 8'd0, 8'd0},
            "Test_7_Single_Value"
        );

        // Test 8: Nhân số khác dấu
        run_test_case(
            {8'd1, 8'd2, 8'd3, 8'd4, 8'd5, 8'd6, 8'd7, 8'd8, 8'd9},
            {-8'd1, -8'd2, -8'd3, -8'd4, -8'd5, -8'd6, -8'd7, -8'd8, -8'd9},
            "Test_8_Negative_Multiplication"
        );
        
        // Hiển thị kết quả tổng thể
        #(CLK_PERIOD*2);
        if (errors == 0)
            $display("\nTẤT CẢ TESTS ĐÃ PASS!");
        else
            $display("\nCÓ %d TESTS THẤT BẠI!", errors);
        
        #(CLK_PERIOD*5);
        $finish;
    end
    
    // Tạo file VCD để quan sát sóng
    initial begin
        $dumpfile("macc_tb.vcd");
        $dumpvars(0, macc_tb);
    end

endmodule